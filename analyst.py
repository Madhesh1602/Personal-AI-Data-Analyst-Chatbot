import io
import tempfile
from pathlib import Path
from scipy import stats
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import textwrap
import sys
import os
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langfuse.langchain import CallbackHandler


# ----------------- Load data -----------------
def _looks_like_csv(raw_bytes: bytes) -> bool:
    try:
        sample = raw_bytes[:2048].decode(errors="ignore")
    except Exception:
        return False

    # very light heuristic: delimiter + multiple lines
    delimiters = [",", ";", "\t"]
    lines = sample.splitlines()
    if len(lines) < 2:
        return False

    return any(d in lines[0] for d in delimiters)

def load_data(file_or_path) -> pd.DataFrame:
  """
  Accepts Streamlit UploadedFile, path string/Path, or file-like object.
  Returns pandas DataFrame.
  """
  if isinstance(file_or_path, (str, Path)):
      p = Path(file_or_path)
      suffix = p.suffix.lower()

      if suffix == ".csv":
          return pd.read_csv(p)
      if suffix in {".xls", ".xlsx"}:
          return pd.read_excel(p)
      if suffix == ".json":
          return pd.read_json(p)

      # fallback
      return pd.read_csv(p)

  # Streamlit UploadedFile or file-like
  name = getattr(file_or_path, "name", "")
  suffix = Path(name).suffix.lower()
  raw = file_or_path.read()

  if isinstance(raw, str):
    raw = raw.encode("utf-8")

  bio = io.BytesIO(raw)

  if suffix == ".csv" or (not suffix and _looks_like_csv(raw)):
    bio.seek(0)
    return pd.read_csv(bio)

  if suffix in {".xls", ".xlsx"}:
    bio.seek(0)
    return pd.read_excel(bio)

  if suffix == ".json":
    bio.seek(0)
    return pd.read_json(bio)

  # final fallback
  bio.seek(0)
  try:
    return pd.read_csv(bio)
  except Exception:
    bio.seek(0)
    return pd.read_json(bio)
  
def detect_column_types(df: pd.DataFrame) -> dict:
  """
  Conservative column type inference.
  Returns stable schema usable by downstream logic.
  """
  schema = {
      "numeric": [],
      "datetime": [],
      "categorical": [],
      "identifier": []
  }

  for c in df.columns:
      series = df[c]

      # Numeric
      if pd.api.types.is_numeric_dtype(series):
          schema["numeric"].append(c)
          continue

      # Datetime (strict check only)
      if pd.api.types.is_datetime64_any_dtype(series):
          schema["datetime"].append(c)
          continue

      # Identifier heuristic (high cardinality strings)
      nunique = series.nunique(dropna=True)
      if series.dtype == object and nunique > max(50, 0.8 * len(series)):
          schema["identifier"].append(c)
          continue

      # Categorical (low cardinality)
      if nunique <= 50:
          schema["categorical"].append(c)
          continue

  return schema


def flatten_inferred_schema(inferred: dict) -> dict:
    flat = {}
    for col_type, cols in inferred.items():
        for c in cols:
            flat[c] = {
                "type": col_type,
                "source": "inferred"
            }
    return flat

def normalize_declared_schema(schema_df: pd.DataFrame) -> dict:
    declared = {}

    for _, row in schema_df.iterrows():
        col_raw = row["column_name"]
        col_key = col_raw.strip().lower()   # 🔴 normalize key

        declared[col_key] = {
            "original_name": col_raw.strip(),
            "description": str(row.get("description", "")).strip(),
            "type": str(row.get("type", "")).strip().lower() or None,
            "source": "declared"
        }

    return declared


def merge_schema(df: pd.DataFrame, inferred: dict, schema_df: pd.DataFrame) -> dict:
  inferred_flat = flatten_inferred_schema(inferred)
  declared = normalize_declared_schema(schema_df)

  unified = {}

  for col in df.columns:
    col_key = col.strip().lower()   # 🔴 normalize key
    
    if col_key in declared:
      meta = declared[col_key]
      declared_type = map_type(meta.get("type"))
      inferred_type = map_type(inferred_flat.get(col, {}).get("type"))
      unified[col] = {
          "description": meta["description"],
          "type": declared_type or inferred_type,
          "source": "declared"
      }
    else:
      # Fall back to inferred schema
      inferred_meta = inferred_flat.get(col, {})
      unified[col] = {
        "description": "",
        "type": inferred_meta.get("type", "unknown"),
        "source": "inferred"
      }

  return unified

def validate_schema(unified_schema: dict):
  warnings = []
  infos = []

  for col, meta in unified_schema.items():
      if meta["type"] == "unknown":
          warnings.append(f"Column '{col}' has unknown type")
      if meta["source"] == "inferred" and not meta["description"]:
          infos.append(f"No description provided for column '{col}'")

  return warnings, infos

TYPE_MAP = {
    "string": "categorical",
    "category": "categorical",
    "numeric": "numeric",
    "number": "numeric",
    "identifier": "identifier",
    "id": "identifier",
    "int": "numeric",
    "float": "numeric"
}

def map_type(raw_type):
  """
  Normalize a raw type string to the canonical type using TYPE_MAP.
  Returns None if raw_type is falsy, otherwise returns the mapped type.
  """
  if raw_type is None:
    return None
  rt = str(raw_type).strip().lower()
  return TYPE_MAP.get(rt, rt)


def suggest_prompts(unified_schema: dict, max_suggestions: int = 8):
  """
  Return a list of helpful, schema-safe prompt suggestions.
  Derived ONLY from unified schema (declared + inferred).
  """
  numeric = [c for c, m in unified_schema.items() if m["type"] == "numeric"]
  datetime = [c for c, m in unified_schema.items() if m["type"] == "datetime"]
  categorical = [c for c, m in unified_schema.items() if m["type"] == "categorical"]
  target = [c for c, m in unified_schema.items() if m["type"] == "target"]

  suggestions = []

  # Dataset summary (always safe)
  suggestions.append(
      "Summarize the dataset in 5 bullet points (rows, columns, missing values, numeric columns, key categories)."
  )

  # Target-based questions (high value)
  if target:
    t = target[0]
    suggestions.append(f"How many records fall into each value of '{t}'?")
    if categorical:
      suggestions.append(f"Show '{t}' distribution by '{categorical[0]}'.")
    if numeric:
      suggestions.append(f"Compare average '{numeric[0]}' for each '{t}' category.")

  # Categorical exploration
  if categorical:
    c = categorical[0]
    suggestions.append(f"Show the top 10 counts for the categorical column '{c}'.")

  # Numeric exploration
  if numeric:
    n = numeric[0]
    suggestions.append("Show summary statistics for numeric columns.")
    suggestions.append(f"Create a histogram of the numeric column '{n}'.")
    suggestions.append(f"Show the top 10 rows sorted by '{n}' descending.")

    if len(numeric) >= 2:
      suggestions.append(
          f"Create a scatter plot comparing '{numeric[0]}' (x) vs '{numeric[1]}' (y)."
      )

  # Time series (only if explicitly datetime)
  if datetime and numeric:
    suggestions.append(
        f"Create a monthly time series of '{numeric[0]}' using the datetime column '{datetime[0]}'."
      )

  # Correlation (numeric only)
  if len(numeric) >= 2:
    suggestions.append("Show the correlation matrix heatmap for numeric columns.")

  # Anomaly detection (guarded)
  if numeric:
    suggestions.append(
        "Find rows that look like anomalies using z-score > 3 on numeric columns and show top 20."
    )

  return suggestions[:max_suggestions]

INTENTS = {
    "SUMMARY_DATASET": {
        "patterns": ["summarize"],
        "rules": {
            "required_types": [],
            "min_columns": 0
        }
    },
    "TOP_COUNTS": {
        "patterns": ["top", "count"],
        "rules": {
            "required_types": ["categorical"],
            "min_columns": 1
        }
    },
    "SUMMARY_STATS": {
        "patterns": ["summary statistics", "describe"],
        "rules": {
            "required_types": ["numeric"],
            "min_columns": 1
        }
    },
    "HISTOGRAM": {
        "patterns": ["histogram"],
        "rules": {
            "required_types": ["numeric"],
            "min_columns": 1
        }
    },
    "SCATTER": {
        "patterns": ["scatter", "vs"],
        "rules": {
            "required_types": ["numeric"],
            "min_columns": 2
        }
    },
    "TIME_SERIES": {
        "patterns": ["time series", "monthly"],
        "rules": {
            "required_types": ["datetime", "numeric"],
            "min_columns": 2
        }
    },
    "CORRELATION": {
        "patterns": ["correlation", "heatmap"],
        "rules": {
            "required_types": ["numeric"],
            "min_columns": 2
        }
    },
    "ANOMALIES_ZSCORE": {
        "patterns": ["anomal", "z-score"],
        "rules": {
            "required_types": ["numeric"],
            "min_columns": 1
        }
    }
}

def detect_intent(prompt: str) -> str | None:
    """
    Detect analysis intent based on known patterns.
    Returns intent name or None.
    """
    p = prompt.strip().lower()

    for intent, meta in INTENTS.items():
        for pat in meta["patterns"]:
            if pat in p:
                return intent

    return None


def enforce_schema(intent: str, unified_schema: dict) -> tuple[bool, str]:
    """
    Check whether the intent is allowed under the current schema.
    """
    if intent not in INTENTS:
        return False, "Unsupported analysis type."

    rules = INTENTS[intent]["rules"]

    # Count columns by type
    type_counts = {}
    for meta in unified_schema.values():
        t = meta["type"]
        type_counts[t] = type_counts.get(t, 0) + 1

    # Required type check
    for req in rules["required_types"]:
        if type_counts.get(req, 0) == 0:
            return False, f"No columns of type '{req}' available for this analysis."

    # Minimum column check
    total = sum(type_counts.get(t, 0) for t in rules["required_types"])
    if total < rules["min_columns"]:
        return False, "Not enough compatible columns for this analysis."

    return True, ""

INTENT_RULES = {
    "SURVIVAL_COUNT": {
        "required_types": ["target"],
        "min_columns": 1
    },
    "SUMMARY_DATASET": {
        "required_types": [],
        "min_columns": 0
    },
    "ANOMALIES_ZSCORE": {
        "required_types": ["numeric"],
        "min_columns": 1
    },
    "HISTOGRAM": {
        "required_types": ["numeric"],
        "min_columns": 1
    },
    "SCATTER": {
        "required_types": ["numeric"],
        "min_columns": 2
    },
    "CORRELATION": {
        "required_types": ["numeric"],
        "min_columns": 2
    },
    "TIME_SERIES": {
        "required_types": ["datetime", "numeric"],
        "min_columns": 2
    }
}

def prompt_to_code(intent: str, prompt: str, df: pd.DataFrame):
    """
    Generate runnable python code for a PRE-VALIDATED intent.
    Assumes schema enforcement has already passed.
    """
    p = prompt.strip().lower()

    if intent == "SUMMARY_DATASET":
        return textwrap.dedent("""
            info = []
            info.append(f"Rows: {len(df)}, Columns: {len(df.columns)}")
            info.append("Column types: " + ", ".join([f"{c}:{str(df[c].dtype)[:10]}" for c in df.columns[:10]]))
            miss = df.isnull().sum().sort_values(ascending=False).head(10)
            info.append("Top missing: " + ", ".join([f"{idx}:{val}" for idx,val in miss.items() if val>0]))
            numeric = df.select_dtypes(include=['number']).columns.tolist()
            info.append(f"Numeric columns count: {len(numeric)}")
            result = "\\n".join(["- "+i for i in info])
        """)

    if intent == "TOP_COUNTS":
        import re
        m = re.search(r"'([^']+)'", prompt)
        col = m.group(1) if m else None
        if col:
            return textwrap.dedent(f"""
                result = df['{col}'].value_counts(dropna=False).head(10).reset_index()
                result.columns = ['value','count']
            """)

    if intent == "SUMMARY_STATS":
        return textwrap.dedent("""
            result = df.select_dtypes(include=['number']).describe().T
        """)

    if intent == "HISTOGRAM":
        import re
        m = re.search(r"'([^']+)'", prompt)
        col = m.group(1) if m else None
        if col:
            return textwrap.dedent(f"""
                plt.figure(figsize=(6,4))
                df['{col}'].dropna().astype(float).hist(bins=30)
                plt.title('Histogram of {col}')
                plt.xlabel('{col}')
                plt.ylabel('count')
                result_img_path = None
            """)

    if intent == "SCATTER":
        import re
        m = re.search(r"'([^']+)' \\(x\\) vs '([^']+)' \\(y\\)", prompt)
        if m:
            xcol, ycol = m.group(1), m.group(2)
            return textwrap.dedent(f"""
                plt.figure(figsize=(6,4))
                df.plot.scatter(x='{xcol}', y='{ycol}')
                plt.title('{ycol} vs {xcol}')
                result_img_path = None
            """)

    if intent == "TOP_SORTED":
        import re
        m = re.search(r"by '([^']+)'", prompt)
        if m:
            col = m.group(1)
            return textwrap.dedent(f"""
                result = df.sort_values('{col}', ascending=False).head(10).reset_index(drop=True)
            """)
        
    if intent == "TIME_SERIES":
        import re
        m = re.search(r"sum of '([^']+)' using the datetime column '([^']+)'", prompt)
        if m:
            ag, dcol = m.group(1), m.group(2)
            return textwrap.dedent(f"""
                tmp = df.copy()
                tmp['{dcol}'] = pd.to_datetime(tmp['{dcol}'], errors='coerce')
                res = tmp.dropna(subset=['{dcol}'])
                res = res.set_index('{dcol}').resample('M')['{ag}'].sum().reset_index()
                result = res
            """)
  
    if intent == "CORRELATION":
        return textwrap.dedent("""
            corr = df.select_dtypes(include=['number']).corr()
            plt.figure(figsize=(6,5))
            plt.imshow(corr, cmap='viridis', aspect='auto')
            plt.colorbar()
            plt.xticks(range(len(corr)), corr.columns, rotation=90)
            plt.yticks(range(len(corr)), corr.columns)
            plt.title('Correlation matrix')
            result_img_path = None
        """)

    if intent == "ANOMALIES_ZSCORE":
        return textwrap.dedent("""
            num = df.select_dtypes(include=['number'])
            if num.shape[1] == 0:
                result = pd.DataFrame()
            else:
                z = np.abs(stats.zscore(num, axis=0, nan_policy='omit'))
                z_df = pd.DataFrame(z, columns=num.columns, index=df.index)
                
                mask_abs = z_df.max(axis=1)
                mask = mask_abs > 3
                
                df_anom = df.loc[mask].copy()
                df_anom["max_abs_zscore"] = mask_abs[mask]
                
                print(f"Found {mask.sum()} anomalous rows (z-score > 3). Showing top 20 by max_abs_zscore.")
                               
                result = (df_anom.sort_values("max_abs_zscore", ascending=False).head(20).reset_index(drop=True))
        """)

    # Fallback (should not happen if routing is correct)
    return None


def run_code(df: pd.DataFrame, code: str):
  SAFE_BUILTINS = {
    #basic types
    "int": int,
    "float": float,
    "string": str,
    "str": str,
    "bool": bool,
    "list": list,
    "dict": dict,
    "set": set,
    "percentage": float,
    "currency": str,
    # exceptions (IMPORTANT)
    "ValueError": ValueError,
    "TypeError": TypeError,
    "KeyError": KeyError,
    #basic functions
    "len": len,
    "range": range,
    "enumerate": enumerate,
    "any": any,
    "all": all,
    "min": min,
    "max": max,
    "sum": sum,
    "abs": abs,
    "print": print,
    #imports
    "__import__": __import__
  }

  local_ns = {"df": df}
  old_stdout = sys.stdout
  stdout_buf = io.StringIO()
  sys.stdout = stdout_buf

  try:
      exec(code, {"__builtins__": SAFE_BUILTINS, "pd": pd, "np": np, "plt": plt, "stats": stats,},local_ns,)


      if "result_img_path" in local_ns and local_ns["result_img_path"]:
          return {"type": "image", "path": local_ns["result_img_path"]}

      if plt.get_fignums():
          with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as f:
              plt.savefig(f.name, bbox_inches="tight", dpi=150)
              return {"type": "image", "path": f.name}

      if "result" in local_ns:
          res = local_ns["result"]
          print("Result Type:", type(res))
          if isinstance(res, pd.DataFrame):
            return {"type": "dataframe", "df": res}
          else:
            return {"type": "text", "output": str(res)}

      out = stdout_buf.getvalue().strip()
      return {"type": "text", "output": out or "Execution finished. No result produced."}

  except Exception as e:
      return {"type": "text", "output": f"Execution error: {e}"}

  finally:
      sys.stdout = old_stdout
      plt.close("all")


def ask_llm(prompt: str, model: str = "gpt-5-mini-2025-08-07", timeout: int = 60) -> str:
  """
  Send prompt to LLM API gateway. Returns the response text.
  Uses TokenFetcher for auth and Langfuse for observability.
  """
  
  try:
      # 1. Setup Environment Credentials
      client_id = os.getenv("LLM_CLIENT_ID")
      client_secret = os.getenv("LLM_CLIENT_SECRET")


      if not client_id or not client_secret:
          return "[LLM-error] Missing LLM_CLIENT_ID or LLM_CLIENT_SECRET"

      # 2. Initialize Authentication
      token_fetcher = TokenFetcher() 

      # 3. Initialize Observability (Langfuse)
      # The import path was fixed to use langfuse.langchain
      # use the CallbackHandler class that we imported above
      langfuse_handler = CallbackHandler()


      # 4. Initialize the LLM Client
      llm = ChatOpenAI(
          model = model,
          base_url = "https://api.pivpn.core..com/llmapi/api/v1",
          api_key = token_fetcher.token, 
          temperature = 0,
          callbacks = [langfuse_handler],
          request_timeout = timeout
      )
      
      # 5. Invoke the model
      response = llm.invoke([HumanMessage(content=prompt)])
      return response.content

  except ImportError as e:
      return f"[LLM-error] Missing dependency: {e}. Try: pip install langfuse langchain-openai"
  except Exception as e:
      return f"[LLM-failed] {str(e)}"