import streamlit as st
import pandas as pd
from dotenv import load_dotenv
# Ensure 'ask_llm' in 'analyst.py' is the updated version using LLM/LangChain
from analyst import load_data, detect_column_types, merge_schema, validate_schema, suggest_prompts, detect_intent, enforce_schema, prompt_to_code, run_code, ask_llm

st.set_page_config(page_title = "Personal AI Data Analyst", layout="wide")
st.title("🧠  Personal AI Data Analyst — Interactive Dashboard")

st.sidebar.header("Settings")

# Updated Sidebar for LLM
use_llm = st.sidebar.checkbox("Use LLM for custom prompts", value=True)
llm_model = st.sidebar.text_input("Model Name", value="gpt-5-mini-2025-08-07")
st.sidebar.markdown("This tool uses the internal LLM API gateway.")

# Load environment variables from .env file
load_dotenv()

uploaded_files = st.file_uploader("Upload CSV, Excel, or JSON along with a file containing column descriptions", type=["csv","xls","xlsx","json"], accept_multiple_files = True)

if not uploaded_files or len(uploaded_files) < 2:
    st.info("Please upload BOTH files: the data file and the column description CSV.")
    st.stop()

if len(uploaded_files) > 2:
    st.error("Please upload only two files: one data file and one column description CSV.")
    st.stop()

data_file = None
schema_file = None

for f in uploaded_files:
  if f.name.lower().endswith(".csv"):
    try:
      preview = pd.read_csv(f, nrows=5)
      f.seek(0)  # Reset file pointer after reading
      
      if "column_name" in preview.columns:
          schema_file = f
      else:
          data_file = f
    except Exception:
      data_file = f
  else:
      data_file = f

if data_file is None or schema_file is None:
  st.error(
      "Could not identify both files.\n\n"
      "Ensure:\n"
      "- Column description file is a CSV with a 'column_name' column\n"
      "- Data file is CSV / Excel / JSON"
  )
  st.stop()

# Load data
try:
    df = load_data(data_file)
except Exception as e:
    st.error(f"Failed to load file: {e}")
    st.stop()

# Load schema
try:
    schema_df = load_data(schema_file)
except Exception as e:
    st.error(f"Failed to load file: {e}")
    st.stop()

# Validate schema
required_cols = {"column_name", "description"}

if not required_cols.issubset(schema_df.columns):
  st.error(
      "Column description CSV must contain columns: "
      "'column_name' and 'description'"
  )
  st.stop()

df_cols = {c.lower() for c in df.columns}
schema_cols = {c.lower() for c in schema_df["column_name"]}

missing = df_cols - schema_cols

if missing:
  st.warning(f"No descriptions provided for columns: {', '.join(sorted(missing))}")


inferred_schema = detect_column_types(df)

UNIFIED_SCHEMA = merge_schema(df, inferred_schema, schema_df)

warnings, infos = validate_schema(UNIFIED_SCHEMA)

for w in warnings:
  st.warning(w)

for i in infos:
  st.info(i)

st.success("Schema loaded and validated")
with st.expander("Advanced: View unified schema"):
  st.json(UNIFIED_SCHEMA, expanded=False)

# Create metadata dictionary
COLUMN_METADATA = (
    schema_df
    .set_index("column_name")
    .to_dict(orient="index")
)

st.success("Data and column descriptions loaded successfully")
with st.expander("Preview data (first 100 rows)"):
    st.dataframe(df.head(100))

# Generate schema-aware suggestions
suggestions = suggest_prompts(UNIFIED_SCHEMA)
st.markdown("## Suggested analyses (pick one or write your own)")
col1, col2 = st.columns([3, 1])

with col1:
  selected = st.selectbox(
      "Choose a suggested prompt",
      options=suggestions,
      index=0 if suggestions else None,
      disabled=not suggestions
  )

  custom = st.text_area(
      "Or write a custom prompt (leave blank to use the selected suggestion)",
      height=80,
      placeholder="Ask a question about your data..."
  )

with col2:
  st.markdown("**Quick actions**")

  if st.button("Refresh suggestions"):
    st.experimental_rerun()


# Determine final prompt
final_prompt = custom.strip() if custom and custom.strip() else selected

if not final_prompt:
  st.info("Please select a suggested analysis or write a custom prompt.")
  st.stop()

st.markdown("### Final prompt")
st.write(final_prompt)

# Run button
if st.button("Run analysis"):
    st.session_state["run"] = True

if st.session_state.get("run"):
    with st.spinner("Running..."):
        # First try deterministic conversion (regex/rules defined in analyst.py)
        intent = detect_intent(final_prompt)

        if intent is not None:
          allowed, reason = enforce_schema(intent, UNIFIED_SCHEMA)
          if not allowed:
            st.warning(f"Cannot run this analysis: {reason}")
            st.stop()


        code = prompt_to_code(intent, final_prompt, df)
        
        if code:
            res = run_code(df, code)
        else:
            # No deterministic code found. Fallback to LLM.
            if use_llm:
                # Craft a system instruction that asks for python in a ```python``` block
                # Note: We pass this as a single string to ask_llm, which wraps it in a HumanMessage.
                system = (
                    "You are a helpful data analyst and will respond with Python code only.\n"
                    "You must return code inside a ```python ... ``` block. The DataFrame is named `df`.\n"
                    "Use pandas for data manipulation and matplotlib for charts. Do not import heavy libs.\n"
                    "If returning a chart, produce matplotlib code that draws the figure (no show()) and nothing else.\n"
                )
                
                # Combine system instructions with user prompt
                raw_input = system + "\n# User prompt: " + final_prompt
                
                # Call the updated LLM function
                llm_out = ask_llm(raw_input, model=llm_model)
                
                # Check for errors returned by ask_llm
                if llm_out.startswith("[LLM-"):
                    st.warning("LLM unavailable or returned an error.")
                    st.error(llm_out)
                    st.stop()
                
                # Attempt to extract python block
                if "```python" in llm_out:
                    try:
                        # Extract code between ```python and ```
                        code = llm_out.split("```python")[1].split("```")[0]
                        res = run_code(df, code)
                    except Exception as e:
                        st.error(f"Failed to execute code from LLM: {e}")
                        with st.expander("View generated code"):
                            st.code(code, language='python')
                        st.stop()
                else:
                    st.error("LLM did not return a python code block. Showing raw output:")
                    st.write(llm_out)
                    st.stop()
            else:
                st.error("This is a custom prompt that cannot be parsed deterministically. Please enable 'Use LLM' in the sidebar.")
                st.stop()

    # Display result
    if res["type"] == "text":
        st.markdown("#### Output (text)")
        st.text(res["output"])
    elif res["type"] == "dataframe":
        st.markdown("#### Output (table)")
        st.dataframe(res["df"])
        # Provide CSV download
        csv = res["df"].to_csv(index=False).encode("utf-8")
        st.download_button("Download result as CSV", data=csv, file_name="result.csv", mime="text/csv")
    elif res["type"] == "image":
        st.markdown("#### Output (chart)")
        st.image(res["path"], use_column_width=True)
    else:
        st.write("Unknown result type", res)