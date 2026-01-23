# AI Data Analyst Chatbot (Structured Data)

## Architecture Diagram

![Architecture Diagram](screenshots/architecture-flowchart.png):

All designed to run locally.

All the way from **data upload → schema validation → intent detection → safe execution → analysis output**.

---

## What is this project?

This project is a **Personal AI Data Analyst** for structured data.

It behaves like a junior data analyst who:
- understands column meanings
- respects schema constraints
- validates questions before running analysis
- produces tables, charts, and CSV outputs
- falls back to an LLM only when required

Unlike pure “chat-with-data” tools, this system prioritizes **determinism, correctness, explainability, and safety**.

---

## Key Features

- Upload structured datasets (CSV / XLSX / JSON)
- Upload a column description schema
- Automatic schema inference and merging
- Schema validation with warnings
- Schema-aware prompt suggestions
- Intent detection and enforcement
- Deterministic Python code generation
- Sandboxed execution
- Table, chart, and CSV outputs
- LLM fallback for unsupported prompts

---

## Getting Started

Two main options:
1. Run locally using Python and Streamlit
2. Extend later for hosted or containerized deployments

This README focuses on local setup.

## Prerequisites

- Comfortable writing Python
- Familiarity with pandas and tabular data
- Basic understanding of data analysis
- Optional: understanding of LLM concepts

## Setup

Tested with Python 3.10+.

### Clone repo

```
https://github.com/Madhesh1602/Personal-AI-Data-Analyst-Chatbot.git
```

```
cd Personal-AI-Data-Analyst
```


### Create environment

```
python -m venv venv
```

### Activate environment

Linux/macOS:
```
source venv/bin/activate
```

Windows: 
```
.\venv\Scripts\activate
```

### Install requirements

```
pip install -r requirements.txt
```

## LLM Configuration

This project supports multiple ways to connect to a Large Language Model (LLM).

You can use **enterprise credentials** (default) or switch to a **public API key** such as OpenAI or Gemini.

### Option 1: Enterprise / Org Credentials

The application is preconfigured to use organization-issued credentials via environment variables.

Create a `.env` file in the project root:

```
CLIENT_ID = your_org_client_id
CLIENT_SECRET = your_org_client_secret
```

### Option 2: OpenAI/ Gemini API Key

```
OPENAI_API_KEY = your_openai_api_key
```
(or)

```
GEMINI_API_KEY = your_gemini_api_key
```

### Run the application

```
streamlit run main.py
```

### Input Files

Supported formats:
* CSV
* Excel
* JSON

**Note:** I tested this application using a titanic dataset.

### Column Description File (Required)

Supported formats:
* CSV

**Example**

| Column_name | Description | Type |
| ----- | ----- | ----- |
| **PassengerId**| Unique passenger identifier | identifier |
| **Survived** | Survival indicator (1=yes; 0=no) | numeric |
| **Pclass** | Class which the passenger belongs to (1,2,3) | numeric |


### How the System Works

1. Load dataset
2. Infer column types
3. Normalize declared schema
4. Merge inferred and declared schema
5. Validate schema inconsistencies
6. Generate schema-aware prompt suggestions
7. Detect user intent
8. Enforce schema rules
9. Generate deterministic Python code
10. Execute code in sandbox
11. Fallback to LLM if required
12. Render output


### Example Questions

* How many passengers are in the dataset?
* How many passengers survived vs did not survive?
* What percentage of passengers survived?
* How many male and female passengers are there?
* What is the average age of passengers?
* How many passengers have missing age values?
* How many unique ticket numbers are there?
* Which embarkation port has the most passengers?

### Why Schema-Aware?

Pure LLM-based systems may hallucinate or misinterpret columns.

**This system:**

* understands your data
* enforces constraints
* prevents invalid analysis
* Schema first. LLM second.


### When is the LLM used?

* Only when deterministic rules do not apply
* Only after schema validation
* Always within a controlled execution flow









