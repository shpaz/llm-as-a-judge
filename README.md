# RAG Model Evaluation Project

This document contains all the necessary scripts and configuration files to set up and run the RAG (Retrieval-Augmented Generation) evaluation framework.

## 1. README.md

This is the main guide for the project.

### RAG Model Evaluation Framework

This project provides a comprehensive framework for evaluating Retrieval-Augmented Generation (RAG) models using the `ragas` library and OpenAI's GPT models. It includes scripts to automate environment setup, run a deterministic evaluation, and produce a statistical summary of the results.

#### **Table of Contents**
1. [Project Overview](#project-overview)
2. [Understanding the Evaluation Metrics](#understanding-the-evaluation-metrics)
3. [Prerequisites](#prerequisites)
4. [Setup Instructions](#setup-instructions)
5. [Running the Evaluation](#running-the-evaluation)
6. [File Descriptions](#file-descriptions)

---

#### **1. Project Overview**

Retrieval-Augmented Generation (RAG) is a technique that enhances the capabilities of Large Language Models (LLMs) by connecting them to external knowledge bases. Instead of relying solely on its training data, a RAG system first **retrieves** relevant information from a specific source (like internal documents or a database) and then uses that information to **generate** a more accurate, factual, and context-aware answer.

**The Challenge:** How can we be sure that our RAG system is performing well? When we make changes, how do we know if they are actually improvements?

**The Solution:** This project provides a standardized, automated framework to solve that problem. By using the `ragas` library, we can run a suite of tests against our RAG pipeline to produce quantitative, objective scores. This allows us to benchmark performance, track improvements over time, and identify specific weaknesses in either the retrieval or generation components of our system.

---

#### **2. Understanding the Evaluation Metrics**

This framework calculates four key metrics to provide a holistic view of the system's performance.

##### **`faithfulness`**
- **The Question It Answers:** Does the answer stick to the facts provided in the retrieved text?
- **What It Measures:** This metric is crucial for detecting **hallucinations**. A low faithfulness score means the model is making things up that are not supported by the context it was given. A high score means the answer is factually grounded.

##### **`answer_relevancy`**
- **The Question It Answers:** Is the answer actually on-topic and relevant to the user's question?
- **What It Measures:** This checks if the answer is focused and useful. An answer can be factually correct but completely irrelevant. A low score here means the model is not addressing the core of the user's query.

##### **`context_recall`**
- **The Question It Answers:** Did the retrieval system find all the necessary information from the knowledge base to answer the question?
- **What It Measures:** This evaluates the **retriever** component of the RAG system. A low score indicates that the retriever failed to find the crucial pieces of information needed, making it impossible for the generator to form a complete answer.

##### **`context_precision`**
- **The Question It Answers:** Of all the information retrieved, how much of it was actually useful and not just noise?
- **What It Measures:** This also evaluates the **retriever**, but focuses on the signal-to-noise ratio. A low score means the retriever is pulling in a lot of irrelevant junk along with the useful information, which can confuse the generator and lead to less precise answers.

---

#### **3. Prerequisites**

Before you begin, ensure you have the following installed on your system:
- **Python 3.8+**: [Download Python](https://www.python.org/downloads/)
- **Git**: For version control (optional, but good practice).

---

#### **4. Setup Instructions**

Follow these steps to set up the project environment.

##### **Step 1: Get the Project Files**
Save all the files provided in this document into a new project directory.

##### **Step 2: Create the Environment Variables File**
The evaluation script requires an OpenAI API key.

1.  Create a file named `.env` in the root of the project directory.
2.  Copy the contents of the `.env.example` section below into the `.env` file and add your secret key.

##### **Step 3: Set Up and Activate the Virtual Environment**
A script is provided to automatically create and activate a Python virtual environment.

* **On macOS or Linux:**
    Run the script using the `source` command.
    ```bash
    source setup_and_activate.sh
    ```

* **On Windows (Command Prompt):**
    Simply run the batch file by name.
    ```batch
    setup_and_activate.bat
    ```
After running the script, your terminal prompt should change to show `(.venv)` at the beginning.

##### **Step 4: Install Dependencies**
Once the virtual environment is active, install all the required Python packages using the `requirements.txt` file.

```bash
pip install -r requirements.txt
```

---

#### **5. Running the Evaluation**

With the setup complete, you can now run the evaluation script.

1.  **Ensure your virtual environment is active.**
2.  **Place your data file** (`ragas_dataset.csv`) in the root of the project directory.
3.  **Execute the main Python script:**

    ```bash
    python evaluate.py
    ```

##### **Expected Output**
The script will perform the following actions:
1. Load the OpenAI API key from your `.env` file.
2. Connect to OpenAI's API, using GPT-4o as the evaluator for high-quality results.
3. Run the evaluation on the entire dataset.
4. Save the detailed, row-by-row results to a new file named `ragas_evaluation_full_dataset.csv`.
5. Print a statistical summary for each metric to your console.

---

#### **6. File Descriptions**

* `evaluate.py`: The main Python script that runs the RAGAS evaluation.
* `setup_and_activate.sh`: Setup and activation script for macOS and Linux.
* `setup_and_activate.bat`: Setup and activation script for Windows.
* `requirements.txt`: A list of all Python packages required for the project.
* `.env`: A file for storing your secret API keys (you must create this).
* `ragas_dataset.csv`: The input data file to be evaluated.
* `README.md`: This file.

---
---

## 2. evaluate.py

This is the main script to run the evaluation.

```python
import os
import subprocess
import sys
import pandas as pd
from dotenv import load_dotenv
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_recall,
    context_precision,
)
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

# --- Configuration for Determinism & Performance ---
# Load the OPENAI_API_KEY from your .env file
load_dotenv()

# Explicitly configure models for consistency and high-quality Hebrew support
EVALUATION_LLM = ChatOpenAI(
    model="gpt-4o",
    temperature=0.0,  # Ensures deterministic, repeatable scores
    request_timeout=120 # Increase timeout for more stability
)
EVALUATION_EMBEDDINGS = OpenAIEmbeddings(model="text-embedding-3-small")

# --- Data Loading and Preparation ---
print("\nLoading the dataset from 'ragas_dataset.csv'...")
try:
    df = pd.read_csv('ragas_dataset.csv')
except FileNotFoundError:
    print("Error: 'ragas_dataset.csv' not found. Make sure the script is in the same directory as your dataset.")
    exit()

print("Preparing the full dataset for RAGAS evaluation...")

# Prepare a new DataFrame for RAGAS
df_prepared = pd.DataFrame()
df_prepared['question'] = df['question']
df_prepared['answer'] = df['answer']
# 'contexts' must be a list of strings
df_prepared['contexts'] = df['ground_truth'].apply(lambda x: [str(x)] if pd.notna(x) else [])
# 'reference' must be a single raw string
df_prepared['reference'] = df['ground_truth'].apply(lambda x: str(x) if pd.notna(x) else '')

ragas_dataset = Dataset.from_pandas(df_prepared)
print(f"Full dataset prepared with {len(ragas_dataset)} rows.")

# --- Main Execution Block ---
if __name__ == "__main__":
    metrics_to_evaluate = [
        faithfulness,
        answer_relevancy,
        context_recall,
        context_precision,
    ]

    print("\n--- Starting Evaluation on the ENTIRE Dataset ---")
    print("This may take several minutes to complete.")

    results = evaluate(
        dataset=ragas_dataset,
        metrics=metrics_to_evaluate,
        llm=EVALUATION_LLM,
        embeddings=EVALUATION_EMBEDDINGS,
        raise_exceptions=False,
    )

    print("\nEvaluation complete.")

    # --- Save and Process Results ---
    if results:
        evaluated_df = results.to_pandas()
        output_filename = 'ragas_evaluation_full_dataset.csv'
        evaluated_df.to_csv(output_filename, index=False, encoding='utf-8-sig')

        print(f"\n✅✅✅ Evaluation results for the full dataset saved to '{output_filename}' ✅✅✅")

        # --- Aggregated Score Calculation ---
        print("\n--- Calculating Aggregated Scores for the Full Dataset ---")
        metric_columns = [
            'faithfulness',
            'answer_relevancy',
            'context_recall',
            'context_precision'
        ]

        existing_metrics = [m for m in metric_columns if m in evaluated_df.columns]

        if existing_metrics:
            aggregated_scores = evaluated_df[existing_metrics].describe()

            print("\nStatistical Summary for Each Metric:")
            print(aggregated_scores.round(3).to_markdown())
        else:
            print("No metric scores were found in the results.")

    else:
        print("Evaluation did not return any results.")
```

---

## 3. setup_and_activate.sh (for macOS/Linux)

```bash
#!/bin/bash
# -----------------------------------------------------------------------------
# setup_and_activate.sh - Creates and/or activates a venv for macOS & Linux
# -----------------------------------------------------------------------------
# USAGE:
# You MUST run this script using the 'source' command for it to work correctly.
# In your terminal, type:
#
#   source setup_and_activate.sh
#
# -----------------------------------------------------------------------------

# Define the path to the virtual environment directory
VENV_DIR=".venv"

# Check if the activation file exists
if [ -f "$VENV_DIR/bin/activate" ]; then
    echo "Virtual environment found. Activating..."
    source "$VENV_DIR/bin/activate"
    echo "Virtual environment activated. You should see '(.venv)' in your prompt."
else
    echo "Virtual environment not found. Creating one now..."
    python3 -m venv "$VENV_DIR"
    
    if [ -f "$VENV_DIR/bin/activate" ]; then
        echo "Virtual environment created successfully. Activating..."
        source "$VENV_DIR/bin/activate"
        echo "Virtual environment activated. You should see '(.venv)' in your prompt."
    else
        echo "Error: Failed to create the virtual environment."
    fi
fi
```

---

## 4. setup_and_activate.bat (for Windows)

```batch
@echo off
rem --------------------------------------------------------------------------
rem setup_and_activate.bat - Creates and/or activates a venv for Windows
rem --------------------------------------------------------------------------
rem USAGE:
rem Simply run this batch file from your command prompt by typing:
rem
rem   setup_and_activate.bat
rem
rem --------------------------------------------------------------------------

set VENV_DIR=.venv

rem Check if the activation batch file exists
if exist "%VENV_DIR%\Scripts\activate.bat" (
    echo Virtual environment found. Activating...
    call "%VENV_DIR%\Scripts\activate.bat"
    echo Virtual environment activated.
) else (
    echo Virtual environment not found. Creating one now...
    python -m venv %VENV_DIR%
    
    if exist "%VENV_DIR%\Scripts\activate.bat" (
        echo Virtual environment created successfully. Activating...
        call "%VENV_DIR%\Scripts\activate.bat"
        echo Virtual environment activated.
    ) else (
        echo Error: Failed to create the virtual environment.
    )
)
```

---

## 5. requirements.txt

```text
pandas
datasets
ragas
langchain-openai
python-dotenv
tabulate
```

---

## 6. .env.example

Create a file named `.env` and copy this content into it, replacing the placeholder with your real key.

```dotenv
# This is an example .env file.
# Create a real '.env' file and add your secret key.

OPENAI_API_KEY="sk-YourSecretKeyGoesHere"
```
