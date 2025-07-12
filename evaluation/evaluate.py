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

# --- PART 0: Install Dependencies ---
print("--- Checking/Installing Dependencies ---")
def install(package):
    try:
        # A simple way to check if a package is installed without invoking pip every time
        __import__(package.split('-')[0])
    except ImportError:
        print(f"Installing {package}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])

try:
    install("pandas")
    install("datasets")
    install("ragas")
    install("langchain-openai")
    install("python-dotenv")
    install("tabulate")
    print("Dependencies are up to date.")
except Exception as e:
    print(f"Error installing dependencies: {e}")
    exit()

# --- Configuration for Determinism & Performance ---
# Load the OPENAI_API_KEY from your .env file
load_dotenv()

EVALUATION_LLM = ChatOpenAI(
    model="gpt-4o",
    temperature=0.0,
    request_timeout=120 # Increased timeout for more stability on larger datasets
)
EVALUATION_EMBEDDINGS = OpenAIEmbeddings(model="text-embedding-3-small")

# --- Data Loading and Preparation ---
print("\nLoading the dataset from 'ragas_dataset.csv'...")
try:
    # ---‼️ KEY CHANGE: Using the full DataFrame, not a sample ‼️---
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
