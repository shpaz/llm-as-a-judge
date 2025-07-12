import pandas as pd
import re

from docx import Document

# Load the .docx file
doc = Document("output.docx")
full_text = "\n".join([para.text for para in doc.paragraphs])

# Pattern to extract each sample (question, ground_truth, answer)
pattern = r"user_query:\s*(.*?)\n-+\ntagged_answer:\s*(.*?)\n-+\nfinal_response_output:\s*(.*?)\n-+"

matches = re.findall(pattern, full_text, re.DOTALL)

# Convert to DataFrame
df = pd.DataFrame(matches, columns=["question", "ground_truth", "answer"])

# Clean up whitespace
df = df.applymap(lambda x: x.strip())

# Remove rows where any of the essential fields are missing
df.replace("nan", pd.NA, inplace=True)
df.dropna(subset=["question", "ground_truth", "answer"], inplace=True)

# Remove answers that include 'explanation' or internal answer tags
df = df[~df['answer'].str.contains("explanation|<internal answer>", case=False, na=False)]

# Save to CSV (optional)
df.to_csv("ragas_dataset.csv", index=False, encoding="utf-8")

# Print preview
print(df.head())

