import os
import pandas as pd
from pypdf import PdfReader
from src.config import PDF_FOLDER, PROCESSED_FOLDER

def extract_text_from_pdfs(output_file="data/processed/raw_texts.csv"):
    """
    Extract text from all PDFs in PDF_FOLDER and save to a CSV file.
    """
    data = []

    # Check if we already have processed files
    if os.path.exists(output_file):
        existing_df = pd.read_csv(output_file)
        existing_files = set(existing_df["filename"])
    else:
        existing_df = pd.DataFrame(columns=["filename", "text"])
        existing_files = set()

    # Iterate over PDFs
    for filename in os.listdir(PDF_FOLDER):
        if filename.endswith(".pdf") and filename not in existing_files:
            pdf_path = os.path.join(PDF_FOLDER, filename)
            extracted_text = ""
            try:
                reader = PdfReader(pdf_path)
                for page in reader.pages:
                    extracted_text += page.extract_text() or ""
            except Exception as e:
                print(f"⚠️ Error with {filename}: {e}")
                extracted_text = ""
            
            data.append({"filename": filename, "text": extracted_text})

    # Merge with old data
    if not data:
        return existing_df

    new_df = pd.DataFrame(data)
    df = pd.concat([existing_df, new_df], ignore_index=True)

    # Save result
    os.makedirs(PROCESSED_FOLDER, exist_ok=True)
    df.to_csv(output_file, index=False, encoding="utf-8")
    print(f"✅ Extracted texts saved to {output_file}")
    return df

if __name__ == "__main__":
    extract_text_from_pdfs()

