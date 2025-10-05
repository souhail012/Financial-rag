import re
import pandas as pd
from src.config import PROCESSED_FOLDER, CHUNK_SIZE, CHUNK_OVERLAP

def clean_text(text: str) -> str:
    """Basic text cleaning."""
    if not isinstance(text, str):
        return ""
    text = re.sub(r'[\n\r\t]', ' ', text)  # remove newlines/tabs
    text = re.sub(r'[^\w\s.,;:$%\-()]', ' ', text)  # keep finance symbols
    text = re.sub(r'\s+', ' ', text)  # collapse spaces
    return text.strip()

def chunk_text(text: str, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """Split text into overlapping chunks."""
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i + chunk_size])
        if len(chunk) > 50:  # ignore very small chunks
            chunks.append(chunk)
    return chunks

def preprocess_texts(input_csv="data/processed/raw_texts.csv", output_csv="data/processed/clean_chunks.csv"):
    """Clean and chunk texts, then save."""
    df = pd.read_csv(input_csv)
    all_chunks = []

    for _, row in df.iterrows():
        filename = row["filename"]
        raw_text = row["text"]
        cleaned = clean_text(raw_text)
        chunks = chunk_text(cleaned)

        for idx, chunk in enumerate(chunks):
            all_chunks.append({
                "filename": filename,
                "chunk_id": idx,
                "text": chunk
            })

    out_df = pd.DataFrame(all_chunks)
    out_df.to_csv(output_csv, index=False, encoding="utf-8")
    print(f"✅ Cleaned & chunked texts saved to {output_csv}")
    return out_df

if __name__ == "__main__":
    preprocess_texts()
