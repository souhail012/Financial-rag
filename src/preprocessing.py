import re
import unicodedata
import pandas as pd
from pathlib import Path
import logging
from typing import List, Dict

# NLTK stopwords
from nltk.corpus import stopwords
from src.config import PROCESSED_FOLDER, CHUNK_SIZE, CHUNK_OVERLAP

# Ensure stopwords are available
try:
    STOPWORDS = set(stopwords.words("english"))
except LookupError:
    import nltk
    nltk.download("stopwords")
    STOPWORDS = set(stopwords.words("english"))
import os

# Extend with domain-specific stopwords
FINANCE_STOPWORDS = {
    "inc", "ltd", "company", "group", "fiscal", "year", "report", "note",
    "statement", "financial", "corporation", "limited", "table"
}
STOPWORDS |= FINANCE_STOPWORDS

# Configure logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO
)
logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------
# 🧹 TEXT CLEANING UTILITIES
# ----------------------------------------------------------------------

def normalize_unicode(text: str) -> str:
    """Normalize Unicode and remove odd characters."""
    return unicodedata.normalize("NFKC", text)


def clean_text(text: str, remove_stopwords: bool = False) -> str:
    """Basic cleaning for financial document text."""
    if not isinstance(text, str):
        return ""

    # Normalize and sanitize text
    text = normalize_unicode(text)
    text = re.sub(r'[\n\r\t]', ' ', text)
    # Keep numbers, $, %, (), /, -
    text = re.sub(r'[^\w\s.,;:$%\-()/%]', ' ', text)
    text = re.sub(r'\s+', ' ', text)

    if remove_stopwords:
        words = text.split()
        text = " ".join([w for w in words if w.lower() not in STOPWORDS])

    return text.strip()


# ----------------------------------------------------------------------
# ✂️ CHUNKING FUNCTIONS
# ----------------------------------------------------------------------

def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    """
    Split text into overlapping chunks by word count.
    """
    words = text.split()
    chunks = []
    step = max(1, chunk_size - overlap)

    for i in range(0, len(words), step):
        chunk = " ".join(words[i:i + chunk_size])
        if len(chunk.split()) >= 30:
            chunks.append(chunk)

    return chunks


# ----------------------------------------------------------------------
# 📊 TABLE-TO-TEXT CONVERSION (for embedding)
# ----------------------------------------------------------------------

def table_to_text_chunk(df: pd.DataFrame, filename: str, page: int, t_idx: int) -> str:
    """
    Convert a small table into plain text for embedding in RAG.

    """
    if df is None or df.empty:
        return ""

    max_rows = min(len(df), 10)
    headers = " | ".join(df.columns.astype(str).tolist())
    lines = [headers]

    for _, row in df.head(max_rows).iterrows():
        row_str = " | ".join(row.astype(str).tolist())
        lines.append(row_str)

    chunk = (
        f"TABLE from {filename} (page {page}, table {t_idx}):\n"
        + "\n".join(lines)
    )
    return chunk


def load_and_convert_tables(tables_index_csv: str) -> List[Dict[str, str]]:
    """
    Read the tables index and convert each table CSV into text chunks.
    Returns a list of dicts: {filename, chunk_id, text, type='table'}
    """
    if not Path(tables_index_csv).exists():
        logger.info(f"No table index found at {tables_index_csv}")
        return []

    tidf = pd.read_csv(tables_index_csv)
    all_table_chunks = []
    for _, row in tidf.iterrows():
        csv_path = row.get("csv_path")
        pdf = row.get("pdf")
        page = int(row.get("page", 0))
        t_idx = int(row.get("table_index", 0))
        if not Path(csv_path).exists():
            continue
        try:
            df = pd.read_csv(csv_path)
            text = table_to_text_chunk(df, pdf, page, t_idx)
            if text.strip():
                all_table_chunks.append({
                    "filename": pdf,
                    "chunk_id": f"table_{page}_{t_idx}",
                    "text": text,
                    "length_words": len(text.split()),
                    "type": "table"
                })
        except Exception as e:
            logger.warning(f"Failed reading {csv_path}: {e}")

    logger.info(f"Converted {len(all_table_chunks)} tables to text chunks.")
    return all_table_chunks


# ----------------------------------------------------------------------
# 🚀 MAIN PIPELINE
# ----------------------------------------------------------------------

def preprocess_texts(
    input_csv: str = "data/processed/raw_texts.csv",
    tables_index_csv: str = "data/processed/tables_index.csv",
    output_csv: str = "data/processed/clean_chunks.csv",
    remove_stopwords: bool = False,
    include_tables: bool = True
):
    """
    Clean + chunk text and optionally include table data.
    """
    input_path = Path(input_csv)
    output_path = Path(output_csv)

    if not input_path.exists():
        raise FileNotFoundError(f"❌ Input text file not found: {input_path}")

    df = pd.read_csv(input_path)
    all_chunks = []
    total_docs, total_chunks = 0, 0

    logger.info("Starting preprocessing...")

    for _, row in df.iterrows():
        filename = row.get("filename", "unknown")
        raw_text = row.get("text", "")

        if not isinstance(raw_text, str) or not raw_text.strip():
            logger.warning(f"Skipping empty text for file: {filename}")
            continue

        cleaned = clean_text(raw_text, remove_stopwords)
        chunks = chunk_text(cleaned)

        for idx, chunk in enumerate(chunks):
            all_chunks.append({
                "filename": filename,
                "chunk_id": idx,
                "text": chunk,
                "length_words": len(chunk.split()),
                "type": "text"
            })

        total_docs += 1
        total_chunks += len(chunks)

    # Add table chunks if available
    if include_tables:
        table_chunks = load_and_convert_tables(tables_index_csv)
        all_chunks.extend(table_chunks)
        total_chunks += len(table_chunks)

    if not all_chunks:
        logger.error("❌ No chunks generated — check your extraction or cleaning rules.")
        return None

    out_df = pd.DataFrame(all_chunks)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(output_path, index=False, encoding="utf-8")

    avg_len = out_df["length_words"].mean()
    logger.info("✅ Preprocessing completed successfully!")
    logger.info(f"📊 Documents processed: {total_docs}")
    logger.info(f"🧩 Total chunks: {total_chunks}")
    logger.info(f"📝 Average chunk length: {avg_len:.2f} words")
    logger.info(f"💾 Output saved to: {output_path}")

    return out_df


if __name__ == "__main__":
    preprocess_texts(
        remove_stopwords=True,
        include_tables=True
    )
