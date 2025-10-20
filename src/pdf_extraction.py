import os
import csv
import logging
from typing import List, Dict, Any, Optional

import pandas as pd
from pypdf import PdfReader
import pdfplumber

# Optional OCR imports (only used if you enable OCR fallback)
try:
    from pdf2image import convert_from_path
    import pytesseract
    OCR_AVAILABLE = True
except Exception:
    OCR_AVAILABLE = False

from src.config import PDF_FOLDER, PROCESSED_FOLDER

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def extract_text_with_pypdf(pdf_path: str) -> str:
    text = []
    try:
        reader = PdfReader(pdf_path)
        for p in reader.pages:
            try:
                # p.extract_text() returns None sometimes
                txt = p.extract_text() or ""
                text.append(txt)
            except Exception as e:
                logger.warning(f"pypdf: failed extracting page: {e}")
                text.append("")
    except Exception as e:
        logger.error(f"pypdf: failed opening {pdf_path}: {e}")
        return ""
    return "\n".join(text)


def extract_tables_with_pdfplumber(pdf_path: str) -> List[Dict[str, Any]]:
    """
    Returns list of tables info: {pdf, page, table_index, dataframe}
    """
    tables_info = []
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page_num, page in enumerate(pdf.pages):
                try:
                    tables = page.extract_tables()
                except Exception as e:
                    logger.warning(f"pdfplumber failed on page {page_num}: {e}")
                    tables = []

                for t_idx, table in enumerate(tables):
                    # table is a list of rows (list of lists)
                    if not table or len(table) == 0:
                        continue
                    # If first row looks like header, use it
                    df = pd.DataFrame(table)
                    # try to promote first row to header if reasonable
                    if df.shape[1] > 1:
                        # naive header detection: non-numeric cells in first row
                        header_candidate = df.iloc[0].astype(str).tolist()
                        df = df[1:]
                        df.columns = header_candidate
                    tables_info.append({
                        "pdf": os.path.basename(pdf_path),
                        "page": page_num,
                        "table_index": t_idx,
                        "dataframe": df
                    })
    except Exception as e:
        logger.error(f"pdfplumber open failed for {pdf_path}: {e}")
    return tables_info


def ocr_pdf_page_images(pdf_path: str, dpi: int = 200) -> List[str]:
    """
    Convert PDF to images and OCR each page. Returns list of OCRed page texts.
    Requires pdf2image + pytesseract + poppler + tesseract installed.
    """
    if not OCR_AVAILABLE:
        raise RuntimeError("OCR dependencies (pdf2image/pytesseract) not available.")
    pages = convert_from_path(pdf_path, dpi=dpi)
    texts = []
    for img in pages:
        try:
            txt = pytesseract.image_to_string(img)
        except Exception as e:
            logger.warning(f"OCR failed on a page: {e}")
            txt = ""
        texts.append(txt)
    return texts


def extract_all_pdfs(output_text_csv: str = os.path.join(PROCESSED_FOLDER, "raw_texts.csv"),
                     tables_dir: str = os.path.join(PROCESSED_FOLDER, "tables"),
                     tables_index_csv: str = os.path.join(PROCESSED_FOLDER, "tables_index.csv"),
                     ocr_fallback: bool = False):
    os.makedirs(PROCESSED_FOLDER, exist_ok=True)
    os.makedirs(tables_dir, exist_ok=True)

    rows = []
    tables_index_rows = []

    # load existing to avoid reprocessing
    if os.path.exists(output_text_csv):
        existing_df = pd.read_csv(output_text_csv)
        processed_files = set(existing_df["filename"].tolist())
    else:
        existing_df = pd.DataFrame(columns=["filename", "text"])
        processed_files = set()

    for fname in os.listdir(PDF_FOLDER):
        if not fname.lower().endswith(".pdf"):
            continue
        if fname in processed_files:
            logger.info(f"Skipping already processed file: {fname}")
            continue

        pdf_path = os.path.join(PDF_FOLDER, fname)
        logger.info(f"Processing {fname}")

        # 1) Extract text
        try:
            text = extract_text_with_pypdf(pdf_path)
        except Exception as e:
            logger.error(f"Text extraction failed for {fname}: {e}")
            text = ""

        # OCR fallback if enabled and pypdf extracted nothing
        if (not text or text.strip() == "") and ocr_fallback:
            logger.info(f"No text found in {fname}. Running OCR fallback (this will be slower).")
            try:
                ocr_pages = ocr_pdf_page_images(pdf_path)
                text = "\n".join(ocr_pages)
            except Exception as e:
                logger.error(f"OCR failed for {fname}: {e}")
                text = ""

        rows.append({"filename": fname, "text": text})

        # 2) Extract tables with pdfplumber
        try:
            tables = extract_tables_with_pdfplumber(pdf_path)
            for t in tables:
                pdf_name = t["pdf"]
                page = t["page"]
                tidx = t["table_index"]
                df = t["dataframe"]
                safe_name = f"{os.path.splitext(pdf_name)[0]}_p{page}_t{tidx}.csv"
                out_path = os.path.join(tables_dir, safe_name)
                try:
                    df.to_csv(out_path, index=False)
                    tables_index_rows.append({
                        "pdf": pdf_name,
                        "page": page,
                        "table_index": tidx,
                        "csv_path": out_path,
                        "nrows": df.shape[0],
                        "ncols": df.shape[1]
                    })
                except Exception as e:
                    logger.warning(f"Failed saving table csv for {fname} page {page} table {tidx}: {e}")
        except Exception as e:
            logger.error(f"Table extraction failed for {fname}: {e}")

    # merge with existing
    if os.path.exists(output_text_csv):
        existing_df = pd.read_csv(output_text_csv)
        new_df = pd.DataFrame(rows)
        df = pd.concat([existing_df, new_df], ignore_index=True)
    else:
        df = pd.DataFrame(rows)

    df.to_csv(output_text_csv, index=False, encoding="utf-8")
    logger.info(f"Saved extracted texts to {output_text_csv}")

    # Save tables index
    if tables_index_rows:
        tidf = pd.DataFrame(tables_index_rows)
        if os.path.exists(tables_index_csv):
            old_tidf = pd.read_csv(tables_index_csv)
            tidf = pd.concat([old_tidf, tidf], ignore_index=True)
        tidf.to_csv(tables_index_csv, index=False, encoding="utf-8")
        logger.info(f"Saved tables index to {tables_index_csv}")
    else:
        logger.info("No tables extracted.")

    return df
if __name__ == "__main__":
    extract_all_pdfs(ocr_fallback=True)

