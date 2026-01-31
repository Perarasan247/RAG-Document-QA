"""
Document loaders for different file formats
"""

import fitz  # PyMuPDF
import pandas as pd
import json
import ebooklib
from ebooklib import epub
from bs4 import BeautifulSoup
from typing import List, Dict
import os


class DocumentLoader:
    """Base class for document loading"""

    @staticmethod
    def load_pdf(file_path: str) -> List[Dict]:
        """Load PDF and extract text with page numbers"""
        chunks = []
        try:
            doc = fitz.open(file_path)
            for page_num, page in enumerate(doc, start=1):
                text = page.get_text()
                if text.strip():
                    chunks.append(
                        {
                            "text": text,
                            "source": os.path.basename(file_path),
                            "page": page_num,
                            "doc_type": "pdf",
                            "file_path": file_path,
                        }
                    )
            doc.close()
        except Exception as e:
            print(f"Error loading PDF {file_path}: {e}")
        return chunks

    @staticmethod
    def load_txt(file_path: str) -> List[Dict]:
        """Load text file"""
        chunks = []
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                text = f.read()
                if text.strip():
                    chunks.append(
                        {
                            "text": text,
                            "source": os.path.basename(file_path),
                            "page": 1,
                            "doc_type": "txt",
                            "file_path": file_path,
                        }
                    )
        except Exception as e:
            print(f"Error loading TXT {file_path}: {e}")
        return chunks

    @staticmethod
    def load_csv(file_path: str) -> List[Dict]:
        """Load CSV file"""
        chunks = []
        try:
            df = pd.read_csv(file_path)
            # Convert entire CSV to text representation
            text = df.to_string()
            chunks.append(
                {
                    "text": text,
                    "source": os.path.basename(file_path),
                    "page": 1,
                    "doc_type": "csv",
                    "file_path": file_path,
                    "rows": len(df),
                }
            )

            # Also create chunks for each row for granular search
            for idx, row in df.iterrows():
                row_text = f"Row {idx + 1}: " + ", ".join(
                    [f"{col}: {val}" for col, val in row.items()]
                )
                chunks.append(
                    {
                        "text": row_text,
                        "source": os.path.basename(file_path),
                        "page": idx + 1,
                        "doc_type": "csv",
                        "file_path": file_path,
                        "row": idx + 1,
                    }
                )
        except Exception as e:
            print(f"Error loading CSV {file_path}: {e}")
        return chunks

    @staticmethod
    def load_json(file_path: str) -> List[Dict]:
        """Load JSON file"""
        chunks = []
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                text = json.dumps(data, indent=2)
                chunks.append(
                    {
                        "text": text,
                        "source": os.path.basename(file_path),
                        "page": 1,
                        "doc_type": "json",
                        "file_path": file_path,
                    }
                )
        except Exception as e:
            print(f"Error loading JSON {file_path}: {e}")
        return chunks

    @staticmethod
    def load_epub(file_path: str) -> List[Dict]:
        """Load EPUB ebook"""
        chunks = []
        try:
            book = epub.read_epub(file_path)
            chapter_num = 0

            for item in book.get_items():
                if item.get_type() == ebooklib.ITEM_DOCUMENT:
                    chapter_num += 1
                    soup = BeautifulSoup(item.get_content(), "html.parser")
                    text = soup.get_text()

                    if text.strip():
                        chunks.append(
                            {
                                "text": text,
                                "source": os.path.basename(file_path),
                                "page": chapter_num,
                                "doc_type": "epub",
                                "file_path": file_path,
                                "chapter": chapter_num,
                            }
                        )
        except Exception as e:
            print(f"Error loading EPUB {file_path}: {e}")
        return chunks

    @staticmethod
    def load_document(file_path: str) -> List[Dict]:
        """Auto-detect file type and load accordingly"""
        ext = os.path.splitext(file_path)[1].lower()

        loaders = {
            ".pdf": DocumentLoader.load_pdf,
            ".txt": DocumentLoader.load_txt,
            ".csv": DocumentLoader.load_csv,
            ".json": DocumentLoader.load_json,
            ".epub": DocumentLoader.load_epub,
        }

        loader = loaders.get(ext)
        if loader:
            return loader(file_path)
        else:
            print(f"Unsupported file type: {ext}")
            return []
