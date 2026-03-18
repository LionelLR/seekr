"""PDF text extraction using pypdf.

Converts PDF pages into text chunks suitable for embedding and search.
Page numbers are used as position markers (``start`` / ``end``) instead
of timestamps, since PDFs have no inherent time dimension.
"""

from pathlib import Path
from typing import Dict, List


def get_pdf_info(path: str) -> Dict:
    """Return basic metadata for a PDF file.

    Args:
        path: Filesystem path to the PDF.

    Returns:
        Dict with ``title`` (str) and ``page_count`` (int).
    """
    try:
        import pypdf
        reader = pypdf.PdfReader(path)
        meta = reader.metadata or {}
        title = str(meta.get("/Title", "")).strip() or Path(path).stem
        return {"title": title, "page_count": len(reader.pages)}
    except Exception:
        return {"title": Path(path).stem, "page_count": 0}


def extract_pdf_segments(path: str, chars_per_chunk: int = 3000) -> List[Dict]:
    """Extract text from a PDF and return page-grouped chunks.

    Pages are accumulated into chunks until *chars_per_chunk* characters
    are reached, at which point a new chunk is started at the next page.
    Empty pages are skipped.

    Args:
        path: Filesystem path to the PDF.
        chars_per_chunk: Soft limit on characters per chunk.  Splitting
            only happens at page boundaries.

    Returns:
        List of dicts with keys:

        - ``text``: Combined text for the chunk.
        - ``start``: First page number in the chunk (1-indexed).
        - ``end``: Last page number in the chunk (1-indexed).
    """
    import pypdf

    reader = pypdf.PdfReader(path)
    total_pages = len(reader.pages)
    segments: List[Dict] = []
    current_text = ""
    current_start = 1

    for i, page in enumerate(reader.pages):
        page_num = i + 1
        text = (page.extract_text() or "").strip()
        if not text:
            continue

        if current_text and len(current_text) + len(text) > chars_per_chunk:
            segments.append({
                "text": current_text.strip(),
                "start": current_start,
                "end": page_num - 1,
            })
            current_text = text
            current_start = page_num
        else:
            current_text = (current_text + "\n\n" + text).strip() if current_text else text
            if not current_text:
                current_start = page_num

    if current_text.strip():
        segments.append({
            "text": current_text.strip(),
            "start": current_start,
            "end": total_pages,
        })

    return segments
