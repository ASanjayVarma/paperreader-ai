# pdf_utils.py
import fitz  # pymupdf
import io

def extract_text_from_pdf(file_like) -> str:
    """
    Accepts an uploaded file-like object (Streamlit's UploadedFile)
    Returns the full extracted text as a single string.
    """
    # fitz accepts a filename or bytes
    if hasattr(file_like, "read"):
        data = file_like.read()
    else:
        # assume filepath
        with open(file_like, "rb") as f:
            data = f.read()

    doc = fitz.open(stream=data, filetype="pdf")
    pages = []
    for page in doc:
        txt = page.get_text("text")
        pages.append(txt)
    doc.close()
    full_text = "\n\n".join(pages)
    # Normalize whitespace
    return full_text
