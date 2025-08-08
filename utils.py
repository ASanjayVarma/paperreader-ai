# utils.py
from fpdf import FPDF
import io

def make_txt_bytes(text: str) -> bytes:
    return text.encode("utf-8")

def make_pdf_bytes(text: str, title: str = "Summary") -> bytes:
    """
    Create a simple PDF with the text (basic formatting) and return bytes.
    """
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Arial", size=14)
    pdf.multi_cell(0, 8, txt=title)
    pdf.ln(4)
    pdf.set_font("Arial", size=11)
    # Break text into lines of reasonable length
    lines = text.splitlines()
    for line in lines:
        pdf.multi_cell(0, 6, txt=line)
    bio = io.BytesIO()
    pdf.output(bio)
    bio.seek(0)
    return bio.read()
