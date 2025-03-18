import fitz  # PyMuPDF

def extract_text_from_pdf(pdf_path):
    """Extract text from a multi-page PDF."""
    doc = fitz.open(pdf_path)
    text = "\n".join([page.get_text("text") for page in doc])
    return text
