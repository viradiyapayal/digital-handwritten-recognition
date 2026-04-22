import fitz  # PyMuPDF
from PIL import Image
import io

def pdf_to_image(pdf_bytes: bytes) -> Image.Image:
    """Convert first page of PDF to PIL Image."""
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    page = doc[0]
    # Higher DPI for better quality
    pix = page.get_pixmap(dpi=300)
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    return img