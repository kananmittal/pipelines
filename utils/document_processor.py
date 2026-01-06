import fitz  # PyMuPDF
import docx
import io
import logging

logger = logging.getLogger(__name__)

class DocumentProcessor:
    def __init__(self):
        pass

    def read_pdf(self, file_obj) -> str:
        """Read text from a PDF file object"""
        text = ""
        try:
            # If file_obj is bytes, wrap in BytesIO, else read
            if isinstance(file_obj, bytes):
                stream = io.BytesIO(file_obj)
            else:
                stream = file_obj
                
            doc = fitz.open(stream=stream.read(), filetype="pdf")
            for page in doc:
                text += page.get_text() + "\n"
        except Exception as e:
            logger.error(f"Error reading PDF: {e}")
            return ""
        return text

    def read_docx(self, file_path) -> str:
        """Read text from a DOCX file path"""
        text = ""
        try:
            doc = docx.Document(file_path)
            full_text = []
            for para in doc.paragraphs:
                full_text.append(para.text)
            text = '\n'.join(full_text)
        except Exception as e:
            logger.error(f"Error reading DOCX: {e}")
            return ""
        return text

    def preprocess_text(self, text: str) -> str:
        """Clean and normalize text"""
        if not text:
            return ""
        # Basic cleanup
        text = text.replace('\xa0', ' ')
        return text.strip()
