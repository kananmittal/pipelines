import os

class Config:
    def __init__(self):
        self.DEFAULT_MODEL = "llama3" 
        self.TEXT_MODEL = "llama3"
        self.VISION_MODEL = "qwen2.5-vl"
        self.DEFAULT_QUESTION = "What are the key financial highlights and risks?"
        self.RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
        
        # Input File Configuration (Standardized)
        self.INPUT_FILES = {
            "transcript": ["transcript.pdf", "Transcript.pdf", "TRANSCRIPT.pdf"],
            "notes": ["notes.pdf", "Notes.pdf", "NOTES.pdf", "notes.docx", "Notes.docx"],
            "ppt": ["ppt.pdf", "PPT.pdf", "presentation.pdf"],
            "qna": ["qna.pdf", "QnA.pdf", "QNA.pdf", "questions.pdf", "qna.docx", "QnA.docx"]
        }
        
    def create_directories(self):
        os.makedirs(self.RESULTS_DIR, exist_ok=True)
