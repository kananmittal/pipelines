import os

class Config:
    # Add Google Gemini to providers
    PROVIDERS = {
        'llama3.2': 'ollama',
        'qwen2.5:14b': 'ollama',
        'gemini-1.5-pro': 'google',
        'gemini-1.5-flash': 'google',
        'gemini-2.5-flash': 'google'
    }
    
    # Default selection
    DEFAULT_MODEL = "gemini-2.5-flash"

    def __init__(self):
        self.TEXT_MODEL = "llama3"
        self.VISION_MODEL = "qwen2.5-vl"
        self.DEFAULT_QUESTION = "What are the key financial highlights and risks?"
        # List of manual questions to ask if QnA file is missing or parsing fails
        self.QUESTIONS_LIST = [
            "What are the key financial highlights?",
            "What are the operational risks mentioned?",
            "How is the company performing compared to last year?",
             # User can add more questions here
        ]
        self.RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
        
        # Input File Configuration (Standardized)
        self.INPUT_FILES = {
            "transcript": ["transcript.pdf", "Transcript.pdf", "TRANSCRIPT.pdf", "transcript.docx", "Transcript.docx"],
            "notes": ["notes.pdf", "Notes.pdf", "NOTES.pdf", "notes.docx", "Notes.docx"],
            "ppt": ["ppt.pdf", "PPT.pdf", "presentation.pdf"],
            "qna": ["qna.pdf", "QnA.pdf", "QNA.pdf", "questions.pdf", "qna.docx", "QnA.docx"]
        }
        
    def create_directories(self):
        os.makedirs(self.RESULTS_DIR, exist_ok=True)
