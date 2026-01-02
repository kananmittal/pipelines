import os

class Config:
    def __init__(self):
        self.DEFAULT_MODEL = "qwen2.5vl"
        self.DEFAULT_QUESTION = "What are the key financial highlights and risks?"
        self.RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
        
    def create_directories(self):
        os.makedirs(self.RESULTS_DIR, exist_ok=True)
