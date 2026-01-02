import base64
import io
import os
import logging
import requests
import json
from typing import List, Dict, Any, Union
from pdf2image import convert_from_path
from PIL import Image

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class VisionProcessor:
    """
    Handles image processing for multimodal tasks.
    """
    
    @staticmethod
    def load_pdf_as_images(pdf_path: str) -> List[Image.Image]:
        """
        Convert PDF pages to PIL Images.
        """
        try:
            logger.info(f"Converting PDF to images: {pdf_path}")
            images = convert_from_path(pdf_path)
            
            # Check for DRY_RUN_LIMIT
            limit = os.getenv("DRY_RUN_LIMIT")
            if limit:
                try:
                    limit_int = int(limit)
                    logger.info(f"DRY_RUN_LIMIT set: processing only first {limit_int} pages.")
                    images = images[:limit_int]
                except ValueError:
                    logger.warning(f"Invalid DRY_RUN_LIMIT value: {limit}")

            logger.info(f"Successfully converted {len(images)} pages.")
            return images
        except Exception as e:
            logger.error(f"Error converting PDF to images: {e}")
            raise

    @staticmethod
    def encode_image_to_base64(image: Image.Image) -> str:
        """
        Convert a PIL Image to a Base64 string.
        """
        buffered = io.BytesIO()
        image.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
        return img_str

class OllamaInterface:
    """
    Interface for local Ollama instance (Qwen2.5-VL).
    """
    def __init__(self, model_name: str = "qwen2.5vl", base_url: str = "http://localhost:11434"):
        self.model_name = model_name
        self.base_url = base_url
        self.api_generate = f"{base_url}/api/generate"
        self.api_chat = f"{base_url}/api/chat"
        logger.info(f"Ollama Interface initialized with model: {self.model_name}")

    def analyze_image(self, base64_image: str, prompt: str = "Describe this image in detail.") -> str:
        """
        Send a Base64 image to Qwen2.5-VL for analysis.
        """
        logger.info("Sending image to Ollama for analysis...")
        
        payload = {
            "model": self.model_name,
            "messages": [
                {
                    "role": "user",
                    "content": prompt,
                    "images": [base64_image]
                }
            ],
            "stream": False 
        }

        try:
            response = requests.post(self.api_chat, json=payload)
            response.raise_for_status()
            result = response.json()
            description = result.get("message", {}).get("content", "")
            logger.info("Image analysis completed.")
            return description
        except requests.exceptions.RequestException as e:
            logger.error(f"Ollama API request failed: {e}")
            return f"Error analyzing image: {e}"

    def chat(self, messages: List[Dict[str, str]]) -> str:
        """
        Standard text-only chat.
        """
        payload = {
            "model": self.model_name,
            "messages": messages,
            "stream": False
        }
        
        try:
            response = requests.post(self.api_chat, json=payload)
            response.raise_for_status()
            return response.json().get("message", {}).get("content", "")
        except requests.exceptions.RequestException as e:
            logger.error(f"Ollama Chat failed: {e}")
            return f"Error in chat: {e}"

if __name__ == "__main__":
    # verification test
    import os
    print("Testing multimodal_utils...")
    
    # Create a dummy image if ppt.pdf doesn't exist for testing logic
    if not os.path.exists("ppt.pdf"):
        print("ppt.pdf not found, skipping PDF load test.")
    else:
        processor = VisionProcessor()
        images = processor.load_pdf_as_images("ppt.pdf")
        if images:
            print(f"Loaded {len(images)} pages.")
            b64 = processor.encode_image_to_base64(images[0])
            print(f"Base64 snippet: {b64[:50]}...")
            
            # Ollama test
            ollama = OllamaInterface()
            print("Sending Page 1 to Ollama...")
            desc = ollama.analyze_image(b64, "Describe the chart on this slide.")
            print(f"Analysis: {desc}")
