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
    Interface originally for Ollama, but now routed to Google Gemini (gemini-2.5-flash)
    to support remote headless servers without local Ollama instances running.
    """
    def __init__(self, model_name: str = "gemini-2.5-flash", base_url: str = ""):
        self.model_name = "gemini-2.5-flash"
        
        try:
            from google import genai
            api_key = os.getenv("GEMINI_API_KEY")
            if not api_key:
                logger.warning("GEMINI_API_KEY not found in environment. Gemini tasks may fail.")
                self.client = None
            else:
                self.client = genai.Client(api_key=api_key)
                logger.info("OllamaInterface automatically re-routed to Gemini 2.5 Flash.")
        except ImportError:
            logger.error("google-genai SDK not installed. Please run: pip install google-genai")
            self.client = None

    def analyze_image(self, base64_image: str, prompt: str = "Describe this image in detail.") -> str:
        """
        Send an image to Gemini for analysis (shimmed from Ollama).
        """
        if not self.client:
            return "Error: Gemini Client not initialized."
            
        logger.info("Sending image to Gemini for analysis...")
        try:
            from google.genai import types
            
            # Convert base64 string back to bytes for Gemini Part
            image_bytes = base64.b64decode(base64_image)
            
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=[
                    types.Part.from_bytes(
                        data=image_bytes,
                        mime_type='image/jpeg',
                    ),
                    prompt
                ]
            )
            logger.info("Image analysis completed.")
            return response.text
        except Exception as e:
            logger.error(f"Gemini API request failed: {e}")
            return f"Error analyzing image: {e}"

    def chat(self, messages: List[Dict[str, str]]) -> str:
        """
        Standard text-only chat using Gemini 2.5 Flash.
        Converts Ollama-style message arrays to a simple Gemini string prompt.
        """
        if not self.client:
            return "Error: Gemini Client not initialized."
            
        # Convert Ollama message format to a single text prompt
        full_prompt = "\n".join([msg.get("content", "") for msg in messages])
        
        try:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=full_prompt
            )
            return response.text
        except Exception as e:
            logger.error(f"Gemini Chat failed: {e}")
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
