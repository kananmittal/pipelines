import logging
import sys
import os
import json
from dotenv import load_dotenv
from google import genai
from google.genai import types

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from multimodal_utils import OllamaInterface

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)

class LLMInterface:
    def __init__(self, model_name: str = None):
        from config import Config
        self.config = Config()
        
        # Determine model names
        # Default to Config.TEXT_MODEL ("llama3") for text tasks if not provided
        self.model_name = model_name or self.config.TEXT_MODEL
        
        # Initialize Ollama for TEXT tasks (e.g. Llama 3)
        self.ollama = OllamaInterface(model_name=self.model_name)
        logger.info(f"LLM Interface: Text Model initialized as {self.model_name}")
        
        # Vision tasks should use Config.VISION_MODEL ("qwen2.5-vl") externally via VisionProcessor
        
        # Initialize Gemini for Judge Layer
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            logger.warning("GEMINI_API_KEY not found in .env. Hallucination scoring may fail.")
            self.gemini = None
        else:
            self.gemini = genai.Client(api_key=api_key)
            logger.info("Gemini Client initialized for Judge Layer")

    def calculate_hallucination_score(self, answer: str, visual_facts: str, textual_facts: str) -> dict:
        """
        Computes Hallucination Score using the 'LLM as a judge' prompt from the paper/image.
        """
        if not self.gemini:
            return {"error": "Gemini API key missing", "final_score": 0.0}

        # Prompt from 'C. LLM as a judge' in the image
        prompt = f"""You are a hallucination detection specialist. Carefully read the provided answer along with the corresponding question and source text. First, identify whether any part of the answer introduces content not present in the source, including analytical reasoning and numerical values. Then, distinguish between the two types of hallucinations:
- Analytical hallucination: logical inferences or conclusions not grounded in the text.
- Numerical hallucination: incorrect or fabricated numerical values or calculations.
For each type, could you explain step-by-step whether the answer remains faithful to the source or introduces inaccuracies? Finally, assign a hallucination score from 0 to 1 for both categories, where zero means entirely factual and 1 means completely hallucinated. Return two numbers only: one for analytical hallucination and one for numerical hallucination, in that order.

Source Text (Textual + Visual Context):
{textual_facts[:10000]}
{visual_facts[:5000]}

Answer to Evaluate:
{answer}

Return ONLY a JSON object with keys: "analytical_score", "numerical_score", "reasoning".
"""
        
        # Retry logic for Gemini API (handle 429 Resource Exhausted)
        max_retries = 5
        base_delay = 2
        import time

        for attempt in range(max_retries):
            try:
                response = self.gemini.models.generate_content(
                    model="gemini-2.0-flash",
                    contents=prompt,
                    config=types.GenerateContentConfig(response_mime_type="application/json")
                )
                scores = json.loads(response.text, strict=False)
                
                # Handle case where Gemini returns a list
                if isinstance(scores, list):
                    if len(scores) > 0:
                        scores = scores[0]
                    else:
                        scores = {}

                # Average the two scores for a final score (0 = factual, 1 = hallucinated)
                analytical = float(scores.get('analytical_score', 0))
                numerical = float(scores.get('numerical_score', 0))
                final_score = (analytical + numerical) / 2.0
                
                scores['final_score'] = round(final_score, 4)
                return scores

            except Exception as e:
                # Check for rate limit error in string representation
                if "429" in str(e) or "Too Many Requests" in str(e) or "Resource Exhausted" in str(e):
                    wait_time = base_delay * (2 ** attempt)
                    logger.warning(f"Gemini API Rate Limit (429). Retrying in {wait_time}s... (Attempt {attempt + 1}/{max_retries})")
                    time.sleep(wait_time)
                else:
                    logger.error(f"Gemini Hallucination Scoring failed: {e}")
                    return {"error": str(e), "final_score": 0.0}
        
        logger.error(f"Gemini Hallucination Scoring failed after {max_retries} retries.")
        return {"error": "Max retries exceeded", "final_score": 0.0}

    def extract_information(self, text: str, type: str) -> str:
        # Prompt from 'A. Prompts for LLM pipelines' (Extraction)
        prompt = f"""You are a financial information extraction expert. Your task is to identify explicit factual statements from a single document, such as a conference call transcript or an analyst note, without paraphrasing, summarizing, or interpreting the content.
- Scan the document for directly stated facts, figures, and statements attributed to speakers.
- Focus on concrete signals: revenue numbers, growth figures, cost changes, strategic announcements, and outlook guidance.
- Isolate each relevant passage's original phrase as a bullet point. Can you include the context only if necessary to maintain clarity?
- Avoid combining multiple ideas or generating abstract summaries; capture the raw insight. Please ensure that no fabricated details are introduced and all the content is in the document.
- Your goal is to build a faithful fact bank of the document's explicit information for downstream reasoning or question-answering tasks.

Document content:
{text[:15000]}"""
        return self.ollama.chat([{"role": "user", "content": prompt}])

    def consolidate_information(self, text1: str, text2: str) -> str:
        # Prompt from 'A. Prompts for LLM pipelines' (Consolidation)
        prompt = f"""Let's generate a consolidated summary of the two source documents: a transcript of an earnings call (conference call) and a note (bullet form) derived from the same transcript. I want you to carefully read the entire transcript and notes to understand the context.
1) Identify and extract the key topics and insights discussed in depth from the documents.
2) Please pay attention to any numerical data presented in the documents.
3) When including numbers in the summary, ensure they are:
    a) Explicitly stated values from the documents (do not fabricate numbers).
    b) Appropriately represented with clear context from the documents.
4) Synthesize the extracted information and numbers into a concise, logical summary.
5) Conserve all the essential information from the transcript and notes, so you can use it to answer any question.

Document 1 (Transcript):
{text1[:10000]}

Document 2 (Notes):
{text2[:5000]}"""
        return self.ollama.chat([{"role": "user", "content": prompt}])

    def generate_single_response(self, prompt: str, options: dict = None) -> dict:
        response = self.ollama.chat([{"role": "user", "content": prompt}])
        return {'response': response, 'generation_time': 0.5, 'parameters': options or {}}

    def generate_qa_responses(self, context: str, question: str) -> dict:
        # Prompt from 'B. For answering questions'
        prompt = f"""You are an expert financial analyst. Would you be able to read the given summary text from an earnings call?
- Identify the key facts or figures relevant to the question.
- Reason through how these facts answer the question.
- Provide a concise one-line answer based only on the provided summary.

Summary Text:
{context[:10000]}

Question:
{question}"""
        response = self.ollama.chat([{"role": "user", "content": prompt}])
        return {
            'multiple_responses': [{'response': response, 'generation_time': 0.5, 'parameters': {}}],
            'best_response': {'response': response, 'generation_time': 0.5, 'parameters': {}}
        }
    
    def critique_and_refine(self, text: str, extraction: str) -> str:
        # Keeping previous prompt as no specific critique prompt was provided in image
        prompt = f"Original Text:\n{text[:5000]}\n\nExtraction:\n{extraction}\n\nCritique and Refine the extraction:"
        return self.ollama.chat([{"role": "user", "content": prompt}])
