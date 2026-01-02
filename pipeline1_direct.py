import logging
import json
import os
from datetime import datetime
from typing import Dict, Any, List
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import Multimodal Utils
from multimodal_utils import VisionProcessor, OllamaInterface
from config import Config

logger = logging.getLogger(__name__)

class Pipeline1Direct:
    """
    Pipeline 1: Visual-Only Baseline (PPT Slides)
    
    This pipeline ignores the transcript and extracts information ONLY from the 
    PowerPoint slides (converted to images) using a Vision LLM (Qwen2-VL).
    
    Goal: Establish the 'Visual Truth' - what does the eye see?
    """
    
    def __init__(self, model_name: str = "qwen2.5vl"):
        self.config = Config()
        # Use Qwen2.5-VL for vision tasks
        self.model_name = model_name 
        self.vision_processor = VisionProcessor()
        # We need LLMInterface for the Gemini Hallucination Score
        from models.llm_interface import LLMInterface
        self.llm = LLMInterface(model_name=self.model_name)
        self.ollama = self.llm.ollama
        
        logger.info(f"Pipeline 1 (Visual-Only) initialized with model: {self.model_name}")
    
    def process_documents(self) -> str:
        """
        Process documents for Pipeline 1 (Slides Only).
        
        Returns:
            Aggregated visual facts from all slides.
        """
        logger.info("Pipeline 1: Starting Visual Document processing")
        
        pdf_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ppt.pdf")
        if not os.path.exists(pdf_path):
            logger.error(f"PPT file not found at {pdf_path}")
            return "Error: ppt.pdf not found."
            
        # 1. Convert PDF to Images
        logger.info(f"Loading slides from: {pdf_path}")
        images = self.vision_processor.load_pdf_as_images(pdf_path)
        
        visual_extraction = []
        
        # 2. Analyze each slide
        logger.info(f"Analyzing {len(images)} slides with {self.model_name}...")
        for i, img in enumerate(images):
            # Encode to Base64
            b64_img = self.vision_processor.encode_image_to_base64(img)
            
            # Prompt for financial extraction
            prompt = (
                "You are a financial analyst. Analyze this slide. "
                "Extract all key financial numbers, trends, charts, and table data shown. "
                "Be precise with numbers and direction (up/down). "
                "If there are no financial facts, purely describe the visual layout."
            )
            
            logger.info(f"Processing Slide {i+1}...")
            description = self.ollama.analyze_image(b64_img, prompt)
            
            slide_summary = f"--- Slide {i+1} ---\n{description}\n"
            visual_extraction.append(slide_summary)
            
        # 3. Aggregate
        full_visual_context = "\n".join(visual_extraction)
        
        logger.info("Pipeline 1: Visual processing completed")
        return full_visual_context
    
    def run_pipeline(self, question: str = None) -> Dict[str, Any]:
        """
        Run the complete Pipeline 1 process.
        """
        start_time = datetime.now()
        
        if question is None:
            question = self.config.DEFAULT_QUESTION
        
        logger.info(f"Pipeline 1: Starting complete pipeline run")
        logger.info(f"Question: {question}")
        
        # Step 1: Extract Visual Facts
        visual_facts = self.process_documents()
        
        # Step 2: Generate Answer based ONLY on Visuals
        logger.info("Pipeline 1: Generating answer from Visual Facts")
        
        # We use the Chat interface for the final answer
        messages = [
            {"role": "system", "content": "You are a helpful financial assistant. Answer the user's question using ONLY the provided slide descriptions."},
            {"role": "user", "content": f"Here are the descriptions of the presentation slides:\n\n{visual_facts}\n\nQuestion: {question}"}
        ]
        
        answer = self.ollama.chat(messages)
        
        # Step 3: Compute Hallucination Score (Gemini)
        # For Visual-Only pipeline, 'textual facts' is empty because we ignored the transcript.
        logger.info("Pipeline 1: Computing Hallucination Score")
        hallucination_results = self.llm.calculate_hallucination_score(
            answer=answer,
            visual_facts=visual_facts, 
            textual_facts="N/A (Visual Only Pipeline)"
        )
        
        end_time = datetime.now()
        execution_time = (end_time - start_time).total_seconds()
        
        # Compile results
        results = {
            'pipeline_name': 'Pipeline 1: Visual-Only Baseline',
            'pipeline_type': 'visual_only',
            'model_used': self.model_name,
            'question': question,
            'processed_information': visual_facts, 
            'generated_answer': answer,
            'hallucination_score': hallucination_results,
            'execution_time': execution_time,
            'timestamp': end_time.isoformat()
        }
        
        logger.info(f"Pipeline 1: Completed in {execution_time:.2f} seconds")
        logger.info(f"Hallucination Score: {hallucination_results.get('final_score', 'Error')}")
        return results
    
    def save_results(self, results: Dict[str, Any], output_dir: str = None) -> str:
        """
        Save pipeline results.
        """
        if output_dir is None:
            output_dir = os.path.join(self.config.RESULTS_DIR, "pipeline1")
        
        os.makedirs(output_dir, exist_ok=True)
        
        results_file = os.path.join(output_dir, "pipeline1_results.json")
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
            
        return results_file
    
    def cleanup(self):
        pass

def main():
    logging.basicConfig(level=logging.INFO)
    pipeline = Pipeline1Direct()
    results = pipeline.run_pipeline()
    pipeline.save_results(results)
    print("Pipeline 1 Completed.")

if __name__ == "__main__":
    main()