import logging
import json
import os
from datetime import datetime
from typing import Dict, Any, List
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.llm_interface import LLMInterface
from utils.document_processor import DocumentProcessor
from config import Config

logger = logging.getLogger(__name__)

class Pipeline5ConsolidationIterative:
    """
    Pipeline 5: Consolidation with Iterative Refinement
    
    This pipeline implements Algorithm 5:
    1. Consolidate (Transcript + Notes + Visuals)
    2. Extract Information
    3. Critique Extraction
    4. Refine Extraction
    5. Answer Questions
    """
    
    def __init__(self, model_name: str = None):
        self.config = Config()
        self.model_name = model_name or self.config.DEFAULT_MODEL
        self.llm = LLMInterface(self.model_name)
        self.doc_processor = DocumentProcessor()
    
    def process_documents(self) -> Dict[str, Any]:
        """
        Process documents implementing the Consolidate -> Extract -> Critique -> Refine flow.
        """
        logger.info("Pipeline 5: Starting document processing")
        
        # --- Data Loading ---
        transcript = self.doc_processor.load_transcript()
        notes = self.doc_processor.load_notes()
        transcript = self.doc_processor.preprocess_text(transcript)
        notes = self.doc_processor.preprocess_text(notes)
        
        # --- Visual Path (integrated into consolidation) ---
        from multimodal_utils import VisionProcessor, OllamaInterface
        pdf_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ppt.pdf")
        
        visual_context = ""
        if os.path.exists(pdf_path):
            vp = VisionProcessor()
            # Use explicit vision model for PPT analysis
            ollama = OllamaInterface(model_name=self.config.VISION_MODEL)
            images = vp.load_pdf_as_images(pdf_path)
            slide_summaries = []
            for i, img in enumerate(images):
                b64 = vp.encode_image_to_base64(img)
                desc = ollama.analyze_image(b64, "Summarize key financial data/trends in this slide.")
                slide_summaries.append(f"[Slide {i+1}]: {desc}")
            visual_context = "\n\n=== VISUAL SUMMARIES ===\n" + "\n".join(slide_summaries)
        else:
            logger.warning("ppt.pdf not found, skipping visual context.")

        # --- Algorithm Step 1: Consolidate ---
        logger.info("Pipeline 5: Step 1 - Consolidate")
        # Merging visuals into transcript for consolidation context
        text_with_visuals = transcript + visual_context
        consolidated_document = self.llm.consolidate_information(text_with_visuals, notes)
        
        # --- Algorithm Step 2: Initial Extraction ---
        logger.info("Pipeline 5: Step 2 - Initial Extraction")
        initial_extraction = self.llm.extract_information(consolidated_document, "financial")
        
        # --- Algorithm Step 3: Critique ---
        logger.info("Pipeline 5: Step 3 - Critique")
        # We critique the extraction against the 'Ground Truth' which for this step is the Consolidated Doc
        # Note: We could also pass raw visuals here for extra grounding
        critique_prompt = f"""You are a Critical Reviewer.
Original Source Info:
{consolidated_document[:5000]}

Extracted Info:
{initial_extraction}

Task: Identify missing details, inaccuracies, or hallucinations in the Extracted Info compared to the Source.
Provide specific feedback."""
        
        critique_response = self.llm.generate_single_response(critique_prompt, options={'temperature': 0.1})
        critique = critique_response['response']
        
        # --- Algorithm Step 4: Refine ---
        logger.info("Pipeline 5: Step 4 - Refine")
        refined_extraction = self.llm.critique_and_refine(initial_extraction, critique)
        
        logger.info("Pipeline 5: Document processing completed")
        
        return {
            "processed_text": refined_extraction,
            "consolidated_document": consolidated_document,
            "initial_extraction": initial_extraction,
            "critique": critique,
            "visual_context": visual_context,
            "transcript": transcript,
            "notes": notes
        }
    
    def run_pipeline(self, question: str = None) -> Dict[str, Any]:
        """
        Run complete Pipeline 5.
        """
        start_time = datetime.now()
        
        if question is None:
            question = self.config.DEFAULT_QUESTION
            
        logger.info(f"Pipeline 5: Starting run for question: {question}")
        
        # Process Documents (Steps 1-4)
        docs_data = self.process_documents()
        refined_extraction = docs_data["processed_text"]
        
        # --- Algorithm Step 5: Answer ---
        logger.info("Pipeline 5: Step 5 - Answer")
        qa_results = self.llm.generate_qa_responses(refined_extraction, question)
        
        # Validation / Hallucination Score
        logger.info("Pipeline 5: Computing Hallucination Score")
        hallucination_results = self.llm.calculate_hallucination_score(
            answer=qa_results['best_response']['response'],
            visual_facts=docs_data.get("visual_context", ""),
            textual_facts=docs_data.get("consolidated_document", "") # Check against consolidated doc
        )
        
        end_time = datetime.now()
        execution_time = (end_time - start_time).total_seconds()
        
        results = {
            'pipeline_name': 'Pipeline 5: Consolidation with Iterative Refinement',
            'pipeline_type': 'consolidation_iterative',
            'model_used': self.model_name,
            'question': question,
            'processed_information': refined_extraction,
            'intermediate_steps': {
                'consolidated_document': docs_data['consolidated_document'],
                'initial_extraction': docs_data['initial_extraction'],
                'critique': docs_data['critique']
            },
            'best_answer': {
                'answer': qa_results['best_response']['response'],
                'temperature': qa_results['best_response']['parameters'].get('temperature', 0.1),
                'generation_time': qa_results['best_response']['generation_time']
            },
            'hallucination_score': hallucination_results,
            'execution_time': execution_time,
            'timestamp': end_time.isoformat(),
        }
        
        logger.info(f"Pipeline 5: Completed in {execution_time:.2f} seconds")
        logger.info(f"Hallucination Score: {hallucination_results.get('final_score', 'Error')}")
        
        return results

    def save_results(self, results: Dict[str, Any], output_dir: str = None) -> str:
        if output_dir is None:
            output_dir = os.path.join(self.config.RESULTS_DIR, "pipeline5")
        
        os.makedirs(output_dir, exist_ok=True)
        
        results_file = os.path.join(output_dir, "pipeline5_results.json")
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
            
        # Detailed Log
        log_file = os.path.join(output_dir, "pipeline5_detailed_log.txt")
        with open(log_file, 'w', encoding='utf-8') as f:
            f.write(f"Pipeline 5: Consolidation + Iterative Refinement - Log\n{'='*50}\n\n")
            f.write(f"Question: {results['question']}\n\n")
            
            f.write(f"--- 1. CONSOLIDATED DOCUMENT ---\n{results['intermediate_steps']['consolidated_document']}\n\n")
            f.write(f"--- 2. INITIAL EXTRACTION ---\n{results['intermediate_steps']['initial_extraction']}\n\n")
            f.write(f"--- 3. CRITIQUE ---\n{results['intermediate_steps']['critique']}\n\n")
            f.write(f"--- 4. REFINED EXTRACTION ---\n{results['processed_information']}\n\n")
            f.write(f"--- 5. FINAL ANSWER ---\n{results['best_answer']['answer']}\n\n")
            f.write(f"--- VALIDATION ---\nHallucination Score: {results['hallucination_score'].get('final_score')}\n")
            
        logger.info(f"Pipeline 5 results saved to: {results_file}")
        return results_file
    
    def cleanup(self):
        pass

def main():
    logging.basicConfig(level=logging.INFO)
    p = Pipeline5ConsolidationIterative()
    res = p.run_pipeline()
    p.save_results(res)
    print("Pipeline 5 Completed.")

if __name__ == "__main__":
    main()
