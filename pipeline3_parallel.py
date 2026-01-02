import logging
import json
import os
from datetime import datetime
from typing import Dict, Any
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.llm_interface import LLMInterface
from utils.document_processor import DocumentProcessor
from config import Config

logger = logging.getLogger(__name__)

class Pipeline3Parallel:
    """
    Pipeline 3: Parallel Extraction with Reconciliation
    
    This pipeline extracts information from transcript and notes separately
    in parallel, then reconciles the results into a unified representation.
    """
    
    def __init__(self, model_name: str = None):
        self.config = Config()
        self.model_name = model_name or self.config.DEFAULT_MODEL
        self.llm = LLMInterface(self.model_name)
        self.doc_processor = DocumentProcessor()
    
    def process_documents(self) -> Dict[str, Any]:
        """
        Process documents for Pipeline 3.
        
        Returns:
            Processed information string
        """
        logger.info("Pipeline 3: Starting document processing")
        
        # Load transcript (pages 2-14) and notes (all pages)
        transcript = self.doc_processor.load_transcript()
        notes = self.doc_processor.load_notes()
        
        # Preprocess text
        transcript = self.doc_processor.preprocess_text(transcript)
        notes = self.doc_processor.preprocess_text(notes)
        
        logger.info(f"Loaded transcript: {len(transcript)} characters")
        logger.info(f"Loaded notes: {len(notes)} characters")
        
        # Step 1: Parallel extraction from each document
        logger.info("Pipeline 3: Performing parallel extraction")
        
        # Extract from transcript
        logger.info("Pipeline 3: Extracting from transcript")
        transcript_extraction = self.llm.extract_information(transcript, "financial")
        
        # Extract from notes
        logger.info("Pipeline 3: Extracting from notes")
        notes_extraction = self.llm.extract_information(notes, "financial")
        
        # --- NEW: Extract from Visuals ---
        from multimodal_utils import VisionProcessor, OllamaInterface
        logger.info("Pipeline 3: Extracting from Visuals (PPT)")
        pdf_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ppt.pdf")
        
        visual_extraction = ""
        if os.path.exists(pdf_path):
            vp = VisionProcessor()
            ollama = OllamaInterface()
            images = vp.load_pdf_as_images(pdf_path)
            
            visual_facts_list = []
            for i, img in enumerate(images):
                b64 = vp.encode_image_to_base64(img)
                # Prompt specifically for FACTS to cross-check
                prompt = "Extract all verifiable financial facts, numbers, and trends visible in this slide. Do not hallucinate."
                fact = ollama.analyze_image(b64, prompt)
                visual_facts_list.append(f"Slide {i+1}: {fact}")
                import time
                time.sleep(5) # Safety delay to prevent Ollama overload
            
            visual_extraction = "\n".join(visual_facts_list)
        else:
            logger.warning("ppt.pdf not found, skipping visual extraction.")

        
        # Step 2: Reconcile the parallel extractions
        logger.info("Pipeline 3: Reconciling parallel extractions")
        reconciled_info = self.reconcile_extractions(transcript_extraction, notes_extraction, visual_extraction)
        
        logger.info("Pipeline 3: Document processing completed")
        return reconciled_info
    
    def reconcile_extractions(self, transcript_extraction: str, notes_extraction: str, visual_extraction: str = "") -> str:
        """
        Reconcile information extracted from transcript, notes, and visuals.
        """
        reconciliation_prompt = f"""You are a strict Judge evaluating financial claims from multiple sources.
You have three sources of information:
1. Transcript (Spoken by executives)
2. Notes (Written summary)
3. Visuals (Charts and Slides from the presentation)

Your Task:
1. Create a unified summary of the financial performance.
2. **CROSS-MODAL VERIFICATION (CRITICAL)**: Compare the Textual claims (Transcript/Notes) against the Visual Evidence.
   - If the Text says "Revenue up" but Visual chart shows a decline, FLAG THIS as a "Possible Hallucination" or "Contradiction".
   - Explicitly list any discrepancies between what was said and what was shown.
   - If Visuals confirm the Text, note that as "Visually Verified".

Transcript Extraction:
{transcript_extraction}

Notes Extraction:
{notes_extraction}

Visual Evidence (Slides):
{visual_extraction}

Reconciled Summary & Verification Report:"""


        reconciled_response = self.llm.generate_single_response(
            reconciliation_prompt,
            options={
                'temperature': 0.1, # Lower temperature for judging
                'top_p': 0.9,
                'num_predict': 2048
            }
        )
        
        return {
            'reconciled_info': reconciled_response['response'],
            'visual_extraction': visual_extraction,
            'transcript_extraction': transcript_extraction,
            'notes_extraction': notes_extraction
        }
    
    def run_pipeline(self, question: str = None) -> Dict[str, Any]:
        """
        Run the complete Pipeline 3 process.
        
        Args:
            question: Question to ask (uses default if None)
            
        Returns:
            Dictionary with all results
        """
        start_time = datetime.now()
        
        if question is None:
            question = self.config.DEFAULT_QUESTION
        
        logger.info(f"Pipeline 3: Starting complete pipeline run")
        logger.info(f"Question: {question}")
        
        # Step 1: Process documents (parallel extraction + reconciliation)
        docs_data = self.process_documents()
        processed_info = docs_data['reconciled_info']
        visual_extraction = docs_data['visual_extraction']
        
        # Step 2: Generate answers
        logger.info("Pipeline 3: Generating answers")
        qa_results = self.llm.generate_qa_responses(processed_info, question)
        
        # Step 3: Compute Hallucination Score (Gemini)
        logger.info("Pipeline 3: Computing Hallucination Score")
        hallucination_results = self.llm.calculate_hallucination_score(
            answer=qa_results['best_response']['response'],
            visual_facts=visual_extraction, 
            textual_facts=f"Transcript Ext: {docs_data['transcript_extraction']}\nNotes Ext: {docs_data['notes_extraction']}"
        )
        
        end_time = datetime.now()
        execution_time = (end_time - start_time).total_seconds()
        
        # Compile results
        results = {
            'pipeline_name': 'Pipeline 3: Parallel Extraction with Reconciliation',
            'pipeline_type': 'parallel_extraction',
            'model_used': self.model_name,
            'question': question,
            'processed_information': processed_info,
            'best_answer': {
                'answer': qa_results['best_response']['response'],
                'temperature': qa_results['best_response']['parameters'].get('temperature', 0.1),
                'generation_time': qa_results['best_response']['generation_time']
            },
            'hallucination_score': hallucination_results,
            'execution_time': execution_time,
            'timestamp': end_time.isoformat(),
        }
        
        logger.info(f"Pipeline 3: Completed in {execution_time:.2f} seconds")
        logger.info(f"Hallucination Score: {hallucination_results.get('final_score', 'Error')}")
        
        return results
    
    def save_results(self, results: Dict[str, Any], output_dir: str = None) -> str:
        """
        Save pipeline results to file.
        
        Args:
            results: Results dictionary
            output_dir: Output directory (uses default if None)
            
        Returns:
            Path to saved file
        """
        if output_dir is None:
            output_dir = os.path.join(self.config.RESULTS_DIR, "pipeline3")
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Save main results
        results_file = os.path.join(output_dir, "pipeline3_results.json")
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        # Save detailed log
        log_file = os.path.join(output_dir, "pipeline3_detailed_log.txt")
        with open(log_file, 'w', encoding='utf-8') as f:
            f.write(f"Pipeline 3: Parallel Extraction with Reconciliation - Detailed Log\n")
            f.write(f"{'='*50}\n\n")
            f.write(f"Timestamp: {results['timestamp']}\n")
            f.write(f"Model: {results['model_used']}\n")
            f.write(f"Question: {results['question']}\n")
            f.write(f"Execution Time: {results['execution_time']:.2f} seconds\n\n")
            
            f.write(f"PROCESSED INFORMATION:\n")
            f.write(f"{'-'*30}\n")
            f.write(f"{results['processed_information']}\n\n")
            
            f.write(f"HALLUCINATION SCORE:\n")
            f.write(f"{'-'*30}\n")
            f.write(f"Final Score: {results['hallucination_score'].get('final_score', 'N/A')}\n")
            f.write(json.dumps(results['hallucination_score'], indent=2))
            
        logger.info(f"Pipeline 3 results saved to: {results_file}")
        return results_file
    
    def cleanup(self):
        """Clean up resources."""
        pass

def main():
    """Main function for running Pipeline 3 standalone."""
    logging.basicConfig(level=logging.INFO)
    
    # Create config and directories
    config = Config()
    config.create_directories()
    
    # Run pipeline
    pipeline = Pipeline3Parallel()
    results = pipeline.run_pipeline()
    
    # Save results
    pipeline.save_results(results)
    
    # Cleanup
    pipeline.cleanup()
    
    print(f"Pipeline 3 completed successfully!")
    print(f"Hallucination Score: {results['hallucination_score'].get('final_score')}")

if __name__ == "__main__":
    main()