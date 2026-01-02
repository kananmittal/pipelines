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

class Pipeline2Consolidation:
    """
    Pipeline 2: Document Consolidation First
    
    This pipeline consolidates transcript and notes into a single document
    before performing extraction.
    """
    
    def __init__(self, model_name: str = None):
        self.config = Config()
        self.model_name = model_name or self.config.DEFAULT_MODEL
        self.llm = LLMInterface(self.model_name)
        self.doc_processor = DocumentProcessor()
    
    def process_documents(self) -> Dict[str, Any]:
        """
        Process documents for Pipeline 2.
        
        Returns:
            Processed information string
        """
        logger.info("Pipeline 2: Starting document processing")
        
        # Load transcript (pages 2-14) and notes (all pages)
        transcript = self.doc_processor.load_transcript()
        notes = self.doc_processor.load_notes()
        
        # Preprocess text
        transcript = self.doc_processor.preprocess_text(transcript)
        notes = self.doc_processor.preprocess_text(notes)
        
        logger.info(f"Loaded transcript: {len(transcript)} characters")
        logger.info(f"Loaded notes: {len(notes)} characters")
        
        # --- NEW: Visual Path ---
        from multimodal_utils import VisionProcessor, OllamaInterface
        
        logger.info("Pipeline 2: Processing Visuals from PPT")
        pdf_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ppt.pdf")
        
        visual_context = ""
        if os.path.exists(pdf_path):
            vp = VisionProcessor()
            ollama = OllamaInterface() # Uses default qwen2.5-vl
            
            images = vp.load_pdf_as_images(pdf_path)
            logger.info(f"Analyzing {len(images)} slides for consolidation...")
            
            slide_summaries = []
            for i, img in enumerate(images):
                b64 = vp.encode_image_to_base64(img)
                # Brief prompt for consolidation context
                desc = ollama.analyze_image(b64, "Summarize the key financial data and trends in this slide.")
                slide_summaries.append(f"[Slide {i+1}]: {desc}")
            
            visual_context = "\n\n=== VISUAL SLIDE SUMMARIES ===\n" + "\n".join(slide_summaries)
        else:
            logger.warning("ppt.pdf not found, skipping visual context.")
            
        # Append visual context to transcript for consolidation
        transcript_plus_visuals = transcript + visual_context
        
        # Step 1: Consolidate documents first
        logger.info("Pipeline 2: Consolidating documents (Transcript + Notes + Visuals)")
        consolidated_text = self.llm.consolidate_information(transcript_plus_visuals, notes)
        
        # Step 2: Extract information from consolidated text
        logger.info("Pipeline 2: Extracting information from consolidated text")
        extracted_info = self.llm.extract_information(consolidated_text, "financial")
        
        logger.info("Pipeline 2: Document processing completed")
        return {
            "processed_text": extracted_info,
            "visual_context": visual_context,
            "transcript": transcript
        }
    
    def run_pipeline(self, question: str = None) -> Dict[str, Any]:
        """
        Run the complete Pipeline 2 process.
        
        Args:
            question: Question to ask (uses default if None)
            
        Returns:
            Dictionary with all results
        """
        start_time = datetime.now()
        
        if question is None:
            question = self.config.DEFAULT_QUESTION
        
        logger.info(f"Pipeline 2: Starting complete pipeline run")
        logger.info(f"Question: {question}")
        
        # Step 1: Process documents (consolidate first, then extract)
        docs_data = self.process_documents()
        processed_info = docs_data["processed_text"]
        visual_context = docs_data.get("visual_context", "")
        transcript = docs_data.get("transcript", "")
        
        # Step 2: Generate answers
        logger.info("Pipeline 2: Generating answers")
        qa_results = self.llm.generate_qa_responses(processed_info, question)
        
        # Step 3: Compute Hallucination Score (Gemini)
        logger.info("Pipeline 2: Computing Hallucination Score")
        # For consolidated pipeline, we treat the consolidated text as 'Textual/Visual' merged context
        # But to fit the formula, we can pass the consolidated text as textual facts, and maybe empty visuals?
        # Or better: Pass the raw extracted visual context we appended.
        hallucination_results = self.llm.calculate_hallucination_score(
            answer=qa_results['best_response']['response'],
            visual_facts=visual_context, # The raw visual summaries
            textual_facts=transcript # The raw transcript
        )
        
        end_time = datetime.now()
        execution_time = (end_time - start_time).total_seconds()
        
        # Compile results
        results = {
            'pipeline_name': 'Pipeline 2: Document Consolidation First',
            'pipeline_type': 'consolidation_first',
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
        
        logger.info(f"Pipeline 2: Completed in {execution_time:.2f} seconds")
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
            output_dir = os.path.join(self.config.RESULTS_DIR, "pipeline2")
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Save main results
        results_file = os.path.join(output_dir, "pipeline2_results.json")
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        # Save detailed log
        log_file = os.path.join(output_dir, "pipeline2_detailed_log.txt")
        with open(log_file, 'w', encoding='utf-8') as f:
            f.write(f"Pipeline 2: Document Consolidation First - Detailed Log\n")
            f.write(f"{'='*50}\n\n")
            f.write(f"Timestamp: {results['timestamp']}\n")
            f.write(f"Model: {results['model_used']}\n")
            f.write(f"Question: {results['question']}\n")
            f.write(f"Execution Time: {results['execution_time']:.2f} seconds\n\n")
            
            f.write(f"PROCESSED INFORMATION:\n")
            f.write(f"{'-'*30}\n")
            f.write(f"{results['processed_information']}\n\n")
            
            f.write(f"SEMANTIC ENTROPY METRICS:\n")
            f.write(f"{'-'*30}\n")
            for key, value in results['entropy_metrics'].items():
                f.write(f"{key}: {value}\n")
            f.write(f"\n")
            
            f.write(f"INTERPRETATION:\n")
            f.write(f"{'-'*30}\n")
            for key, value in results['interpretation'].items():
                f.write(f"{key}: {value}\n")
            f.write(f"\n")
            
            f.write(f"CLUSTER VISUALIZATION:\n")
            f.write(f"{'-'*30}\n")
            f.write(self.clusterer.visualize_clusters())
            
        logger.info(f"Pipeline 2 results saved to: {results_file}")
        return results_file
    
    def cleanup(self):
        """Clean up resources."""
        if hasattr(self, 'clusterer'):
            self.clusterer.cleanup()

def main():
    """Main function for running Pipeline 2 standalone."""
    logging.basicConfig(level=logging.INFO)
    
    # Create config and directories
    config = Config()
    config.create_directories()
    
    # Run pipeline
    pipeline = Pipeline2Consolidation()
    results = pipeline.run_pipeline()
    
    # Save results
    pipeline.save_results(results)
    
    # Cleanup
    pipeline.cleanup()
    
    print(f"Pipeline 2 completed successfully!")
    print(f"Semantic entropy: {results['entropy_metrics']['semantic_entropy']:.4f}")
    print(f"Confidence level: {results['interpretation']['confidence_level']}")

if __name__ == "__main__":
    main()