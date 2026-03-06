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

class Pipeline4Iterative:
    """
    Pipeline 4: Iterative Refinement
    
    This pipeline enhances the baseline by introducing a self-critique mechanism.
    It performs initial extraction, then critiques and refines the response.
    """
    
    def __init__(self, model_name: str = None):
        self.config = Config()
        self.model_name = model_name or self.config.DEFAULT_MODEL
        self.llm = LLMInterface(self.model_name)
        self.doc_processor = DocumentProcessor()
        self.current_folder = None
    
    def process_documents(self, target_folder: str = None) -> Dict[str, Any]:
        """
        Process documents for Pipeline 4.
        """
        logger.info("Pipeline 4: Starting document processing")
        
        # Handle Batch Folder Logic
        if target_folder:
            self.current_folder = target_folder
        else:
             if not getattr(self, 'current_folder', None):
                 base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                 edudata_dir_candidates = [
                     os.path.join(base_dir, "data", "edudata"),
                     os.path.join(base_dir, "data", "Edudata"),
                     os.path.join(base_dir, "edudata"),
                     os.path.join(base_dir, "Edudata")
                 ]
                 for candidate in edudata_dir_candidates:
                     if os.path.exists(candidate):
                         # Accept any valid subdirectory instead of just digits
                         subs = [os.path.join(candidate, d) for d in os.listdir(candidate) if os.path.isdir(os.path.join(candidate, d))]
                         # Prioritize a folder that actually has a ppt.pdf file in it
                         valid_subs = [s for s in subs if os.path.exists(os.path.join(s, "ppt.pdf")) or os.path.exists(os.path.join(s, "presentation.pdf"))]
                         
                         if valid_subs:
                             self.current_folder = sorted(valid_subs)[0]
                             logger.info(f"Defaulting to first data folder WITH visuals: {self.current_folder}")
                             break
                         elif subs:
                             self.current_folder = sorted(subs)[0]
                             logger.info(f"Defaulting to first data folder: {self.current_folder}")
                             break

        if not getattr(self, 'current_folder', None):
             logger.error("No target folder specified for processing.")
             return {}
             
        # Find transcript
        transcript_path = None
        for fname in self.config.INPUT_FILES.get("transcript", []):
            path = os.path.join(self.current_folder, fname)
            if os.path.exists(path):
                transcript_path = path
                break
                
        if not transcript_path:
             logger.error("No transcript found")
             return {}
             
        if transcript_path.lower().endswith('.docx'):
            transcript = self.doc_processor.read_docx(transcript_path)
        else:
            with open(transcript_path, 'rb') as f:
                transcript = self.doc_processor.read_pdf(f)
                
        # Preprocess text
        transcript = self.doc_processor.preprocess_text(transcript)
        
        logger.info(f"Loaded transcript: {len(transcript)} characters")
        logger.info("Pipeline 4: Performing initial extraction (Text Only)")
        initial_extraction = self.llm.extract_information(transcript, "financial")
        
        # --- NEW: Visual Grounding ---
        from multimodal_utils import VisionProcessor, OllamaInterface
        logger.info("Pipeline 4: Visual Grounding - Loading PPT")
        pdf_path = os.path.join(self.current_folder, "ppt.pdf")
        if not os.path.exists(pdf_path):
             pdf_path = os.path.join(self.current_folder, "presentation.pdf")
             if not os.path.exists(pdf_path):
                 # Fallback: grab the first .pdf file in the folder
                 pdf_files = [f for f in os.listdir(self.current_folder) if f.lower().endswith('.pdf')]
                 if pdf_files:
                     pdf_path = os.path.join(self.current_folder, pdf_files[0])
                     logger.info(f"Using auto-detected PDF for visuals: {pdf_path}")
                 else:
                     pdf_path = "/nonexistent/path/for/safety.pdf" # will fail the next check gracefully
        if os.path.exists(pdf_path):
            vp = VisionProcessor()
            ollama = OllamaInterface()
            images = vp.load_pdf_as_images(pdf_path)
            
            # Analyze slides for specific claims made in initial_extraction?
            # For simplicity, we get a full visual summary first
            slide_facts = []
            for i, img in enumerate(images):
                b64 = vp.encode_image_to_base64(img)
                desc = ollama.analyze_image(b64, "Extract all financial numbers and trends from this slide for verification purposes.")
                slide_facts.append(f"Slide {i+1}: {desc}")
            
            visual_evidence = "\n".join(slide_facts)
            
            # Step 2: Visual Critique Step
            logger.info("Pipeline 4: performing Critique against Visual Evidence")
            critique_prompt = f"""You are a Fact-Checker.
Your Goal: Verify the Initial Extraction against the Visual Evidence (Slides).
- The Initial Extraction came from a transcript.
- The Visual Evidence comes from the official presentation slides.

Rules:
1. If a number in the Extraction matches the Visuals, keep it.
2. If a number in the Extraction CONTRADICTS the Visuals, CORRECT IT using the Visual value.
3. If the Extraction mentions something not in Visuals, mark it as "Unverified by Visuals".
4. Output a Final Refined Summary that is visually grounded.

Initial Extraction:
{initial_extraction}

Visual Evidence:
{visual_evidence}

Refined Summary:"""
            
            refinement_response = self.llm.generate_single_response(
                critique_prompt,
                options={'temperature': 0.1}
            )
            refined_extraction = refinement_response['response']
        else:
            logger.warning("PPT not found. Falling back to text-only critique.")
            refined_extraction = self.llm.critique_and_refine(transcript, initial_extraction)
        
        logger.info("Pipeline 4: Document processing completed")
        return {
            "processed_text": refined_extraction,
            "visual_evidence": visual_evidence if 'visual_evidence' in locals() else "No visual evidence available.",
            "transcript": transcript
        }
    
    def run_pipeline(self, question: str = None, target_folder: str = None) -> Dict[str, Any]:
        """
        Run the complete Pipeline 4 process.
        
        Args:
            question: Question to ask (uses default if None)
            
        Returns:
            Dictionary with all results
        """
        start_time = datetime.now()
        
        if question is None:
            question = self.config.DEFAULT_QUESTION
        
        logger.info(f"Pipeline 4: Starting complete pipeline run")
        logger.info(f"Question: {question}")
        
        # Step 1: Process documents (initial extraction + self-critique + refinement)
        docs_data = self.process_documents(target_folder=target_folder)
        if not docs_data:
             logger.error("Processing failed")
             return {}
             
        processed_info = docs_data["processed_text"]
        visual_evidence = docs_data.get("visual_evidence", "")
        transcript = docs_data.get("transcript", "")
        
        # Step 2: Generate answers
        logger.info("Pipeline 4: Generating answers")
        qa_results = self.llm.generate_qa_responses(processed_info, question)
        
        # Step 3: Extract answers for clustering
        answers = [response['response'] for response in qa_results['multiple_responses']]
        
        # Step 3: Compute Hallucination Score (Gemini)
        logger.info("Pipeline 4: Computing Hallucination Score")
        # For Pipeline 4, we compare the Refined Answer against the original Text + Visual Evidence
        hallucination_results = self.llm.calculate_hallucination_score(
            answer=qa_results['best_response']['response'],
            visual_facts=visual_evidence, # From the Visual Critique step
            textual_facts=transcript # From the original transcript
        )
        
        end_time = datetime.now()
        execution_time = (end_time - start_time).total_seconds()
        
        # Compile results
        results = {
            'pipeline_name': 'Pipeline 4: Iterative Refinement',
            'pipeline_type': 'iterative_refinement',
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
        
        logger.info(f"Pipeline 4: Completed in {execution_time:.2f} seconds")
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
            output_dir = os.path.join(self.config.RESULTS_DIR, "pipeline4")
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Save main results
        results_file = os.path.join(output_dir, "pipeline4_results.json")
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        # Save detailed log
        log_file = os.path.join(output_dir, "pipeline4_detailed_log.txt")
        with open(log_file, 'w', encoding='utf-8') as f:
            f.write(f"Pipeline 4: Iterative Refinement - Detailed Log\n")
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
            
        logger.info(f"Pipeline 4 results saved to: {results_file}")
        return results_file
    
    def cleanup(self):
        """Clean up resources."""
        if hasattr(self, 'clusterer'):
            self.clusterer.cleanup()

def main():
    """Main function for running Pipeline 4 standalone."""
    logging.basicConfig(level=logging.INFO)
    
    # Create config and directories
    config = Config()
    config.create_directories()
    
    # Run pipeline
    pipeline = Pipeline4Iterative()
    results = pipeline.run_pipeline()
    
    # Save results
    pipeline.save_results(results)
    
    # Cleanup
    pipeline.cleanup()
    
    print(f"Pipeline 4 completed successfully!")
    print(f"Semantic entropy: {results['entropy_metrics']['semantic_entropy']:.4f}")
    print(f"Confidence level: {results['interpretation']['confidence_level']}")

if __name__ == "__main__":
    main()