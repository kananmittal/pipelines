import os
import logging
from CS50_pipelines.pipeline2_consolidation_ppt import Pipeline2ConsolidationPPT
from CS50_pipelines.pipeline3_parallel_ppt import Pipeline3ParallelPPT
from CS50_pipelines.pipeline5_consolidation_ppt import Pipeline5ConsolidationPPT
from CS50_pipelines.pipeline1_direct import Pipeline1Direct
from CS50_pipelines.pipeline4_iterative import Pipeline4Iterative

# Configure Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def get_data_folders(target_directory="definedgedata"):
    """Get all subfolders in the target directory"""
    
    # Ensure we look relative to where this script is located, not where the user terminal is
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_root = os.path.join(base_dir, target_directory)
    
    if not os.path.exists(data_root):
        # Fallback to looking inside 'data' just in case
        fallback_root = os.path.join(base_dir, "data", target_directory)
        if os.path.exists(fallback_root):
             data_root = fallback_root
        else:
             logger.error(f"Data directory '{data_root}' not found!")
             return []
        
    # Support both digit folders (edudata/0) and named folders (definedgedata/degrees)
    folders = [f for f in os.listdir(data_root) if os.path.isdir(os.path.join(data_root, f))]
    
    # If there are no subfolders, treat the root directory itself as the only target
    if not folders:
        # We represent the root directory using an empty string or '.' so os.path.join resolves safely
        return ['']
    
    # Sort folders (try numeric first for edudata, fallback to alphabetical for definedgedata)
    try:
        sorted_folders = sorted(folders, key=int)
    except ValueError:
        sorted_folders = sorted(folders)
        
    return sorted_folders

def run_batch_execution(target_directory="definedgedata"):
    """Run all 5 pipelines on every dataset folder"""
    
    data_folders = get_data_folders(target_directory)
    if not data_folders:
        logger.warning(f"No data folders found in '{target_directory}/' directory.")
        return

    logger.info(f"🚀 Found {len(data_folders)} datasets: {data_folders}")
    
    # Initialize Pipelines (Models loaded once to save time/VRAM if possible, 
    # but be careful about VRAM usage. If VRAM is tight, we might need to load/unload inside the loop.
    # Given 20GB VRAM + 128GB RAM (Offloading), we should try to keep one pipeline loaded at a time)
    
    # Fix the full_folder_path calculation to use absolute base directory
    base_dir = os.path.dirname(os.path.abspath(__file__))
    target_path = os.path.join(base_dir, target_directory)
    if not os.path.exists(target_path):
         target_path = os.path.join(base_dir, "data", target_directory)
         
    for folder_name in data_folders:
        if folder_name == '':
             full_folder_path = target_path
             display_name = target_directory
        else:
             full_folder_path = os.path.join(target_path, folder_name)
             display_name = folder_name
             
        logger.info(f"\n{'='*60}")
        logger.info(f"📂 PROCESSING DATASET: {display_name}")
        logger.info(f"{'='*60}")
        
        # --- PIPELINE 1 ---
        try:
            logger.info(f"▶️ Starting Pipeline 1 (Visual Direct) for {display_name}...")
            p1 = Pipeline1Direct()
            res1 = p1.run_pipeline(target_folder=full_folder_path)
            # p1 doesn't have a save_results explicitly listed in BasePipeline if it doesn't inherit it, but let's assume it does or we just let it run.
            if hasattr(p1, 'save_results'):
                p1.save_results(res1)
            del p1
        except Exception as e:
            logger.error(f"❌ Pipeline {1} Failed for {display_name}: {e}")
            
        # --- PIPELINE 2 ---
        try:
            logger.info(f"▶️ Starting Pipeline {2} (Consolidation) for {display_name}...")
            p2 = Pipeline2ConsolidationPPT()
            res2 = p2.run_pipeline(target_folder=full_folder_path) 
            p2.save_results(res2)
            del p2
            
        except Exception as e:
            logger.error(f"❌ Pipeline {2} Failed for {display_name}: {e}")

        # --- PIPELINE 3 ---
        try:
            logger.info(f"▶️ Starting Pipeline {3} (Parallel) for {display_name}...")
            p3 = Pipeline3ParallelPPT()
            res3 = p3.run_pipeline(target_folder=full_folder_path)
            p3.save_results(res3)
            del p3
            
        except Exception as e:
            logger.error(f"❌ Pipeline {3} Failed for {display_name}: {e}")

        # --- PIPELINE 4 ---
        try:
            logger.info(f"▶️ Starting Pipeline {4} (Iterative) for {display_name}...")
            p4 = Pipeline4Iterative()
            res4 = p4.run_pipeline(target_folder=full_folder_path)
            if hasattr(p4, 'save_results'):
                p4.save_results(res4)
            del p4
        except Exception as e:
            logger.error(f"❌ Pipeline {4} Failed for {display_name}: {e}")

        # --- PIPELINE 5 ---
        try:
            logger.info(f"▶️ Starting Pipeline {5} (Current SOTA) for {display_name}...")
            p5 = Pipeline5ConsolidationPPT()
            res5 = p5.run_pipeline(target_folder=full_folder_path)
            p5.save_results(res5)
            del p5
            
        except Exception as e:
            logger.error(f"❌ Pipeline {5} Failed for {display_name}: {e}")

if __name__ == "__main__":
    print("🚀 Starting Master Batch Execution...")
    try:
        # User requested to run on definedgedata
        run_batch_execution(target_directory="definedgedata")
    except KeyboardInterrupt:
        print("\n🛑 Execution Interrupted by User.")
    except Exception as e:
        logger.error(f"❌ Fatal Error: {e}")
    
    print("\n\n✅ Master Batch Execution Complete!")
