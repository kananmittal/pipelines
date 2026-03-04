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

def get_data_folders(data_root="data"):
    """Get all subfolders in the data directory"""
    if not os.path.exists(data_root):
        logger.error(f"Data directory '{data_root}' not found!")
        return []
        
    folders = [f for f in os.listdir(data_root) if os.path.isdir(os.path.join(data_root, f))]
    # Filter only digit folders for edudata
    folders = [f for f in folders if f.isdigit()]
    # Sort for consistent execution order
    return sorted(folders, key=int)

def run_batch_execution():
    """Run all 3 pipelines on every dataset folder"""
    
    data_folders = get_data_folders()
    if not data_folders:
        logger.warning("No data folders found in 'data/' directory.")
        return

    logger.info(f"🚀 Found {len(data_folders)} datasets: {data_folders}")
    
    # Initialize Pipelines (Models loaded once to save time/VRAM if possible, 
    # but be careful about VRAM usage. If VRAM is tight, we might need to load/unload inside the loop.
    # Given 20GB VRAM + 128GB RAM (Offloading), we should try to keep one pipeline loaded at a time)
    
    for folder_name in data_folders:
        full_folder_path = os.path.abspath(os.path.join("data", folder_name))
        logger.info(f"\n{'='*60}")
        logger.info(f"📂 PROCESSING DATASET: {folder_name}")
        logger.info(f"{'='*60}")
        
        # --- PIPELINE 1 ---
        try:
            logger.info(f"▶️ Starting Pipeline 1 (Visual Direct) for {folder_name}...")
            p1 = Pipeline1Direct()
            res1 = p1.run_pipeline(target_folder=full_folder_path)
            # p1 doesn't have a save_results explicitly listed in BasePipeline if it doesn't inherit it, but let's assume it does or we just let it run.
            if hasattr(p1, 'save_results'):
                p1.save_results(res1)
            del p1
        except Exception as e:
            logger.error(f"❌ Pipeline 1 Failed for {folder_name}: {e}")
            
        # --- PIPELINE 2 ---
        try:
            logger.info(f"▶️ Starting Pipeline 2 (Consolidation) for {folder_name}...")
            p2 = Pipeline2ConsolidationPPT()
            res2 = p2.run_pipeline(target_folder=full_folder_path) 
            p2.save_results(res2)
            del p2
            
        except Exception as e:
            logger.error(f"❌ Pipeline 2 Failed for {folder_name}: {e}")

        # --- PIPELINE 3 ---
        try:
            logger.info(f"▶️ Starting Pipeline 3 (Parallel) for {folder_name}...")
            p3 = Pipeline3ParallelPPT()
            res3 = p3.run_pipeline(target_folder=full_folder_path)
            p3.save_results(res3)
            del p3
            
        except Exception as e:
            logger.error(f"❌ Pipeline 3 Failed for {folder_name}: {e}")

        # --- PIPELINE 4 ---
        try:
            logger.info(f"▶️ Starting Pipeline 4 (Iterative) for {folder_name}...")
            p4 = Pipeline4Iterative()
            res4 = p4.run_pipeline(target_folder=full_folder_path)
            if hasattr(p4, 'save_results'):
                p4.save_results(res4)
            del p4
        except Exception as e:
            logger.error(f"❌ Pipeline 4 Failed for {folder_name}: {e}")

        # --- PIPELINE 5 ---
        try:
            logger.info(f"▶️ Starting Pipeline 5 (Current SOTA) for {folder_name}...")
            p5 = Pipeline5ConsolidationPPT()
            res5 = p5.run_pipeline(target_folder=full_folder_path)
            p5.save_results(res5)
            del p5
            
        except Exception as e:
            logger.error(f"❌ Pipeline 5 Failed for {folder_name}: {e}")

if __name__ == "__main__":
    print("🚀 Starting Master Batch Execution...")
    try:
        run_batch_execution()
    except KeyboardInterrupt:
        print("\n🛑 Execution Interrupted by User.")
    except Exception as e:
        logger.error(f"❌ Fatal Error: {e}")
    
    print("\n\n✅ Master Batch Execution Complete!")
