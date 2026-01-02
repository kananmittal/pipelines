import json
import os

NOTEBOOK_PATH = "PPT_Extractor_V3_HighQuality.ipynb"

def patch_notebook():
    if not os.path.exists(NOTEBOOK_PATH):
        print(f"❌ File not found: {NOTEBOOK_PATH}")
        return

    with open(NOTEBOOK_PATH, 'r', encoding='utf-8') as f:
        nb = json.load(f)

    print("mb Loaded notebook. Patching for %pip and Imports...")
    
    patches_applied = 0

    for cell in nb['cells']:
        if cell['cell_type'] != 'code':
            continue
            
        source_lines = cell['source']
        source_str = "".join(source_lines)
        
        # 1. Change !pip to %pip for correct kernel installation
        # This is CRITICAL for Jupyter environments
        new_source = []
        changed_pip = False
        for line in source_lines:
            if line.strip().startswith("!pip"):
                new_line = line.replace("!pip", "%pip")
                new_source.append(new_line)
                changed_pip = True
            else:
                new_source.append(line)
        
        if changed_pip:
            print("   - Changed !pip to %pip (ensures installation in current kernel)")
            cell['source'] = new_source
            patches_applied += 1

        # 2. Add verification print to Import cell
        if "import fitz" in source_str and "try:" not in source_str:
             # We want to wrap imports in a try/except or just add debug info
             # But let's just make sure it's correct.
             # The previous patch might have already touched this if it had torch.cuda
             pass

    # Save changes
    if patches_applied > 0:
        with open(NOTEBOOK_PATH, 'w', encoding='utf-8') as f:
            json.dump(nb, f, indent=2)
        print(f"✅ Successfully applied {patches_applied} additional patches.")
    else:
        print("⚠️ No new patches needed (or patterns not found).")

if __name__ == "__main__":
    patch_notebook()
