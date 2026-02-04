
import pandas as pd
import numpy as np
import os

# Configuration
NUM_DOCS = 100
QS_PER_DOC = 10
OUTPUT_FILE = "results/human_vs_gpt_eval.csv"

# Ensure output directory
os.makedirs("results", exist_ok=True)

def determine_category(gpt, human):
    diff = gpt - human
    abs_diff = abs(diff)
    
    # Thresholds for 0-1 scale
    # 0 = No Hallucination, 1 = Complete Hallucination
    
    if abs_diff <= 0.05:
        # Very close
        if gpt < 0.1 and human < 0.1:
            return "Perfect Match (No Hallucination)"
        elif gpt > 0.9 and human > 0.9:
            return "Perfect Match (Both Flagged)"
        else:
            return "Perfect Match"
            
    elif abs_diff <= 0.2:
        # Small difference
        return "Strong Agreement"
        
    elif abs_diff <= 0.4:
        # Moderate difference
        if diff < 0:
            # GPT gave lower score (less hallucinated) than human -> GPT was lenient
            return "Disagreement (GPT too lenient)"
        else:
            # GPT gave higher score -> GPT was too strict
            return "Disagreement (GPT too strict)"
            
    else:
        # Large difference (> 0.4)
        if diff < 0:
            return "Major Disagreement (GPT missed error)"
        else:
            return "Major Disagreement (GPT false positive)"

def generate_data():
    records = []
    
    # Random seed for reproducibility
    np.random.seed(42)
    
    for doc_id in range(1, NUM_DOCS + 1):
        for q_id in range(1, QS_PER_DOC + 1):
            
            # Simulate Human Score (Ground Truth)
            # Use a bimodal distribution: mostly valid (low score) or hallucinatory (high score)
            if np.random.random() < 0.7:
                human_score = np.random.beta(1, 5) # Skewed towards 0
            else:
                human_score = np.random.beta(5, 1) # Skewed towards 1
            
            # Simulate GPT Score (Correlated with noise)
            # 70% chance of being close (Good model), 30% chance of divergence
            if np.random.random() < 0.7:
                noise = np.random.normal(0, 0.05)
                gpt_score = human_score + noise
            else:
                # Occasional big errors
                noise = np.random.normal(0, 0.3)
                gpt_score = human_score + noise
            
            # Clip to 0-1
            human_score = np.clip(human_score, 0.0, 1.0)
            gpt_score = np.clip(gpt_score, 0.0, 1.0)
            
            # Generate Category
            category = determine_category(gpt_score, human_score)
            
            records.append({
                "Document_ID": doc_id,
                "Question_ID": f"D{doc_id}_Q{q_id}",
                "GPT_Hallucination_Score": round(gpt_score, 2),
                "Human_Score": round(human_score, 2),
                "Score_Difference": round(gpt_score - human_score, 2),
                "Agreement_Status": category
            })
            
    df = pd.DataFrame(records)
    
    # Save comparison stats summary
    print("\n=== Dataset Summary ===")
    print(df['Agreement_Status'].value_counts())
    
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"\n✅ Dataset generated at: {OUTPUT_FILE}")

if __name__ == "__main__":
    generate_data()
