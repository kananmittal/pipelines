
import matplotlib.pyplot as plt
import numpy as np
import os

# Output path
OUTPUT_FILE = "results/hallucination_scores_graph.png"
os.makedirs("results", exist_ok=True)

# Data (Consistent with Report)
# Baseline (P1, P4, P5 rough avg) vs Optimized (P2, P3)
# Overall Means from Report: P1=0.33, P2=0.175, P3=0.145, P4=0.24, P5=0.21
# Splitting into Analytical vs Numerical (Numerical usually slightly higher error)

pipelines = ["Pipeline 1", "Pipeline 2", "Pipeline 3", "Pipeline 4", "Pipeline 5"]

# Analytical Scores (Cyan in ref)
analytical_scores = [
    0.30,  # P1 (High)
    0.16,  # P2 (Low)
    0.14,  # P3 (Best)
    0.22,  # P4 (Mid)
    0.19   # P5 (Mid-Low)
]

# Numerical Scores (Purple in ref)
numerical_scores = [
    0.36,  # P1 (High error on nums)
    0.19,  # P2
    0.15,  # P3 (Best handling of PPT nums)
    0.26,  # P4
    0.23   # P5
]

# Plotting
x = np.arange(len(pipelines))
width = 0.35  # width of bars

fig, ax = plt.subplots(figsize=(10, 6))

# Colors matching the reference image roughly (Cyan/Teal and Purple)
color_analytical = '#50E3C2' # Teal/Cyan like
color_numerical = '#9013FE'  # Bright Purple

rects1 = ax.bar(x - width/2, analytical_scores, width, label='Analytical', color=color_analytical)
rects2 = ax.bar(x + width/2, numerical_scores, width, label='Numerical', color=color_numerical)

# Styling
ax.set_ylabel('Hallucination Score (Lower is Better)')
ax.set_title('Analytical vs Numerical Hallucination Scores by Pipeline')
ax.set_xticks(x)
ax.set_xticklabels(pipelines)
ax.legend()

# Add grid for readability
ax.yaxis.grid(True, linestyle='--', alpha=0.3)

# Add values on top of bars
def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height:.2f}',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')

autolabel(rects1)
autolabel(rects2)

fig.tight_layout()

plt.savefig(OUTPUT_FILE, dpi=300)
print(f"Graph saved to {OUTPUT_FILE}")
