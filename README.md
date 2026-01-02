# 🤖 Financial Document Analysis Pipeline (Multi-GPU)

A high-performance, modular RAG system for analyzing complex financial documents (Earnings Transcripts, Analyst Notes, and Investor Presentations). This project leverages a **Multi-GPU Architecture** to perform text and visual extraction in parallel.

## 📂 Repository Structure

```
├── data/                       # Input documents
│   ├── ppt.pdf                 # Investor Presentation
│   ├── transcript.pdf          # Earnings Call Transcript
│   └── notes.pdf               # Analyst Summary Notes
│
├── pipelines/                  # Operational Pipeline Scripts
│   ├── advanced/               # 🚀 NEW: Multi-GPU, Vision-Integrated Pipelines
│   │   ├── pipeline2_consolidation_ppt.py
│   │   ├── pipeline3_parallel_ppt.py
│   │   └── pipeline5_consolidation_ppt.py
│   │
│   └── legacy/                 # Old Versions (Text-Only or Single-Threaded)
│       ├── pipeline1_direct.py
│       ├── pipeline3_parallel.py
│       └── ...
│
├── models/                     # LLM Interfaces & Wrappers
│   └── llm_interface.py        # Connects to Ollama & Gemini
│
├── scripts/                    # Utilities & Tools
│   ├── ppt_extractor_v6.py     # Standalone Visual Extractor Module
│   ├── check_models.py
│   └── ...
│
├── config.py                   # Global Configuration (Paths, Models)
└── multimodal_utils.py         # helper for Vision processing
```

---

## 🏗️ Architecture & Pipelines

### 1. The Multi-GPU Engine
This system is designed for servers with **2x NVIDIA GPUs** (e.g., Tesla V100/A100).
- **GPU 0 (`cuda:0`)**: Hosts the **Vision Stack** (Qwen2-VL-7B-Instruct-4bit + PaddleOCR-v4). It handles complex slide analysis (charts, tables, growth rates).
- **GPU 1 (`cuda:1`)**: Hosts the **Text Inference Engine** (Ollama running Mixtral 8x7B or Llama-3-70B). It handles reasoning, consolidation, and QA.

### 2. Available Pipelines (`pipelines/advanced/`)

#### 🔹 [Pipeline 2] Visual Consolidation
*Best for: Deep synthesis of all data sources.*
1.  **Extracts** Visual Data from PPT (GPU 0).
2.  **Merges** with Transcript + Notes.
3.  **Consolidates** into a single master document.
4.  **Extracts** financial facts from the master document.

#### ⚡ [Pipeline 3] Parallel Extraction (Recommended)
*Best for: Maximum Speed & Throughput.*
1.  **Parallel Execution**:
    *   **Stream A**: Transcript Extraction (Text LLM / GPU 1)
    *   **Stream B**: Notes Extraction (Text LLM / GPU 1)
    *   **Stream C**: Visual Extraction (Vision Model / GPU 0)
    *   *All 3 run simultaneously.*
2.  **Reconciliation**: A "Judge" LLM compares the Visual Evidence against the Textual Claims to flag hallucinations.

#### 💎 [Pipeline 5] Iterative Critique
*Best for: Highest Accuracy & Fact-Checking.*
1.  Consolidation & Initial Extraction.
2.  **Critique Loop**: The LLM reviews its own extraction against the **Raw JSON** data extracted from the slides.
3.  **Refinement**: Automatically corrects numbers or claims found to be inconsistent with the visual evidence.

---

## 🚀 Detailed Execution Guide

### Prerequisite: Double-Check Hardware
Ensure `nvidia-smi` shows both GPUs and that they are idle.
```bash
nvidia-smi
# Should list:
# GPU 0: Tesla V100 ...
# GPU 1: Tesla V100 ...
```

### Step 1: Configure the Text Engine (Ollama)
**Goal**: Force Ollama to run ONLY on **GPU 1** so it leaves GPU 0 free for the Vision models.
**Action**: Launch the server with `CUDA_VISIBLE_DEVICES=1`.

```bash
# Terminal 1
export CUDA_VISIBLE_DEVICES=1
ollama serve
```
*Tip: If Ollama is already running as a service, stop it first (`sudo systemctl stop ollama`).*

### Step 2: Configure the Vision Engine (Python Pipeline)
**Goal**: Run the pipeline script, which uses Qwen2-VL and PaddleOCR.
**Action**: The scripts are hardcoded to use `cuda:0` (GPU 0) internally, so you generally **do not** need to set env vars for them.
*   However, ensure you run this in a **separate terminal** where `CUDA_VISIBLE_DEVICES` is NOT set (or set to `0,1`).

```bash
# Terminal 2
cd /path/to/llm_pipeline
python3 pipelines/advanced/pipeline3_parallel_ppt.py
```

### Step 3: Verification
Open a third terminal and run `watch -n 1 nvidia-smi`.
*   You should see **Python** (Vision) processes on **GPU 0**.
*   You should see **Ollama_runner** processes on **GPU 1**.
*   If you see both on the same GPU, stop immediately to avoid OOM.

---

## 🧩 Models Used

| Component | Model Name | Source | Usage |
| :--- | :--- | :--- | :--- |
| **Vision LLM** | Qwen/Qwen2-VL-7B-Instruct | HuggingFace | Chart/Graph Interpretation |
| **OCR Engine** | PaddleOCR v2.7 (PP-OCRv4) | PaddlePaddle | Dense Text/Table Extraction |
| **Text Reasoning** | Mixtral 8x7B-Instruct-v0.1 | Ollama | Extraction, Consolidation, Logic |
| **Judge (Optional)** | Gemini 1.5 Pro / Flash | Google Cloud | Hallucination Scoring (Ground Truth) |

---

## 🔧 Maintenance

*   **Helper Scripts**: Check `scripts/` for tools to debug models or patch notebooks.
*   **Logs**: Execution logs are printed to stdout and contain detailed timings for each step (OCR, VLM Pass 1, VLM Pass 2).
