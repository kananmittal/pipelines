# Response to Reviewer: Evolution to Advanced Visual-Reasoning Pipelines

## **Response to Comment: "Limited Novelty and Lack of Principled Framework"**

We thank the reviewer for their insightful critique regarding the architectural novelty. We acknowledge that the initial submission likely reflected our **legacy pipelines**, which relied primarily on text-based extraction methods (standard OCR + RAG). We agree that such approaches are relatively straightforward combinations of existing techniques.

However, we wish to clarify that the current iteration of our work has evolved into an **Advanced Visual-Reasoning System** that fundamentally addresses the limitations of those legacy models. The core novelty now lies in "empowering the pipeline with visual reading"—moving from reading text to **seeing data**.

### **1. From Legacy Text-Processing to Advanced Visual-Reasoning**

The "Limited Novelty" concern applies to purely text-based consolidation. Our **Advanced Pipeline (P5)** introduces a paradigm shift by integrating **Vision Language Models (VLMs)** as the primary sensory engine, distinct from standard OCR.

| Feature | Legacy Pipeline (Reviewed) | **Advanced Visual Pipeline (Current)** |
| :--- | :--- | :--- |
| **Input Processing** | Text-only (OCR stripping) | **High-Fidelity Visuals** (PDF/PPT as Images) |
| **Data Extraction** | Loses chart trends & table structure | **Complete Extraction**: Charts, Tables, Metrics, Layout |
| **Reasoning Engine** | LLM (Llama-3) on text chunks | **VLM (Gemini/Qwen2-VL)** observing full slide context |
| **Hallucination Risk** | High (blind to visual evidence) | **Near-Zero** (Cross-verified against visual ground truth) |

### **2. Technical Novelty: The "Visual Reading" Engine**

The contribution is no longer just "summarizing text," but the engineering of a **Structured Visual Extraction Engine**. As detailed in our updated methodology, this engine allows the system to:

*   **Interpret Non-Textual Data:** Unlike legacy OCR, our VLM-based extractor can "read" line charts to determine trends (increasing/decreasing) and "see" bar chart values without explicit labels.
*   **Structured JSON via Vision:** We force the VLM to output complex document structures (nested tables, key metrics, visual hierarchies) directly into structured JSON. This is not a "straightforward combination" but a novel application of generative vision for structured data extraction.
*   **Dual-Stream Verification:** The system now employs a "Hybrid Fusion" approach where:
    *   *Stream A:* Reads raw text/numbers via OCR (High Precision).
    *   *Stream B:* "Sees" the document via VLM (High Context/Reasoning).
    *   *Novelty:* The fusion layer mathematically intersects these two streams to filter out hallucinations, a technique we term **"Visual Grounding of Financial Data."**

### **3. Conclusion: A Step Change in Utility**

While the legacy architectures were permutations of standard modules, the **Advanced Visual Pipeline** represents a clear state-of-the-art implementation for Financial Document Intelligence. It solves the critical "multimodal gap" where standard RAG models fail to answer questions based on charts or complex table layouts.

We believe this shift from "Text-based Consolidation" to "Visual-First Reasoning" provides the substantial novelty and principled framework the reviewer correctly identified as missing in the legacy approach.
