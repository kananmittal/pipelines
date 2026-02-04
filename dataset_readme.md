# 📊 Multi-Modal Financial Analysis Dataset

> * **Associated Research**: *Mitigating Hallucination in Multi-Modal Information Systems: A Comparative Analysis of Modular LLM Architectures*

## 📖 Overview

This dataset serves as the **Master Registry** for the multi-modal financial artifacts used in our study on modular LLM architectures. It provides a structured foundation for evaluating how Large Language Models (LLMs) reason over heterogeneous financial documents—specifically **Earnings Call Transcripts**, **Audio Recordings**, **Analyst Notes**, and **Investor Presentations**.

The dataset was curated to reflect real-world information environments where data is often diverse, unstructured, and occasionally incomplete. It underpins the experimental results presented in our research paper, providing the ground truth for availability and cross-sector analysis.

## 🗂️ Dataset Statistics

* **Total Entries**: 567 Company-Quarter pairs
* **Total Artifacts**: ~2,268 (Transcripts, Audio, Notes, Visuals)
* **Coverage**: 13 Economic Sectors (Communication, Finance, Healthcare, Tech, etc.)
* **Completeness**: Intentionally variable to simulate realistic "missing data" scenarios.

## 📂 File Structure

The primary registry file is `Quarterly Sales Data - Sheet1.csv`.

| Column            | Description                        | Role in Pipeline                                       |
| :---------------- | :--------------------------------- | :----------------------------------------------------- |
| `Company`       | Target entity name                 | Grouping Key                                           |
| `Quarter`       | Reporting period (e.g., Q4-2024)   | Temporal Analysis                                      |
| `Sector`        | Industry classification            | Cross-Sector Robustness Checks                         |
| `Transcript`    | `TRUE` if text transcript exists | **Primary Input** (Stream A)                     |
| `Notes`         | `TRUE` if analyst notes exist    | **Supplementary Input** (Stream B)               |
| `PPT`           | `TRUE` if visual deck exists     | **Required for Multimodal Pipelines** (Stream C) |
| `Concall Audio` | `TRUE` if raw audio exists       | Validation / Audio Processing                          |

## 🧪 Research Application

In the context of the paper **"Mitigating Hallucination in Multi-Modal Information Systems"**, this registry is critical for:

1. **Benchmarking**: Defining the subset of "complete" data points used to compare single-modality vs. multi-modality pipelines.
2. **A/B Testing**: Enabling the segmentation of results.
3. **Realism**: Enforcing a constraint where the model must handle cases with missing `Notes` or `Audio`, mirroring deployment conditions.

## 📝 Citation

If you use this dataset or the associated pipeline code in your work, please cite the original research on "Mitigating Hallucination in Multi-Modal Information Systems: A Comparative Analysis of Modular LLM Architectures"
