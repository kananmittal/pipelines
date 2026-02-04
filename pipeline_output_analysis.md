# Analysis of Pipeline Interim Outputs

This report analyzes the nature of the **interim outputs** (the context fed to the final LLM) across the five modular architectures. It specifically contrasts the "Reconciled Document" of Pipeline 3 with the outputs of other pipelines to explain performance differences.

## 1. The Nature of the "Reconciled Document" (Pipeline 3)

In **Pipeline 3 (Parallel Extraction)**, the interim output is **NOT** a flat block of text. Instead, it is a **Structured Knowledge Object** that preserves the provenance of information from different modalities.

### What it looks like (Structure)

The reconciled document is constructed by merging parallel streams:

```text
=== RECONCILED CONTEXT ===

[STREAM A: TEXTUAL EVIDENCE (TRANSCRIPT)]
- The CEO mentioned volume growth declined by 1.8% due to weak demand.
- New product launches: 300+ in 5 years.

[STREAM C: VISUAL EVIDENCE (SLIDES)]
=== SLIDE 12 ===
[Factual Summary]: Chart shows 14% revenue contribution from innovation.
[Raw Data]: {"metric": "Revenue", "segment": "Innovation", "value": "14%"}

[STREAM B: ANALYST NOTES]
- Analysts flagged margin pressure in the decorative segment.

[RECONCILIATION NOTES]
- Confirmed "14% growth" aligns between Transcript and Slide 12.
- Conflict: Transcript says "1.8% decline" vs Notes "flat growth" -> Prioritized Transcript.
```

### Key Characteristics

1. **Segmented & Attributed**: Unlike P1 or P2, P3 keeps visual data validly separate from textual data until the final answer generation. This prevents "context soup" where numbers get mixed up.
2. **Rich Fidelity**: It retains raw JSON data from slides alongside high-level summaries.
3. **Conflict Resolution**: The reconciliation step explicitly flags if the Slides say X and the Transcript says Y, allowing the LLM to judge truthfulness.

---

## 2. Comparative Analysis of Interim Outputs

* PipelineNature of Interim OutputLengthComplexityKey Differences**P1 (Single)****Raw Text Dump**Very LongHigh (Noisy)Contains headers, footers, filler words ("Um", "Ah"). No structure. Hard for LLM to find needles in haystacks.**P2 (Consolidate)****Bulleted Fact List**ShortLow (Clean)Flattens everything into a list of ~20-50 bullets.**Lossy**: Nuance and source context (e.g., "was this in Q3 or Q4?") are often lost during summarization.**P3 (Parallel)****Structured Reconciled Object**MediumHigh (Rich)**Best Balance**. Preserves structure (Slide vs Text). Longer than P2 but organized. Contains "Conflict Flags" for better reasoning.**P4 (Iterative)****Refined Text**MediumMediumSimilar to P1 but with "Self-Correction" notes appended (e.g., "Correction: The value is 1.8%, not 18%").**P5 (Advanced)****Critiqued Narrative**LongHigh (Polished)A multi-part document containing: (1) Initial Summary, (2) Critique/Reflection, (3) Final Polished Narrative. Extremely detailed but computationally heavy.

---

## 3. Systematic Differences & Performance Impact

### Why is P3 Best? (The "Structure" Hypothesis)

* The systematic difference in P3 is **Structural Preservation**.

* **P1/P4** overwhelm the context window with noise.
* **P2** over-compresses, potentially removing the *evidence* required to verify a hallucination.
* **P3** provides the **Evidence + Synthesis**. By keeping the "Raw Data" from the PPT side-by-side with the "Text Summary", the final LLM works like a human cross-referencing sources, leading to the lowest hallucination score (0.145).

### Why P5 is Slower but not always Better

P5's document is "meta-cognitive"—it contains the model's own thinking about the data as `intermediate_steps`. While this reduces some logic errors, the sheer volume creates latency (~86s) and sometimes leads to **over-correction** (hallucinating errors that don't exist), explaining its higher score (0.210) compared to P3.
