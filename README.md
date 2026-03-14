# IndicWG 2025 — Indic Word Grouping

Part of the **BHASHA 2025 Shared Task 2**, co-located with the 1st Workshop on Benchmarks, Harmonization, Annotation, and Standardization for Human-Centric AI in Indian Languages.

---

## Task Overview

Given a plain text sentence in an Indian language, the task is to output a sequence of **semantically cohesive word groups**, marked with `__` separators.

**Example (Hindi):**

```
Input:  कुक आइलैंड्स दक्षिण प्रशांत महासागर के बीच में पोलिनेशिया में स्थित एक द्वीप देश है
Output: कुक__आइलैंड्स दक्षिण प्रशांत__महासागर__के बीच में पोलिनेशिया__में स्थित एक द्वीप देश है
```

| Split | Sentences |
|---|---|
| Train | 550 |
| Dev | 100 |
| Test | 226 |

---

## Our Approach — Team Horizon (BHASHA 2025)

> **Paper:** *Team Horizon at BHASHA Task 2: Fine-tuning Multilingual Transformers for Indic Word Grouping*
> Manav Dhamecha, Gaurav Damor, Sunil Choudhary, Pruthwik Mishra (SVNIT)
> **Paper Link:** https://aclanthology.org/2025.bhasha-1.18.pdf

We framed word grouping as a **BIO token classification** problem and fine-tuned three multilingual encoder models.

**Models evaluated:** `MuRIL`, `XLM-Roberta`, `IndicBERT v2`

**Key design choices:**
- Inverse-frequency class weighting to counter O-label dominance
- Subword-to-word label alignment using `word_ids()`
- 5K augmented Hindi sentences from a rule-based LWG finder

**Results (Exact Match Accuracy):**

| Model | Dev EM (%) | Test EM (%) |
|---|---|---|
| MuRIL | 46.58 | **58.18 (1st)** |
| XLM-R | 39.00 | 53.36 |
| IndicBERT v2 | 35.40 | 52.73 |

MuRIL performed best, likely due to its cased Indic-specific vocabulary preserving morpheme and script cues.

---

---

## Hugging Face Models

The fine-tuned models are available on Hugging Face:

- Google Muril → https://huggingface.co/manavdhamecha77/WG-GoogleMuril
- XLM Roberta → https://huggingface.co/manavdhamecha77/WG-XLM_Roberta
- IndicBERT → https://huggingface.co/manavdhamecha77/WG-IndicBERT

--- 


## Evaluation

**Exact Match Accuracy** — a prediction is correct only if the entire grouped output matches the gold sentence exactly.

---

## Citation

```bibtex
@inproceedings{dhamecha2025wordgrouping,
  title     = {Team Horizon at BHASHA Task 2: Fine-tuning Multilingual Transformers for Indic Word Grouping},
  author    = {Dhamecha, Manav and Damor, Gaurav and Choudhary, Sunil and Mishra, Pruthwik},
  booktitle = {Proceedings of the 1st Workshop on BHASHA 2025},
  pages     = {175--179},
  year      = {2025}
}
```