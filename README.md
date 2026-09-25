# Reducing Hallucinations in Enterprise AI Using RAG and Explainable AI

**Master's Thesis Implementation**  
**Author:** Priyaneiyah Selvakumar  
**Programme:** M.Sc. Data Science, AI and Digital Business  
**Institution:** GISMA University of Applied Sciences

## Overview

This repository contains the experimental implementation for the master's thesis investigating hallucination in Retrieval-Augmented Generation (RAG) for enterprise technical-support question answering.

The experiments use the TechQA dataset and examine how different stages of a RAG pipeline contribute to evidence retrieval, context construction and answer reliability.

The implementation is organised around three experimentally evaluated research questions: RQ1, RQ2 and RQ3.

## Research Questions

### RQ1 — RAG Component Effects

RQ1 investigates how changes to major RAG components affect evidence retrieval and generation performance.

The experiments examine:

- chunking methods
- retrieval strategies
- reranking
- generation configurations

Retrieval performance is evaluated using measures including nDCG@5, Recall@20, MRR and gold-span coverage.

### RQ2 — Stage Attribution

RQ2 investigates where failures occur in the RAG pipeline.

The stage-attribution framework separates failures into:

- **S1 — Segmentation failure**
- **S2 — Retrieval failure**
- **S3 — Context-selection / evidence-coverage failure**
- **S4 — Generation failure**
- **R — Reachable evidence**

The segmentation ceiling is calculated before stage attribution so that failures are assigned using an ordered first-failure procedure.

### RQ3 — Certainty and Calibration

RQ3 investigates whether deployment-time retrieval signals can provide useful evidence about whether sufficient supporting information has reached the generation stage.

The experiments evaluate discrimination and calibration using measures including:

- AUROC
- Brier score
- Expected Calibration Error (ECE)

A citation-based answer-control rule is also evaluated separately.

## Dataset

The experiments use **TechQA**, an enterprise technical-support question-answering dataset.

The experimental corpus contains:

- 910 labelled questions
- 610 answerable questions
- 300 unanswerable questions
- 28,482 technical documents

Large dataset files, model weights, generated caches and intermediate experimental artefacts are not stored in this repository.

## Repository Structure

```text
enterprise-rag-hallucination-thesis/
│
├── notebooks/
│   ├── RQ1_00_Build_Once.ipynb
│   ├── RQ1_A_Chunking_Method.ipynb
│   ├── RQ1_B_Retrieval_Strategy.ipynb
│   ├── RQ1_C_Reranking.ipynb
│   ├── RQ1_D_Generation_Profile.ipynb
│   ├── RQ1_E_Answer_RQ1.ipynb
│   ├── RQ2_0_Segmentation_Ceiling.ipynb
│   ├── RQ2_Stage_Attribution.ipynb
│   └── RQ3_Certainty_Calibration.ipynb
│
├── src/
│   ├── rq1_core.py
│   ├── rq2_core.py
│   └── rq3_core.py
│
├── requirements.txt
├── .gitignore
└── README.md
