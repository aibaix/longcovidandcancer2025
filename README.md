# Long COVID and RCC Progression – Repository

This repository accompanies the study:
**“Dynamic Network Analysis and Machine Learning Reveal the Impact of Long COVID on RCC Progression.”**

## Status

The complete code, preprocessing scripts, and model implementations **will be released upon acceptance** of the manuscript.

## Overview

This project integrates:

* Dynamic disease network analysis using SEER–Medicare data
* A literature‑derived biomedical knowledge graph
* Deep learning–based dynamic prediction models (LSTM)
* External validation using the TriNetX network
* Molecular docking simulations validating key pathways (TGF‑β / NF‑κB, IL‑6 / JAK / STAT)

The goal is to elucidate how Long COVID influences renal cell carcinoma (RCC) progression biologically, clinically, and computationally.

## Repository Structure (planned)

```
longcovid_rcc/
│── data/                     # Processed datasets (released post‑acceptance)
│── src/
│    ├── preprocessing/       # SEER-Medicare preprocessing pipeline
│    ├── network_analysis/    # Disease network construction + metrics
│    ├── knowledge_graph/     # NLP extraction, KG construction
│    ├── lstm_model/          # Dynamic prediction models
│    ├── docking/             # Molecular docking scripts
│    └── utils/               # Shared helper functions
│── notebooks/                # Reproducible Jupyter notebooks
│── figures/                  # Figures generated from the study
│── requirements.txt
│── README.md
```

## Main Components

### 1. **Disease Network Analysis**

* Builds temporal disease networks using RR, φ‑correlations
* Louvain clustering and permutation testing
* Network dynamics across acute → recovery → long-term phases

### 2. **Knowledge Graph Construction**

* Literature-mined biological entities (genes, proteins, pathways)
* Entity linking via UMLS, GO, HPO
* Relation extraction using LLaMA‑3.3 (NER F1 = 0.97)

### 3. **Dynamic Prediction Model (LSTM)**

* 30‑day sliding windows
* 64‑unit LSTM → 32‑unit FC → sigmoid output
* Performance: AUC = 0.83, PR‑AUC = 0.79, Brier = 0.14, ECE = 0.05

### 4. **Molecular Docking**

* AutoDock Vina simulations for TGF‑β1 ↔ NF‑κB p65
* Structural validation using PyMOL + STRING

### 5. **External Validation with TriNetX**

* PSM-matched 1:2 cohorts
* Concordance: Cohen’s κ = 0.82

## Reproducibility

Upon release, the repository will include:

* Fully automated preprocessing pipelines
* Rule‑based progression proxy definitions
* Version‑locked dependency lists
* End‑to‑end notebooks replicating all figures & results

## Citation

```
(Will be added after acceptance.)
```

## Contact

For questions or collaboration inquiries:

Code will be made available once the manuscript is officially accepted.
