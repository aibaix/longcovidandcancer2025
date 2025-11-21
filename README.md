# Long COVID and RCC Progression Analysis Codebase

## Overview

This repository contains the complete Python codebase for replicating the analysis described in the paper *"Dynamic Network Analysis and Machine Learning Reveal the Impact of Long COVID on RCC Progression"*. The code implements data preprocessing, disease network construction, knowledge graph building, LSTM-based predictive modeling, molecular docking simulations, and statistical analyses using SEER-Medicare and TriNetX data.

The framework integrates dynamic disease networks, literature-derived knowledge graphs, and deep learning models to study the oncological impact of Long COVID on renal cell carcinoma (RCC) patients. It provides tools for risk stratification, pathway identification, and intervention simulation.

Key features:
- Data handling and propensity score matching for cohort balancing.
- Construction and analysis of temporal disease networks.
- Extraction of biomedical entities from literature to build knowledge graphs.
- LSTM model for predicting RCC progression with time-series data.
- Molecular docking for validating key pathways (e.g., TGF-β/NF-κB).
- Comprehensive statistical evaluations, including survival analysis and mixed-effects modeling.

This codebase is designed for reproducibility, with modular scripts that can be run independently or via the main execution script.

## Prerequisites

### Python Version
- Python 3.8 or higher (tested on 3.8 and 3.12).

### Dependencies
Install the required packages using pip. A `requirements.txt` file is provided for convenience.

```bash
pip install -r requirements.txt
```

Key dependencies include:
- `pandas` (>=1.3.0): Data manipulation and analysis.
- `numpy` (>=1.20.0): Numerical computations.
- `scikit-learn` (>=1.0.0): Preprocessing, imputation, and matching.
- `networkx` (>=2.6.0): Network construction and analysis.
- `matplotlib` (>=3.4.0) and `seaborn` (>=0.11.0): Visualization.
- `torch` (>=1.10.0): LSTM model implementation.
- `transformers` (>=4.10.0): NLP for entity extraction (using BioBERT).
- `lifelines` (>=0.26.0): Survival analysis.
- `statsmodels` (>=0.13.0): Mixed-effects models and statistical tests.
- `shap` (>=0.40.0): Model interpretability.
- `rdkit` (>=2022.3.0): Molecular preparation for docking.
- `biopython` (>=1.79): PDB parsing.
- `requests` and `beautifulsoup4`: Literature fetching.
- Other: `scipy`, `sklearn.utils` for resampling, etc.

Note: For molecular docking, AutoDock Vina must be installed separately and available in your system's PATH. Download from [here](https://vina.scripps.edu/downloads/). Open Babel is also required for file conversions (install via `conda install -c conda-forge openbabel` or similar).

### Data Requirements
- SEER-Medicare dataset: Place in `seer_medicare.csv` (not included due to privacy; obtain from official sources).
- Ensure the CSV has columns like `age`, `long_covid`, `progression`, `patient_id`, `time`, etc., as referenced in the code.
- For TriNetX validation: Access via API (not implemented here; use aggregate queries as described in the paper).

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-repo/long-covid-rcc-analysis.git
   cd long-covid-rcc-analysis
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Install AutoDock Vina and Open Babel (see above).

4. Prepare data: Place `seer_medicare.csv` in the root directory.

## File Structure

- `data_preprocessing.py`: Functions for loading data, handling missing values, propensity score matching, standardization, and time-series preparation.
- `disease_network.py`: Builds disease networks, computes centralities, detects communities, bootstraps for stability, and performs permutation tests.
- `knowledge_graph.py`: Fetches literature, extracts entities using BioBERT, builds and integrates knowledge graphs.
- `lstm_model.py`: Defines LSTM architecture, datasets, training, evaluation, cross-validation, interpretability (SHAP), and plotting functions.
- `molecular_docking.py`: Downloads PDB files, prepares ligands/receptors, runs AutoDock Vina, analyzes results, performs sensitivity analysis, and validates interactions.
- `statistical_analysis.py`: Group comparisons, Kaplan-Meier survival, Cox PH models, mixed-effects modeling, regression evaluation, landmark analysis, and counterfactual simulations.
- `main.py`: Orchestrates the full pipeline, from data loading to result output (saves visualizations and JSON results).
- `requirements.txt`: List of Python dependencies.
- `README.md`: This file.

## Usage

### Running the Full Pipeline
Execute the main script to run the entire analysis:

```bash
python main.py
```

This will:
- Preprocess data and match cohorts.
- Build and analyze disease networks.
- Construct and integrate knowledge graphs.
- Train and evaluate the LSTM model.
- Perform molecular docking simulations.
- Run statistical analyses.
- Output results to `results.json` and save visualizations (e.g., `disease_network.png`, `roc.png`).

Expected runtime: ~30-60 minutes on a standard machine (GPU recommended for LSTM training).

### Running Individual Modules
You can import and run specific functions for modular use. For example:

```python
# Example: Build and visualize disease network
import pandas as pd
from disease_network import build_disease_network, visualize_network

df = pd.read_csv('seer_medicare.csv')
categories = ['rcc', 'other_renal', 'metabolic', 'cardiovascular', 'inflammatory', 'systemic']
G = build_disease_network(df, categories)
visualize_network(G, 'network.png')
```

### Configuration
- Adjust hyperparameters in scripts (e.g., LSTM hidden_size=64, docking exhaustiveness=8).
- For literature fetching, modify queries in `knowledge_graph.py`.
- GPU usage: Set `torch.device('cuda')` in `lstm_model.py` if available.

## Output

- **Visualizations**: PNG files for networks, ROC curves, calibration plots, survival curves.
- **Metrics**: Printed during execution; saved in `results.json` (e.g., AUC, p-values, docking scores).
- **Logs**: Console output for progress and errors.

## Reproducibility Notes

- Set random seeds (e.g., `np.random.seed(42)`) for consistent results.
- Data imputation uses MICE with 20 iterations.
- Network bootstrapping: 1000 repetitions, stability >=90%.
- LSTM: 5-fold time-series CV, early stopping with patience=10.
- Docking: Grid size 40x40x40 Å, exhaustiveness=8.

For exact replication, use the same data versions (2019-2021 SEER-Medicare).

## Limitations and Extensions

- Assumes binary disease indicators (0/1) in data.
- NLP entity extraction uses BioBERT; fine-tune for better accuracy.
- No TriNetX API integration; manual aggregate queries needed.
- Extend to other cancers by modifying categories and queries.

## Citation

If using this code, cite the original paper:

```
Waiting...
```

## License

MIT License. See `LICENSE` for details.

## Contact

For issues or contributions, open a GitHub issue or pull request.
