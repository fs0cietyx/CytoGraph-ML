<div align="center">
  
# 🧬 CytoGraph-ML
### A Clinical-Grade AI Framework for Precision Oncology & Transcriptomics

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.3+-orange.svg)](https://scikit-learn.org/)
[![GEO Data](https://img.shields.io/badge/Data-NCBI__GEO-green.svg)](https://www.ncbi.nlm.nih.gov/geo/)
[![Status: Peer Review](https://img.shields.io/badge/Status-Under_Peer_Review-purple.svg)]()

<br>
  <p align="center">
    <b>Bridging the gap between Machine Learning, Bioinformatics, and Real-World Clinical Diagnostics.</b>
  </p>
</div>

---

## 📖 Executive Summary (The "No-BS" Primer)

Every cell in the human body contains roughly 20,000 genes. You can think of these genes as **"volume dials"** on an audio mixer. In healthy cells, the dials are perfectly balanced. In cancer cells, these dials get scrambled—some are turned up dangerously high (promoting rapid cell division), while others are muted (turning off the body's natural defenses). 

**The Goal:** Use Machine Learning to look at these 20,000 volume dials and instantly diagnose whether a patient has a tumor.

**The Reality of Clinical AI:** Most AI models in genomics suffer from **"Lab Bias" (Batch Effects)**. If an AI is trained on patients from a hospital in New York using one brand of sequencing equipment, it often fails catastrophically when analyzing a patient from a hospital in London using different equipment. The AI accidentally learns the "noise" of the hospital's machines rather than the actual biology of the disease. 

**Our Solution (CytoGraph-ML):** We built a computational pipeline that acts as a universal translator. It mathematically strips away the laboratory noise, filters out generic biological "smoke" (like inflammation), and forces the AI to lock onto the genuine oncogenic "fire." No black boxes. No false confidence.

---

## 🔬 Key Biological Discoveries

CytoGraph-ML doesn't just output a probability score; it identifies the precise genetic mechanisms driving the tumor. By querying the **NCBI Gene Expression Omnibus (GEO)** and mapping mathematical signals to real-world biology via the `MyGene.info` API, our pipeline isolated key oncogenic drivers in lung cancer:

<p align="center">
  <img src="./feature_importance.png" alt="Feature Importance Plot" width="700"/>
</p>

*   **TCF21 (Transcription Factor 21):** Identified by our model as a primary signal. In medical literature, TCF21 is a known tumor suppressor that is frequently silenced in Lung Adenocarcinoma.
*   **CRYAB (Crystallin Alpha B):** Flagged for its role in stress response and tumor progression.
*   **SLIT3:** A critical regulator of cell migration and angiogenesis (blood vessel formation) around the tumor.

By providing these readable outputs, CytoGraph-ML transitions from a simple calculator to a **Clinical Decision Support System**, allowing oncologists to design targeted therapies based on the specific genes driving an individual patient's tumor.

---

## 🛡️ The "Acid Test": Proving Clinical Safety

In machine learning, it is incredibly easy to achieve 100% accuracy if you train and test on patients from the exact same laboratory. To prove CytoGraph-ML is clinically safe, we subjected it to the "Acid Test" (Cross-Study Validation).

1. **Training:** The AI was trained entirely on patients from **Study A (GSE10072)** (Platform GPL96).
2. **Testing:** The AI was evaluated on an entirely unseen group of patients from **Study B (GSE19804)**, sequenced on completely different hardware (Platform GPL570).

### The Results

| Metric | In-Study Performance (Same Lab) | Cross-Study Performance (Different Lab) |
| :--- | :--- | :--- |
| **Sensitivity (Recall)** | **100%** | **100%** |
| **Accuracy** | 100% | 85.83% |

**What does this mean?** Even when confronted with completely foreign laboratory equipment, **the model did not miss a single cancer patient** (100% Sensitivity/Recall). The drop in accuracy (85.83%) indicates the model became highly conservative—preferring to over-flag suspicious cells rather than risk missing a lethal tumor. This is the exact "paranoid" behavior desired in clinical triage diagnostics.

---

## 💻 Technical Architecture: Every Nook and Corner

For data scientists and bioinformaticians, CytoGraph-ML offers a rigorously designed, "Zero-Leakage" architecture. We abandoned standard linear pipelines in favor of a mathematically hardened approach.

```mermaid
graph TD
    A[Raw RNA-Seq / Microarray Data] -->|GEOparse Ingestion| B(Quantile Normalization)
    B -->|Cross-Lab Alignment| C{Zero-Leakage Pipeline}
    
    subgraph "Bio-Statistical Core (Strictly Fit on Train Folds)"
    C --> D[Median Imputation]
    D --> E[Robust Scaling]
    E --> F[Biological Proxy Blacklist]
    F --> G[Mutual Information Selection]
    G --> H[Random Forest Ensemble]
    end
    
    H --> I[TreeSHAP Interpreter]
    I --> J((Clinical Output & Pathway Maps))
    
    classDef core fill:#f9f9f9,stroke:#333,stroke-width:2px;
    class C,D,E,F,G,H core;
```

### 1. Group-Blind Cross Validation (`GroupShuffleSplit`)
Standard cross-validation leaks data when a patient provides both tumor and normal samples (the "Twin Study" leak). If the normal sample is in training and the tumor is in testing, the model cheats by recognizing the patient's unique genetic fingerprint. Our pipeline utilizes `GroupShuffleSplit` grouping by Patient ID, ensuring that a patient's entire profile is strictly isolated to either the training or testing set.

### 2. Information-Theoretic Feature Selection
RNA-Seq read counts are over-dispersed and violate the Gaussian assumptions of standard linear models (like ANOVA). We implement **Mutual Information (MI)** selection to capture non-linear, complex epistasis between genes and tumor status:

$$ I(X;Y) = \sum_{y \in Y} \sum_{x \in X} p(x,y) \log \left( \frac{p(x,y)}{p(x)p(y)} \right) $$

### 3. Cross-Study Quantile Normalization
To survive the Acid Test, we implemented rank-based Quantile Normalization. This forces the statistical distribution of Study B to match Study A, mathematically eliminating the baseline shift caused by different laboratory machinery:

$$ X_{norm} = F_{target}^{-1}(F_{source}(X)) $$

### 4. Biological Proxy Filtering
To prevent the model from learning "Smoke" instead of "Fire", the pipeline actively filters out non-specific stromal and endothelial markers (e.g., general blood vessel markers like *VWF*, or inflammation markers like *IL6*). This forces the ensemble to lock onto genuine, causal oncogenes rather than generic tissue damage.

---

## 🚀 Brutally Rigorous Reproducibility

This repository is designed for instant reproducibility. You can run the exact Acid Tests discussed in the research paper with a few simple commands.

### Prerequisites
* Python 3.11+
* Git

### Installation
```bash
# Clone the repository
git clone https://github.com/fs0cietyx/CytoGraph-ML.git
cd CytoGraph-ML

# Create a virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install the required scientific libraries
pip install -r requirements.txt
```

### The Test Suite
To verify the integrity of the data pipelines and API mapping:
```bash
PYTHONPATH=src pytest tests/ -v
```

### Running the Validations

**1. The Cross-Study Acid Test (Lung Cancer)**
Downloads GSE10072 and GSE19804, normalizes them, and proves cross-platform generalizability.
```bash
PYTHONPATH=src python3 src/acid_test.py
```

**2. The In-Study Hardened Validation (Lung Cancer)**
Executes Group-Blind 5-Fold Cross Validation on GSE10072.
```bash
PYTHONPATH=src python3 src/validate_real_data.py
```

**3. Cross-Tissue Generalization (Colorectal Cancer)**
Validates that the architecture functions across different cancer origins (GSE21510).
```bash
PYTHONPATH=src python3 src/validate_colorectal.py
```

**4. Permutation Data Leakage Audit**
Runs the model on completely shuffled labels to prove the accuracy is not a statistical artifact.
```bash
PYTHONPATH=src python3 src/audit_model.py
```

All results, including mapped pathways and classification reports, are saved automatically in the `results/` folder.

---

## 📁 Repository Structure

Every script in this repository has a dedicated scientific purpose:

```text
CytoGraph-ML/
├── src/
│   ├── core/
│   │   ├── config.py           # Hyperparameters and threshold settings
│   │   ├── geo_engine.py       # NCBI GEO Fetcher & Quantile Normalizer
│   │   ├── gdc_engine.py       # GDC API fetching tools for raw count data
│   │   ├── trainer.py          # Scikit-learn Zero-Leakage Pipeline
│   │   └── bio_mapper.py       # MyGene.info Pathway mapping API integration
│   ├── api/                    
│   │   └── main.py             # FastAPI Inference Endpoint for deployment
│   ├── acid_test.py            # Main cross-study validation script
│   ├── validate_real_data.py   # In-study evaluation script
│   ├── validate_colorectal.py  # Cross-tissue validation script
│   ├── audit_model.py          # Label permutation test to ensure no data leakage
│   └── robustness_test.py      # Gaussian noise injection to test model stability
├── data/
│   └── raw/                    # Auto-downloaded NCBI matrix files (Git-ignored)
├── models/                     # Auto-saved .joblib pipeline artifacts
├── results/                    # Generated markdown clinical reports
├── MANUSCRIPT.md               # LaTeX source and text for the research preprint
└── docker-compose.yml          # Container configuration for production API
```

---

## 🚧 Limitations & Future Scope

In the spirit of rigorous science, we acknowledge the following limitations:
1. **Microenvironment Contamination:** Bulk transcriptomics captures an average signal of tumor, stromal, and immune cells. Future iterations will integrate deconvolution algorithms (e.g., CIBERSORT) to digitally separate the tumor microenvironment.
2. **Prognostic Scope:** The current pipeline focuses on *Diagnostic Classification* (Tumor vs. Normal). Future expansions will target *Prognostic Regression* (e.g., predicting patient survival time or therapeutic sensitivity).

---

## ⚠️ Clinical Disclaimer

**FOR RESEARCH USE ONLY (RUO). NOT FOR USE IN DIAGNOSTIC PROCEDURES.**

CytoGraph-ML is a scientific research prototype designed to study the generalizability of machine learning models under batch-effect platform shifts. It has not been analytically validated, clinically validated, or cleared/approved by the US Food and Drug Administration (FDA) or any other regulatory body. 

Under no circumstances should this software, its API endpoints, or its predictions be used to diagnose, treat, cure, or monitor any patient. The author(s) assume no liability or responsibility for clinical outcomes or diagnostic routing decisions made using this code.

---

## 🤝 Citation & Contact

This repository serves as the official codebase for the manuscript: 
**"CytoGraph-ML: A Robust Machine Learning Framework for Pan-Cancer Transcriptomic Classification and Explainable Genomic Driver Identification."**

If you use this pipeline or its methodology in your research, please cite our Zenodo upload and GitHub repository.

**Author:** Mainak Biswas  
**Institution:** Kalinga Institute of Industrial Technology (KIIT), India  
**Contact:** 24155779@kiit.ac.in  

<p align="center">
  <i>"Transforming data into diagnosis, safely and transparently."</i>
</p>
