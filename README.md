# Process-Driven Credibility: A CRISP-DM Helix Framework for Robust Pima Diabetes Prediction

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Paper](https://img.shields.io/badge/PDF-Paper-blue)](paper/paper.pdf)
[![Kaggle](https://img.shields.io/badge/Kaggle-Code-20BEFF)](https://www.kaggle.com/code/yakeworld126/crisp-dm-pima)

> **核心论点**：方法学严谨性（而非算法复杂度）才是临床机器学习可信度的真正决定因素。每个公开数据集应有校准过的性能上限——超过该上限的研究自动进入方法学审计。
>
> **Core Thesis**: Methodological rigor, not algorithmic complexity, is the true determinant of clinical ML trustworthiness. Every public benchmark dataset should have a calibrated performance ceiling — studies exceeding it trigger automatic methodological audit.

---

## 📋 Overview

The Pima Indians Diabetes Dataset (PIDD) literature suffers from a severe **"high-metric, low-credibility" paradox** — models claiming >95% accuracy universally fail in real-world deployment. This project provides quantitative measurement of how data leakage inflates reported metrics (+8.6% F1 inflation), and proposes the **CRISP-DM Helix framework** as a protocol-driven solution. A **Dataset Baseline Registry** is proposed to operationalize detection: the PIDD performance ceiling (F1=0.7541) serves as an audit trigger for future work.

### Key Discoveries

| Discovery | Finding | Impact |
|:----------|:--------|:-------|
| **Recall Paradox** | Global SMOTE inflates F1 by +8.6% (0.6759→0.7338) but DECREASES Recall (0.7165→0.7080) | High-accuracy models miss MORE diabetic patients |
| **Leakage Quantification** | Quantitative measurement of leakage-induced metric inflation | Establishes a reproducible benchmark |
| **CRISP-DM Helix** | Dual-strand framework: Clinical Credibility + Engineering Execution | Standard for clinical ML reproducibility |
| **Dataset Baseline Registry** | Verified performance ceiling as automatic audit trigger | Any study exceeding F1=0.7541 on PIDD requires audit |

### True Performance (Leakage-Free)

Under strict CV-fold-isolated preprocessing, an optimized soft-voting ensemble (GBC+LDA+SVC) achieves:

| Metric | Value |
|:-------|:------|
| F1-Score | **0.7541** |
| Recall | **0.7500** |
| Precision | 0.6625 |
| Accuracy | 0.7746 |
| AUC | 0.8481 |

*Compare this to literature claims of 95–100% accuracy — all statistical artifacts from data leakage.*

---

## 📂 Repository Structure

```
crispdm-pima/
├── README.md              # This file — project overview, findings, and usage
├── LICENSE                # MIT License
├── .gitignore             # LaTeX and Python artifacts
│
├── paper/                 # Manuscript
│   ├── paper.tex          # Full LaTeX source (elsarticle, 6 pages)
│   ├── paper.pdf          # Compiled PDF (260KB, 0 errors)
│   └── references.bib     # 17 bibliographic references
│
├── figures/               # Architecture diagrams (future)
│   └── (empty — see paper for TikZ figures)
│
└── analysis/              # Analysis notebooks (future)
    └── (empty — code on Kaggle)
```

---

## 🔬 Methodology: CRISP-DM Helix Framework

The CRISP-DM Helix is a **double-helix architecture** with two intertwined strands:

```
┌─────────────────────────────────────────────────────┐
│                CRISP-DM HELIX                        │
│                                                      │
│  ┌─────────────────┐     ┌──────────────────────┐   │
│  │ Clinical         │     │ Engineering           │   │
│  │ Credibility      │◄───►│ Execution             │   │
│  │ Strand           │     │ Strand                │   │
│  ├─────────────────┤     ├──────────────────────┤   │
│  │ • Maximize Recall│     │ • CV-fold isolation   │   │
│  │ • Minimize FNs   │     │ • Within-fold fit     │   │
│  │ • Clinical trust │     │ • No global preproc   │   │
│  │ • SHAP on clean  │     │ • Reproducible pipes  │   │
│  └─────────────────┘     └──────────────────────┘   │
└─────────────────────────────────────────────────────┘
```

### Core Innovation: Strict Data Isolation

All preprocessing operations are `fit` **exclusively inside each CV training fold**:

1. ✅ **Missing value imputation** (class-conditional median) — fit per fold
2. ✅ **Feature scaling** (StandardScaler) — fit per fold
3. ✅ **SMOTE oversampling** — instantiated per fold
4. ❌ **NO global preprocessing** before data splitting

This eliminates the data leakage that produces the 95-100% accuracy artifacts in existing literature.

### Ablation Study Design

| Level | Preprocessing | F1-Score | Recall | Inflation |
|:------|:-------------|:--------:|:------:|:---------:|
| No Leakage | Isolated SMOTE | 0.6759 | **0.7165** | — |
| Minor Leakage | Global imputation | 0.6759 | 0.7165 | 0% |
| Medium Leakage | Global imputation + scaling | 0.6777 | 0.7202 | +0.3% |
| **Severe Leakage** | **Global imputation + scaling + SMOTE** | **0.7338** | **0.7080 ↓** | **+8.6%** |

---

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- [scikit-learn](https://scikit-learn.org/)
- [SHAP](https://shap.readthedocs.io/)
- LaTeX distribution (for paper compilation)

### Quick Start

```bash
# Clone the repo
git clone https://github.com/yakeworld/crispdm-pima.git
cd crispdm-pima

# The complete modeling pipeline is on Kaggle:
# https://www.kaggle.com/code/yakeworld126/crisp-dm-pima

# Compile the paper (from paper/ directory)
cd paper
pdflatex paper.tex
bibtex paper
pdflatex paper.tex
pdflatex paper.tex
```

---

## 📊 Key Results

### 1. Baseline Performance (34 models, 10-fold CV)

Top 5 baseline models under strict isolation:

| Model | F1 | Recall | AUC |
|:------|:--:|:------:|:---:|
| GradientBoostingClassifier | 0.6857 | 0.7464 | 0.8383 |
| MLPClassifier | 0.6781 | 0.7239 | 0.8296 |
| CatBoostClassifier | 0.6766 | 0.7234 | 0.8354 |
| LogisticRegression | 0.6760 | 0.7165 | 0.8373 |
| LinearDiscriminantAnalysis | 0.6759 | 0.7091 | 0.8370 |

### 2. Feature Importance (SHAP)

Top 3 clinical predictors driving predictions:

- **Glucose**: Mean |SHAP| ≈ 0.16 (dominant)
- **BMI**: Mean |SHAP| ≈ 0.09
- **Age**: Mean |SHAP| ≈ 0.05

### 3. Comparison with Literature

| Study | Claimed Accuracy | Leakage Prevention | Process Framework | XAI |
|:------|:----------------:|:------------------:|:-----------------:|:---:|
| Kurniawan et al. | 100.0% | ❌ No | None | ❌ No |
| Chinnababu et al. | 99.81% | ❌ No | None | ❌ No |
| Akbar et al. | 99.50% | ❌ No | None | ❌ No |
| Talari et al. | 99.07% | ❌ No | None | ❌ No |
| **Ours (CRISP-DM Helix)** | **77.46% (credible)** | **✅ Strict CV isolation** | **CRISP-DM Helix** | **✅ SHAP** |

---

## 📄 Citation

```bibtex
@article{yang2025crispdm,
  author  = {Yang, Xiaokai and others},
  title   = {Process-Driven Credibility: A CRISP-DM Helix Framework 
             for Robust Pima Diabetes Prediction},
  journal = {arXiv preprint},
  year    = {2025},
  howpublished = {\url{https://github.com/yakeworld/crispdm-pima}}
}
```

---

## 📬 Contact

**杨晓凯** — Department of Neurology, Wenzhou People's Hospital  
📧 yakeworld@wmu.edu.cn  
🔬 GitHub: [@yakeworld](https://github.com/yakeworld)

---

## ⚖️ License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
