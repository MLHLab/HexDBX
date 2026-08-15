# Interpreting Antibiotic Resistance Genes in *Staphylococcus aureus*: A DNABERT-XGBoost Framework Using Hexamers

## 📌 Overview

This project presents a **DNABERT-XGBoost framework for the classification and interpretation of antibiotic resistance genes (ARGs) in *Staphylococcus aureus*.**

The framework combines:

* **Overlapping 6-mer (hexamer) tokenization** for DNA sequences
* **Pretrained DNABERT-6** for contextualized DNA sequence representation
* **XGBoost** for multiclass classification
* **GridSearchCV** for hyperparameter optimization
* **SHAP** for model interpretability
* **Functional domain mapping** to connect important sequence features with known antibiotic-resistance mechanisms

The combination of deep sequence representations and gradient-boosting classification provides a powerful approach for ARG classification while maintaining interpretability at both the feature and biological-mechanism levels.

---

## 🧬 Project Highlights

### What This Project Does

1. Collects *S. aureus* genomes from NCBI.
2. Identifies and annotates antibiotic resistance genes using the CARD database.
3. Assigns ARG sequences to five antibiotic/drug classes.
4. Converts DNA sequences into overlapping 6-mer tokens.
5. Extracts contextualized sequence embeddings using pretrained DNABERT-6.
6. Uses mean pooling to obtain 768-dimensional sequence representations.
7. Trains an XGBoost multiclass classifier.
8. Optimizes XGBoost hyperparameters using GridSearchCV with 5-fold stratified cross-validation.
9. Evaluates performance on an independent test set.
10. Uses SHAP to identify influential features.
11. Maps important features to biologically relevant resistance domains and mechanisms.

---

## 📁 Project Structure

```text
dnabert-xgboost-arg/
│
├── README.md
├── requirements.txt
├── LICENSE
│
├── Multiclass_Classification_dnabert.py
├── shap_code.py
├── kmer_search.py
├── gridsearch_xgboost.py
│
├── data/
│   └── Dataset.xlsx
│
├── models/
│   ├── xgb_final_model_*.json
│   ├── scaler_*.pkl
│   ├── class_names_*.npy
│   ├── best_params_*.json
│   └── metadata_*.json
│
└── outputs/
    │
    ├── features/
    │   ├── train_features.csv
    │   ├── test_features.csv
    │   ├── train_labels.csv
    │   └── test_labels.csv
    │
    ├── xgb/
    │   ├── classification_report_test.csv
    │   ├── feature_importance.csv
    │   ├── best_params.csv
    │   ├── gridsearch_results.csv
    │   └── performance_comparison_*.png
    │
    ├── shap_analysis/
    │   ├── class_wise/
    │   ├── violin_plots/
    │   └── summary_plots/
    │
    ├── roc/
    ├── confusion/
    ├── predictions/
    ├── class_performance/
    └── loss/
```

---

## 🔬 Future Work

Potential extensions of this framework include:

* Evaluation on independent external datasets
* Genome-level and cluster-based splitting
* Comparison with traditional k-mer machine-learning approaches
* Comparison with CNN, RNN, and Transformer classifiers
* Fine-tuning DNABERT on ARG-specific sequences
* Incorporation of protein-level representations
* Multi-label ARG prediction
* Resistance-gene subtype classification
* Integration of genomic metadata
* Improved biological annotation of SHAP-important regions
* Cross-species ARG classification
* Deployment as a web-based ARG prediction tool

---

## 🙏 Acknowledgments

This project makes use of resources and technologies developed by:

* **Comprehensive Antibiotic Resistance Database (CARD)** for antibiotic resistance gene annotations
* **NCBI** for genomic sequence resources
* **DNABERT** development team for pretrained genomic language models
* **SHAP** development team for model interpretability tools
* **XGBoost** development team for gradient-boosting algorithms
* **JIS Institute of Advanced Studies and Research** for institutional support

---


