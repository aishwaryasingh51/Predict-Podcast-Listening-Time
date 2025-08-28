<div align="center">

# 🎙️ Predict Podcast Listening Time

**Capstone Project – Professional Certificate in Data Analytics (Imperial College Business School)**  
Author: *Aishwarya Singh*

</div>

## 🧠 Problem Statement
Given podcast episode metadata (content, engagement & production attributes), predict how many **minutes a listener will spend** on the episode (`Listening_Time_minutes`). Accurate estimates can guide:  
- Ad inventory planning  
- Episode length optimization  
- Audience retention strategies  
- Personalized recommendations

The competition evaluates submissions using **Root Mean Squared Error (RMSE)**.

## 📂 Dataset
| File | Description |
|------|-------------|
| `podcast_train.csv` | Training set with features + target `Listening_Time_minutes` |
| `podcast_test.csv`  | Test set without target (you predict it) |
| `podcast_sample_submission.csv` | Format reference for submissions |
| `podcast.ipynb` | End‑to‑end exploratory + modeling workflow |

> Note: Data was synthetically generated from a model trained on an original podcast dataset; distributions are similar but not identical. This allows experimentation with domain adaptation ideas.

## 🔍 Approach Overview
1. **Exploration & Profiling**: Structure, types, missingness, cardinality, target distribution, train vs test drift checks.  
2. **Feature Handling**:  
	- Dropped `id` (identifier only).  
	- Median imputation for numeric, mode imputation for categorical.  
	- One‑hot encoding (drop-first) for tree-based models.  
3. **Baseline**: Mean target predictor to anchor improvement.  
4. **Models Evaluated**:  
	- Decision Tree (manual grid)  
	- Random Forest (targeted grid: depth, estimators, min leaf)  
	- HistGradientBoostingRegressor (learning rate, depth, iterations)  
5. **Model Selection**: Best validation RMSE achieved with Histogram Gradient Boosting.  
6. **Final Training**: Retrained chosen model on full training data (early stopping disabled).  
7. **Inference**: Align encoded test columns to train matrix; generate `submission.csv`.

## 📊 Key Insights (from Notebook)
- Target distribution moderately spread; tree ensembles benefit from depth control + leaf regularization.
- A handful of numeric features exhibit strongest correlation with listening time; categorical effects are uneven (some high cardinality). 
- Histogram Gradient Boosting substantially reduces RMSE vs baseline (more than 50% improvement in validation comparison chart).

## 🛠️ Tech Stack / Dependencies
Core Python (≥3.10 recommended)

| Package | Purpose |
|---------|---------|
| `pandas`, `numpy` | Data wrangling |
| `matplotlib`, `seaborn` | Visualization |
| `scikit-learn` | Modeling & metrics |

Create a minimal `requirements.txt` (optional):
```
pandas
numpy
matplotlib
seaborn
scikit-learn
```

## ▶️ Quick Start
Clone & explore:
```bash
git clone https://github.com/aishwaryasingh51/Predict-Podcast-Listening-Time.git
cd Predict-Podcast-Listening-Time
python -m venv .venv && source .venv/bin/activate  # (macOS / Linux)
pip install -r requirements.txt  # if created
``` 

Open `podcast.ipynb` and run cells in order OR run a lean inference script (you can extract final model code from the notebook).

### Reproduce Submission (Notebook Flow)
1. Run import + EDA cells (optional but recommended).  
2. Execute preprocessing + model training sections.  
3. Final cells output `submission.csv` in repo root.  
4. Inspect head to verify formatting (`id,Listening_Time_minutes`).

## 📐 Evaluation Metric
RMSE:  
$$
\textrm{RMSE} = \left( \frac{1}{N} \sum_{i=1}^{N} (y_i - \widehat{y}_i)^2 \right)^{\frac{1}{2}}
$$

Lower is better. All reported model comparisons reference the same validation split (80/20, fixed seed = 51).

## 🧪 Potential Enhancements
| Category | Ideas |
|----------|-------|
| Feature Engineering | Ratios, interaction terms, target encoding (with CV), time-based derivations |
| Modeling | LightGBM / XGBoost, CatBoost (handles categoricals natively), stacked blending |
| Validation | K-fold CV, stratified by binned target, temporal split if time metadata exists |
| Regularization | Monotonic constraints (if domain knowledge), SHAP-based pruning |
| Deployment | Export fitted model via `pickle` / `joblib`; add API (FastAPI) endpoint |

## 📁 Suggested Repository Structure (Future Refactor)
```
.
├── data/                 # (Optionally move CSVs here; add to .gitignore if private)
├── notebooks/
│   └── podcast.ipynb
├── src/
│   ├── preprocess.py     # Reusable cleaning + encoding
│   ├── train.py          # Training entrypoint
│   └── infer.py          # Batch prediction
├── requirements.txt
└── README.md
```

## 🧭 Reusability Contract (Model Inference)
Inputs: DataFrame with same feature schema as training (excluding `id`, target).  
Outputs: 1D array of predicted `Listening_Time_minutes` (float).  
Assumptions: All missing values handled prior or via same imputation pipeline.  
Failure Modes: Unexpected categorical levels (handled by zero columns post alignment), missing columns.

## 🔒 Reproducibility Notes
- Fixed random seed: 51 across splits & models.  
- All preprocessing defined deterministically (median & mode from train set only).  
- Ensure identical library versions when comparing RMSE.

## 📜 License / Usage
Dataset usage subject to original competition terms (synthetic derivative). Code free for educational reuse—credit appreciated.

## 🙌 Acknowledgements
- Imperial College Business School – Analytics Program guidance.  
- Open-source ecosystem maintainers (scikit-learn, pandas, etc.).

## 📧 Contact
For questions or collaboration ideas, feel free to open an issue or discussion in the repository.

---
If you find this helpful, consider ⭐ starring the repo. Happy modeling! 🎧
