# Hotel Booking Cancellation Prediction — MLOps Project

A machine learning pipeline to predict whether a hotel booking will be cancelled. Built with MLOps best practices including modular pipeline stages, config-driven execution, and experiment tracking.

---

## Project Structure

```
MLPROJ1/
├── config/
│   ├── config.yaml          # Pipeline configuration (paths, column names, thresholds)
│   ├── paths_config.py      # Centralised file path constants
│   └── model_params.py      # LightGBM hyperparameter search space and RandomizedSearchCV settings
├── dataset/
│   └── Hotel_Reservations.csv   # Place your raw dataset here
├── artifacts/
│   ├── raw/                 # Output of data ingestion (raw, train, test CSVs)
│   └── processed/           # Output of preprocessing (cleaned, balanced CSVs)
├── src/
│   ├── data_ingestion.py    # Stage 1: Load and split raw data
│   ├── data_preprocessing.py  # Stage 2: Clean, encode, balance, select features
│   ├── model_training.py    # Stage 3: Train LightGBM with hyperparameter tuning + MLflow tracking
│   ├── logger.py            # Centralised logging
│   └── custom_exception.py  # Custom exception handler
├── utils/
│   └── common.py            # Shared utilities (read_yaml, load_data)
├── requirements.txt
└── setup.py
```

---

## Pipeline Stages

### Stage 1 — Data Ingestion (`src/data_ingestion.py`)

**What it does:**
- Copies the raw CSV from `dataset/` into `artifacts/raw/raw.csv`
- Splits the data into train and test sets based on `train_ratio` from config
- Saves `train.csv` and `test.csv` to `artifacts/raw/`

**Why:**
- Separates raw data from processed data to maintain reproducibility
- Train/test split is done here once so all downstream stages use the same split

**Run:**
```bash
python src/data_ingestion.py
```

---

### Stage 2 — Data Preprocessing (`src/data_preprocessing.py`)

The preprocessing pipeline has 4 steps:

#### 1. `preprocess_data`
- Drops `Unnamed: 0` (CSV index artifact) and `Booking_ID` (no predictive value)
- Removes duplicate rows to avoid model bias
- Drops rows with null values (dataset is large enough that dropping is safe)
- Label encodes categorical columns (converts strings to integers for model compatibility)
- Applies `log1p` transform to numerical columns with high skewness (reduces effect of extreme outliers)

#### 2. `balance_data`
- Uses **SMOTE** (Synthetic Minority Over-sampling Technique) to balance class distribution
- Why: Hotel booking datasets are typically imbalanced — more non-cancellations than cancellations. Without balancing, the model would be biased towards predicting the majority class.

#### 3. `select_features`
- Trains a **RandomForestClassifier** to rank all features by importance
- Keeps only the top N features (N is set via `no_of_features` in config)
- Why: Reduces noise and overfitting, speeds up training
- Test set columns are aligned to match train set features

#### 4. `save_data`
- Saves the final processed train and test CSVs to `artifacts/processed/`
- Saved without index to avoid writing extra unnamed columns

---

### Design Decisions

**Why no scaling/normalization?**
StandardScaler or MinMaxScaler is not applied because the primary model is **LightGBM** (tree-based). Tree-based models split on thresholds and are scale-invariant — scaling has no effect on them. If you switch to a distance-based model (SVM, KNN, Logistic Regression), add a scaling step after label encoding.

**Why no explicit outlier removal?**
Outliers in numerical columns are handled indirectly via `log1p` transform on skewed columns, which compresses extreme values. Explicit outlier removal (e.g. IQR-based) is not done to avoid losing valid edge-case data. LightGBM is robust to outliers by nature.

**Run:**
```bash
python src/data_preprocessing.py
```

---

### Stage 3 — Model Training (`src/model_training.py`)

The model training pipeline has 4 steps:

#### 1. `load_and_split_data`
- Loads processed train and test CSVs from `artifacts/processed/`
- Separates features (`X`) and target (`y`) for both sets
  - **Target (`y`)**: `booking_status` — what we are predicting (cancelled or not)
  - **Features (`X`)**: all remaining columns — the inputs the model learns patterns from

#### 2. `train_lgbm`
- Trains a **LightGBM** classifier
- Uses **RandomizedSearchCV** for hyperparameter tuning — samples random combinations from the search space defined in `config/model_params.py`
- Why RandomizedSearchCV over GridSearchCV: much faster, covers a wider search space, good enough for most cases
- Returns the best model found across all CV folds

#### 3. `evaluate_model`
- Evaluates on the held-out test set
- Computes **Accuracy, Precision, Recall, F1**
- F1 is the primary metric — balances precision and recall, more meaningful than accuracy on imbalanced data

#### 4. `save_model`
- Saves the trained model to `artifacts/models/lgbm_model.pkl` using **joblib**
- Why joblib over pickle: handles numpy arrays more efficiently

#### MLflow Integration
- Every run is tracked in MLflow — datasets, model file, hyperparameters, and metrics are all logged
- Allows comparing multiple runs in the MLflow UI to pick the best model

**Run:**
```bash
python src/model_training.py
```

---

### Design Decisions — Model Training

**Why LightGBM?**
- Purpose-built for tabular/structured data
- Histogram-based splits make it significantly faster than XGBoost on large datasets
- Scale-invariant (no normalization needed)
- Has built-in support for imbalanced data via `scale_pos_weight`

**Why F1 as the scoring metric?**
- Accuracy is misleading when classes are imbalanced
- F1 balances precision (avoid false cancellation alerts) and recall (catch actual cancellations)

**Hyperparameter search space** (`config/model_params.py`):

| Parameter | Range |
|---|---|
| `n_estimators` | 100 – 1000 |
| `max_depth` | 3 – 12 |
| `learning_rate` | 0.01, 0.05, 0.1, 0.2 |
| `num_leaves` | 20 – 150 |
| `min_child_samples` | 10 – 100 |

---

## Configuration (`config/config.yaml`)

```yaml
data_ingestion:
  local_data_path: "dataset/Hotel_Reservations.csv"
  train_ratio: 0.8

data_processing:
  categorical_columns: [...]
  numerical_columns: [...]
  skewness_threshold: 5
  no_of_features: 10
```

- `train_ratio`: fraction of data used for training (0.8 = 80% train, 20% test)
- `skewness_threshold`: columns with skewness above this value get log-transformed
- `no_of_features`: number of top features to keep after feature selection

---

## Setup

```bash
# Create and activate virtual environment
python3 -m venv venv && source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Place dataset
cp /path/to/Hotel_Reservations.csv dataset/

# Run pipeline stages
python src/data_ingestion.py
python src/data_preprocessing.py
python src/model_training.py
```

---

## Dependencies

| Package | Purpose |
|---|---|
| pandas, numpy | Data manipulation |
| scikit-learn | ML models, preprocessing, train/test split |
| imbalanced-learn | SMOTE for class balancing |
| lightgbm, xgboost | Gradient boosting models for training stage |
| mlflow | Experiment tracking |
| flask | Model serving API |
| pyyaml | Config file parsing |
