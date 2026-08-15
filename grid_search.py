# ============================================================
# XGBOOST GRID SEARCH USING DNABERT-6 FEATURES
# ============================================================
#
# Purpose:
#   1. Load DNA sequences and class labels.
#   2. Split data into training and independent test sets.
#   3. Extract DNABERT-6 embeddings from the training data.
#   4. Perform GridSearchCV only on the training data.
#   5. Identify the best XGBoost hyperparameters.
#   6. Save the best parameters and complete Grid Search results.
#
# IMPORTANT:
#   The independent test set is NOT used during Grid Search.
#   It should remain untouched for the final model evaluation
#   in Multiclass_Classification_dnabert.py.
#
# ============================================================


# ============================================================
# INSTALLATION
# ============================================================
#
# Run these commands once in Google Colab if required:
#
# !pip install torch transformers pandas numpy scikit-learn tqdm openpyxl xgboost joblib
#
# ============================================================


# ============================================================
# IMPORT LIBRARIES
# ============================================================

import os
import json
import random
import numpy as np
import pandas as pd
import torch

from tqdm import tqdm

from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    GridSearchCV
)

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

from torch.utils.data import Dataset, DataLoader

from transformers import (
    AutoTokenizer,
    AutoModel
)

import xgboost as xgb


# ============================================================
# CONFIGURATION
# ============================================================

class Config:

    # --------------------------------------------------------
    # DNABERT model
    # --------------------------------------------------------

    MODEL = "zhihan1996/DNA_bert_6"

    # Maximum number of tokens processed by DNABERT.
    MAX_LEN = 256

    # Batch size used during DNABERT feature extraction.
    BATCH = 16

    # Automatically use GPU when available.
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # Random seed for reproducibility.
    SEED = 42

    # --------------------------------------------------------
    # Data split
    # --------------------------------------------------------

    # 20% of the complete dataset is reserved as an
    # independent test set.
    TEST_SIZE = 0.20

    # Number of folds used by GridSearchCV.
    FOLDS = 5

    # --------------------------------------------------------
    # Output directories
    # --------------------------------------------------------

    OUTPUT_DIR = "outputs/grid_search"

    # --------------------------------------------------------
    # Input dataset
    # --------------------------------------------------------

    FILE = (
        "/media/andromeda_r740/Seagate Hub/"
        "MousumiB/MB/NIT_conf/Book1.xlsx"
    )


# ============================================================
# SET RANDOM SEEDS
# ============================================================

def set_seeds(seed=42):
    """
    Set random seeds to improve reproducibility.
    """

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


set_seeds(Config.SEED)


# ============================================================
# CREATE OUTPUT DIRECTORY
# ============================================================

os.makedirs(
    Config.OUTPUT_DIR,
    exist_ok=True
)


# ============================================================
# LOAD DNABERT TOKENIZER
# ============================================================

print("\nLoading DNABERT tokenizer...")

TOKENIZER = AutoTokenizer.from_pretrained(
    Config.MODEL
)

print("Tokenizer loaded successfully.")


# ============================================================
# DNA SEQUENCE → 6-MER TOKENS
# ============================================================

def create_kmers(sequence, k=6):
    """
    Convert a DNA sequence into space-separated 6-mer tokens.

    Example
    -------
    Input:
        ATCGATCG

    Output:
        ATCGAT TCGATC CGATCG
    """

    sequence = str(sequence)

    # If the sequence is shorter than k, return it directly.
    if len(sequence) < k:
        return sequence

    # Generate overlapping k-mers.
    kmers = [
        sequence[i:i + k]
        for i in range(len(sequence) - k + 1)
    ]

    # DNABERT expects the k-mers as space-separated tokens.
    return " ".join(kmers)


# ============================================================
# DATA LOADER
# ============================================================

class Loader:
    """
    Load DNA sequences and class labels from Excel/CSV files.
    """

    def __init__(self, file):

        # Detect file format automatically.
        if file.endswith(".xlsx"):

            self.df = pd.read_excel(
                file,
                engine="openpyxl"
            )

        elif file.endswith(".xls"):

            self.df = pd.read_excel(file)

        else:

            self.df = pd.read_csv(file)

    def load(self):
        """
        Load and clean sequences and labels.
        """

        # ----------------------------------------------------
        # Assume:
        #   Column 0 = DNA sequence
        #   Column 1 = class label
        # ----------------------------------------------------

        sequence_column = self.df.columns[0]

        class_column = self.df.columns[1]

        print(
            f"Sequence column: {sequence_column}"
        )

        print(
            f"Class column: {class_column}"
        )

        # ----------------------------------------------------
        # Clean DNA sequences.
        # ----------------------------------------------------

        sequences = (
            self.df[sequence_column]
            .astype(str)
            .str.upper()
            .str.replace(
                r"[^ATCG]",
                "",
                regex=True
            )
            .tolist()
        )

        # ----------------------------------------------------
        # Obtain labels.
        # ----------------------------------------------------

        labels = self.df[class_column].values

        # ----------------------------------------------------
        # Convert categorical labels into integers.
        # ----------------------------------------------------

        if labels.dtype == object:

            label_encoder = LabelEncoder()

            labels = label_encoder.fit_transform(
                labels
            )

            self.class_names = (
                label_encoder.classes_
            )

            print("\nClass encoding:")

            for index, name in enumerate(
                self.class_names
            ):
                print(
                    f"  {index} -> {name}"
                )

        else:

            self.class_names = np.unique(
                labels
            )

        # ----------------------------------------------------
        # Remove empty sequences.
        # ----------------------------------------------------

        valid_indices = [
            i
            for i, sequence in enumerate(sequences)
            if len(sequence) > 0
        ]

        sequences = [
            sequences[i]
            for i in valid_indices
        ]

        labels = labels[
            valid_indices
        ]

        # ----------------------------------------------------
        # Display dataset information.
        # ----------------------------------------------------

        number_of_classes = len(
            np.unique(labels)
        )

        print(
            f"\nTotal sequences: {len(sequences)}"
        )

        print(
            f"Number of classes: {number_of_classes}"
        )

        return (
            sequences,
            labels,
            number_of_classes,
            self.class_names
        )


# ============================================================
# DNABERT DATASET
# ============================================================

class DNADataset(Dataset):
    """
    PyTorch dataset used to prepare DNA sequences
    for DNABERT feature extraction.
    """

    def __init__(
        self,
        sequences,
        labels=None
    ):

        self.sequences = sequences

        self.labels = labels

        self.tokenizer = TOKENIZER

    def __len__(self):

        return len(self.sequences)

    def __getitem__(self, index):

        # ----------------------------------------------------
        # Convert DNA sequence into overlapping 6-mers.
        # ----------------------------------------------------

        text = create_kmers(
            self.sequences[index],
            k=6
        )

        # ----------------------------------------------------
        # Handle empty tokenized sequences.
        # ----------------------------------------------------

        if not text.strip():

            text = "A A A A A A"

        # ----------------------------------------------------
        # Tokenize using DNABERT tokenizer.
        # ----------------------------------------------------

        encoding = self.tokenizer(

            text,

            max_length=Config.MAX_LEN,

            padding="max_length",

            truncation=True,

            return_tensors="pt"
        )

        # Remove unnecessary batch dimension.
        input_ids = (
            encoding["input_ids"]
            .squeeze()
        )

        attention_mask = (
            encoding["attention_mask"]
            .squeeze()
        )

        # ----------------------------------------------------
        # Return labels when available.
        # ----------------------------------------------------

        if self.labels is not None:

            return {

                "ids": input_ids,

                "mask": attention_mask,

                "label": torch.tensor(
                    self.labels[index],
                    dtype=torch.long
                )
            }

        return {

            "ids": input_ids,

            "mask": attention_mask
        }


# ============================================================
# DNABERT FEATURE EXTRACTOR
# ============================================================

class DNABERTFeatureExtractor(torch.nn.Module):
    """
    Extract fixed-length feature vectors from DNABERT.
    """

    def __init__(self):

        super().__init__()

        print(
            "\nLoading DNABERT model..."
        )

        self.bert = AutoModel.from_pretrained(
            Config.MODEL
        )

        print(
            "DNABERT model loaded successfully."
        )

    def forward(
        self,
        input_ids,
        attention_mask
    ):

        # Pass the tokenized sequence through DNABERT.
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask
        )

        # ----------------------------------------------------
        # Mean pooling across the token dimension.
        #
        # This converts the sequence representation into
        # a fixed-length feature vector.
        # ----------------------------------------------------

        features = (
            outputs.last_hidden_state.mean(
                dim=1
            )
        )

        return features


# ============================================================
# FEATURE EXTRACTION FUNCTION
# ============================================================

def extract_features(
    model,
    dataloader,
    description
):
    """
    Extract DNABERT feature vectors from a dataset.
    """

    model.eval()

    feature_list = []

    label_list = []

    # Disable gradient calculation because DNABERT
    # is being used only as a feature extractor.
    with torch.no_grad():

        for batch in tqdm(
            dataloader,
            desc=description
        ):

            # Move input tensors to GPU/CPU.
            input_ids = (
                batch["ids"]
                .to(Config.DEVICE)
            )

            attention_mask = (
                batch["mask"]
                .to(Config.DEVICE)
            )

            # Generate DNABERT embeddings.
            features = model(
                input_ids,
                attention_mask
            )

            # Move features back to CPU.
            feature_list.append(
                features
                .cpu()
                .numpy()
            )

            # Store labels when available.
            if "label" in batch:

                label_list.append(
                    batch["label"]
                    .cpu()
                    .numpy()
                )

    # Combine all batches.
    features = np.vstack(
        feature_list
    )

    if label_list:

        labels = np.concatenate(
            label_list
        )

        return features, labels

    return features


# ============================================================
# XGBOOST GRID SEARCH
# ============================================================

def run_grid_search(
    X_train,
    y_train,
    number_of_classes
):
    """
    Perform GridSearchCV on XGBoost using DNABERT features.

    Grid Search is performed ONLY on the training data.

    The independent test set is never passed to this function.
    """

    print("\n")
    print("=" * 70)
    print("XGBOOST HYPERPARAMETER GRID SEARCH")
    print("=" * 70)

    # --------------------------------------------------------
    # Define the XGBoost classifier.
    # --------------------------------------------------------

    xgb_model = xgb.XGBClassifier(

        objective="multi:softprob",

        num_class=number_of_classes,

        eval_metric="mlogloss",

        random_state=Config.SEED,

        n_jobs=-1,

        tree_method="hist"
    )

    # --------------------------------------------------------
    # Define the parameter grid.
    #
    # These are the parameters that GridSearchCV will test.
    # --------------------------------------------------------

    parameter_grid = {

        "n_estimators": [
            100,
            200,
            300
        ],

        "max_depth": [
            3,
            6,
            7
        ],

        "learning_rate": [
            0.01,
            0.05,
            0.1
        ],

        "subsample": [
            0.8,
            1.0
        ],

        "colsample_bytree": [
            0.8,
            1.0
        ]
    }

    # --------------------------------------------------------
    # Calculate number of parameter combinations.
    # --------------------------------------------------------

    number_of_combinations = 1

    for parameter_values in parameter_grid.values():

        number_of_combinations *= (
            len(parameter_values)
        )

    print(
        f"\nParameter combinations: "
        f"{number_of_combinations}"
    )

    print(
        f"Cross-validation folds: "
        f"{Config.FOLDS}"
    )

    print(
        "Scoring metric: Macro F1"
    )

    print(
        "\nGrid Search is starting..."
    )

    # --------------------------------------------------------
    # Stratified K-fold cross-validation.
    #
    # Stratification maintains the class distribution in
    # every fold.
    # --------------------------------------------------------

    stratified_cv = StratifiedKFold(

        n_splits=Config.FOLDS,

        shuffle=True,

        random_state=Config.SEED
    )

    # --------------------------------------------------------
    # Create GridSearchCV.
    # --------------------------------------------------------

    grid_search = GridSearchCV(

        estimator=xgb_model,

        param_grid=parameter_grid,

        scoring="f1_macro",

        cv=stratified_cv,

        n_jobs=-1,

        verbose=2,

        return_train_score=True
    )

    # --------------------------------------------------------
    # IMPORTANT:
    #
    # Only X_train and y_train are supplied.
    #
    # The independent test set is NOT used.
    # --------------------------------------------------------

    grid_search.fit(
        X_train,
        y_train
    )

    # --------------------------------------------------------
    # Display the best result.
    # --------------------------------------------------------

    print("\n")
    print("=" * 70)
    print("GRID SEARCH COMPLETED")
    print("=" * 70)

    print(
        "\nBest Macro-F1 Score:"
    )

    print(
        f"{grid_search.best_score_:.4f}"
    )

    print(
        "\nBest XGBoost Parameters:"
    )

    for parameter, value in (
        grid_search.best_params_.items()
    ):

        print(
            f"  {parameter:25s}: {value}"
        )

    # --------------------------------------------------------
    # Save all Grid Search results.
    # --------------------------------------------------------

    results = pd.DataFrame(
        grid_search.cv_results_
    )

    results = results.sort_values(
        by="rank_test_score"
    )

    results.to_csv(

        os.path.join(
            Config.OUTPUT_DIR,
            "xgboost_grid_search_all_results.csv"
        ),

        index=False
    )

    # --------------------------------------------------------
    # Save a compact version of the results.
    # --------------------------------------------------------

    important_columns = [

        "rank_test_score",

        "mean_test_score",

        "std_test_score",

        "mean_train_score",

        "param_n_estimators",

        "param_max_depth",

        "param_learning_rate",

        "param_subsample",

        "param_colsample_bytree"
    ]

    compact_results = results[
        important_columns
    ]

    compact_results.to_csv(

        os.path.join(
            Config.OUTPUT_DIR,
            "xgboost_grid_search_summary.csv"
        ),

        index=False
    )

    # --------------------------------------------------------
    # Save the best parameters as JSON.
    #
    # This file will be used later when running
    # Multiclass_Classification_dnabert.py.
    # --------------------------------------------------------

    best_parameters = {

        key: (
            value.item()
            if hasattr(value, "item")
            else value
        )

        for key, value
        in grid_search.best_params_.items()
    }

    with open(

        os.path.join(
            Config.OUTPUT_DIR,
            "best_xgboost_parameters.json"
        ),

        "w"
    ) as file:

        json.dump(
            best_parameters,
            file,
            indent=4
        )

    # --------------------------------------------------------
    # Save the best CV score.
    # --------------------------------------------------------

    best_score = {

        "best_macro_f1": float(
            grid_search.best_score_
        ),

        "cv_folds": Config.FOLDS,

        "random_seed": Config.SEED,

        "number_of_classes": number_of_classes
    }

    with open(

        os.path.join(
            Config.OUTPUT_DIR,
            "grid_search_best_score.json"
        ),

        "w"
    ) as file:

        json.dump(
            best_score,
            file,
            indent=4
        )

    return grid_search


# ============================================================
# MAIN PROGRAM
# ============================================================

def main():

    print("\n")
    print("=" * 70)
    print(
        "DNABERT-6 + XGBOOST GRID SEARCH"
    )
    print("=" * 70)

    print(
        f"\nDevice: {Config.DEVICE}"
    )

    # --------------------------------------------------------
    # Check input file.
    # --------------------------------------------------------

    if not os.path.exists(
        Config.FILE
    ):

        raise FileNotFoundError(
            f"\nDataset not found:\n"
            f"{Config.FILE}\n\n"
            f"Please update Config.FILE."
        )

    # ========================================================
    # STEP 1: LOAD DATA
    # ========================================================

    print("\n")
    print("=" * 60)
    print("STEP 1: LOADING DATA")
    print("=" * 60)

    loader = Loader(
        Config.FILE
    )

    (
        sequences,
        labels,
        number_of_classes,
        class_names
    ) = loader.load()

    # ========================================================
    # STEP 2: INDEPENDENT TRAIN/TEST SPLIT
    # ========================================================
    #
    # The test set is separated BEFORE Grid Search.
    #
    # This prevents information from the test set leaking
    # into hyperparameter selection.
    #
    # ========================================================

    print("\n")
    print("=" * 60)
    print("STEP 2: TRAIN/TEST SPLIT")
    print("=" * 60)

    X_train_sequences, X_test_sequences, \
    y_train, y_test = train_test_split(

        sequences,

        labels,

        test_size=Config.TEST_SIZE,

        random_state=Config.SEED,

        stratify=labels
    )

    print(
        f"\nTraining sequences: "
        f"{len(X_train_sequences)}"
    )

    print(
        f"Independent test sequences: "
        f"{len(X_test_sequences)}"
    )

    # ========================================================
    # STEP 3: LOAD DNABERT
    # ========================================================

    print("\n")
    print("=" * 60)
    print("STEP 3: DNABERT-6 FEATURE EXTRACTION")
    print("=" * 60)

    dnabert = (
        DNABERTFeatureExtractor()
        .to(Config.DEVICE)
    )

    dnabert.eval()

    # --------------------------------------------------------
    # Create training dataset.
    # --------------------------------------------------------

    train_dataset = DNADataset(
        X_train_sequences,
        y_train
    )

    train_loader = DataLoader(

        train_dataset,

        batch_size=Config.BATCH,

        shuffle=False,

        num_workers=2,

        pin_memory=torch.cuda.is_available()
    )

    # --------------------------------------------------------
    # Extract training DNABERT features.
    # --------------------------------------------------------

    print(
        "\nExtracting training features..."
    )

    X_train_features, y_train_features = (
        extract_features(

            dnabert,

            train_loader,

            "DNABERT training features"
        )
    )

    print(
        f"\nTraining feature shape: "
        f"{X_train_features.shape}"
    )

    # ========================================================
    # STEP 4: SAVE TRAINING FEATURES
    # ========================================================

    print("\n")
    print("=" * 60)
    print("STEP 4: SAVING TRAINING FEATURES")
    print("=" * 60)

    pd.DataFrame(
        X_train_features
    ).to_csv(

        os.path.join(
            Config.OUTPUT_DIR,
            "DNABERT_train_features.csv"
        ),

        index=False
    )

    pd.DataFrame(
        y_train_features,
        columns=["label"]
    ).to_csv(

        os.path.join(
            Config.OUTPUT_DIR,
            "DNABERT_train_labels.csv"
        ),

        index=False
    )

    print(
        "Training features saved."
    )

    # ========================================================
    # STEP 5: RUN GRID SEARCH
    # ========================================================

    print("\n")
    print("=" * 60)
    print("STEP 5: XGBOOST GRID SEARCH")
    print("=" * 60)

    grid_search = run_grid_search(

        X_train_features,

        y_train_features,

        number_of_classes
    )

    # ========================================================
    # STEP 6: FINAL OUTPUT
    # ========================================================

    print("\n")
    print("=" * 70)
    print("GRID SEARCH SUMMARY")
    print("=" * 70)

    print(
        "\nBest Macro-F1:"
    )

    print(
        f"{grid_search.best_score_:.4f}"
    )

    print(
        "\nBest Parameters:"
    )

    for parameter, value in (
        grid_search.best_params_.items()
    ):

        print(
            f"  {parameter:25s}: {value}"
        )

    print("\n")
    print("=" * 70)
    print("GRID SEARCH FINISHED SUCCESSFULLY")
    print("=" * 70)

    print("\nFiles generated:")

    print(
        "  1. outputs/grid_search/"
        "best_xgboost_parameters.json"
    )

    print(
        "  2. outputs/grid_search/"
        "grid_search_best_score.json"
    )

    print(
        "  3. outputs/grid_search/"
        "xgboost_grid_search_all_results.csv"
    )

    print(
        "  4. outputs/grid_search/"
        "xgboost_grid_search_summary.csv"
    )

    print(
        "\nUse the parameters from "
        "'best_xgboost_parameters.json'"
    )

    print(
        "in your Multiclass_Classification_dnabert.py."
    )


# ============================================================
# PROGRAM ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()