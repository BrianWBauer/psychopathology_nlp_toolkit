"""
Goal 4: Supervised Embedding Classification
=============================================
Train classifiers on human-labeled text subsets and scale to full corpus.

Pipeline: Human Labels + Embeddings → Cross-Validated Classifier
          → Predict Full Corpus → Downstream Quantitative Analysis

This pipeline complements the unsupervised methods in Goals 1-3 by
using small amounts of human annotation to train simple classifiers
on embeddings, enabling quantitative analyses not possible with
unsupervised methods alone (e.g., prevalence rates, temporal trends,
cross-group comparisons).

Example use case from the manuscript context:
    - Code 300 Reddit posts as 'internalizing' vs 'not internalizing'
    - Train logistic classifier on post embeddings
    - Scale to 2.9M posts
    - Compute: proportion of internalizing content per subreddit,
      how internalizing content evolves within threads, etc.

Usage:
    python goal4_supervised_classification.py \
        --labeled_data annotated_subset.csv \
        --unlabeled_data full_corpus.csv \
        --text_col text --label_col label
"""

import argparse
import pandas as pd
import numpy as np
from pathlib import Path


def run_classification_pipeline(
    labeled_texts: list,
    labels: list,
    unlabeled_texts: list = None,
    embedding_model_name: str = "all-MiniLM-L6-v2",
    model_type: str = "logistic",
    cv_folds: int = 5,
    confidence_threshold: float = 0.0,
    output_dir: str = "output/goal4",
):
    """
    Run the full supervised classification pipeline.

    Parameters
    ----------
    labeled_texts : list of str
        Human-annotated documents.
    labels : list
        Class labels corresponding to labeled_texts.
    unlabeled_texts : list of str, optional
        Full corpus to classify. If None, only evaluation is run.
    embedding_model_name : str
        Sentence-transformer model for embedding generation.
    model_type : str
        'logistic' or 'svm'.
    cv_folds : int
        Number of stratified CV folds.
    confidence_threshold : float
        Min confidence to accept a prediction (0 = accept all).
    output_dir : str
        Directory for output files.
    """
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

    from toolkit.embeddings import generate_embeddings, load_model
    from toolkit.classification import (
        train_embedding_classifier,
        predict_unlabeled,
        evaluate_label_quality,
    )

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("GOAL 4: SUPERVISED CLASSIFICATION PIPELINE")
    print("=" * 60)

    labels = np.array(labels)

    # ── Step 1: Generate Embeddings for Labeled Data ─────────────────────
    print("\n--- Step 1: Embed Labeled Data ---")
    model = load_model(embedding_model_name)
    labeled_embeddings = generate_embeddings(labeled_texts, model=model)

    # ── Step 2: Label Quality Diagnostic ─────────────────────────────────
    print("\n--- Step 2: Label Quality Check ---")
    quality = evaluate_label_quality(labeled_embeddings, labels)

    # ── Step 3: Train and Evaluate Classifier ────────────────────────────
    print("\n--- Step 3: Train Classifier ---")
    clf_result = train_embedding_classifier(
        labeled_embeddings,
        labels,
        model_type=model_type,
        cv_folds=cv_folds,
        test_size=0.2,
    )

    # Save CV metrics
    cv_df = pd.DataFrame(clf_result["cv_results"]).T
    cv_df.to_csv(output_path / "cv_metrics.csv")

    if "holdout_report" in clf_result:
        holdout_df = pd.DataFrame(clf_result["holdout_report"]).T
        holdout_df.to_csv(output_path / "holdout_metrics.csv")

    if "confusion_matrix" in clf_result:
        cm_df = pd.DataFrame(
            clf_result["confusion_matrix"],
            index=clf_result["classes"],
            columns=clf_result["classes"],
        )
        cm_df.to_csv(output_path / "confusion_matrix.csv")
        print(f"\n  Confusion matrix:\n{cm_df}")

    # ── Step 4: Scale to Unlabeled Corpus ────────────────────────────────
    if unlabeled_texts:
        print(f"\n--- Step 4: Predict Unlabeled Corpus ({len(unlabeled_texts):,} docs) ---")
        unlabeled_embeddings = generate_embeddings(unlabeled_texts, model=model)

        predictions = predict_unlabeled(
            clf_result["model"],
            unlabeled_embeddings,
            texts=unlabeled_texts,
            confidence_threshold=confidence_threshold,
        )

        predictions.to_csv(output_path / "scaled_predictions.csv", index=False)
        print(f"  Predictions saved: {output_path / 'scaled_predictions.csv'}")
    else:
        print("\n  No unlabeled data provided; skipping prediction step.")

    print(f"\nAll results saved to {output_path}/")
    print("=" * 60)

    return clf_result


# ── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Goal 4: Supervised Embedding Classification"
    )
    parser.add_argument(
        "--labeled_data", required=True,
        help="CSV with labeled text data (must have text and label columns)"
    )
    parser.add_argument(
        "--unlabeled_data", default=None,
        help="CSV with unlabeled text data to classify"
    )
    parser.add_argument("--text_col", default="text")
    parser.add_argument("--label_col", required=True, help="Column with human labels")
    parser.add_argument("--model", default="all-MiniLM-L6-v2")
    parser.add_argument(
        "--classifier", default="logistic", choices=["logistic", "svm"]
    )
    parser.add_argument("--cv_folds", type=int, default=5)
    parser.add_argument("--confidence_threshold", type=float, default=0.0)
    parser.add_argument("--output_dir", default="output/goal4")

    args = parser.parse_args()

    labeled_df = pd.read_csv(args.labeled_data)
    labeled_texts = labeled_df[args.text_col].dropna().tolist()
    labels = labeled_df[args.label_col].dropna().tolist()

    unlabeled_texts = None
    if args.unlabeled_data:
        unlabeled_df = pd.read_csv(args.unlabeled_data)
        unlabeled_texts = unlabeled_df[args.text_col].dropna().tolist()

    run_classification_pipeline(
        labeled_texts=labeled_texts,
        labels=labels,
        unlabeled_texts=unlabeled_texts,
        embedding_model_name=args.model,
        model_type=args.classifier,
        cv_folds=args.cv_folds,
        confidence_threshold=args.confidence_threshold,
        output_dir=args.output_dir,
    )
