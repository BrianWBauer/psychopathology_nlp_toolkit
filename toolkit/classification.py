"""
Supervised Classification on Embeddings
=========================================
Train simple classifiers on manually labeled embedding subsets and
scale predictions to an unlabeled corpus.

Workflow:
    1. Human annotators label a small subset (e.g., 200-500 documents)
       with binary or multi-class labels
    2. Generate embeddings for the labeled subset
    3. Train a regularized logistic classifier on the embeddings
    4. Evaluate via stratified k-fold cross-validation
    5. Apply the trained model to the full unlabeled corpus
    6. Use predicted labels and probabilities for downstream analysis
       (e.g., prevalence estimates, temporal trends, cross-group comparisons)

This approach complements the unsupervised methods in Goals 1-3 by
incorporating prior knowledge through human judgment, enabling
quantitative analyses not possible with centroids alone (e.g., count
of internalizing posts per subreddit over time).
"""

import numpy as np
import pandas as pd
from typing import Optional, List, Dict, Union
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    cross_validate,
)
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    confusion_matrix,
)
from sklearn.calibration import CalibratedClassifierCV
import warnings


def train_embedding_classifier(
    embeddings: np.ndarray,
    labels: np.ndarray,
    model_type: str = "logistic",
    cv_folds: int = 5,
    test_size: float = 0.2,
    C: float = 1.0,
    max_iter: int = 1000,
    random_state: int = 42,
) -> dict:
    """
    Train a classifier on labeled embeddings with cross-validated evaluation.

    Uses stratified k-fold CV for robust performance estimation, then
    retrains on the full labeled set for deployment to unlabeled data.

    Parameters
    ----------
    embeddings : np.ndarray
        Shape (n_labeled, embedding_dim). Embeddings for labeled documents.
    labels : np.ndarray
        Class label for each labeled document.
    model_type : str
        'logistic' (L2-regularized logistic regression, recommended) or
        'svm' (linear SVM with Platt scaling for probability estimates).
    cv_folds : int
        Number of stratified CV folds for evaluation. A single train/test
        split is insufficient for the small labeled sets typical in this
        workflow (200-500 documents).
    test_size : float
        Proportion held out for a final evaluation split (in addition to
        CV). Set to 0 to use all data for CV only.
    C : float
        Inverse regularization strength. Lower values = stronger
        regularization. Use GridSearchCV for tuning if performance
        is suboptimal.
    max_iter : int
        Maximum iterations for convergence.
    random_state : int
        For reproducibility.

    Returns
    -------
    dict with keys:
        'model' : fitted classifier (trained on full labeled set)
        'cv_results' : dict of cross-validation metrics
        'holdout_report' : dict (if test_size > 0)
        'holdout_auc' : float (if test_size > 0)
        'classes' : np.ndarray of class labels
    """
    labels = np.asarray(labels)
    classes = np.unique(labels)
    n_classes = len(classes)

    print(f"Training {model_type} classifier")
    print(f"  Labeled samples: {len(labels)}")
    print(f"  Classes: {dict(zip(*np.unique(labels, return_counts=True)))}")

    # ── Build classifier ─────────────────────────────────────────────────
    if model_type == "logistic":
        clf = LogisticRegression(
            C=C,
            class_weight="balanced",
            max_iter=max_iter,
            random_state=random_state,
            solver="lbfgs",
        )
    elif model_type == "svm":
        # LinearSVC doesn't produce probabilities natively; wrap with
        # CalibratedClassifierCV for predict_proba support
        base_svm = LinearSVC(
            C=C,
            class_weight="balanced",
            max_iter=max_iter,
            random_state=random_state,
        )
        clf = CalibratedClassifierCV(base_svm, cv=3)
    else:
        raise ValueError(f"Unknown model_type: {model_type}. Use 'logistic' or 'svm'.")

    # ── Cross-validation ─────────────────────────────────────────────────
    scoring = ["accuracy", "f1_weighted", "precision_weighted", "recall_weighted"]
    if n_classes == 2:
        scoring.append("roc_auc")

    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cv_results = cross_validate(
            clf, embeddings, labels,
            cv=cv,
            scoring=scoring,
            return_train_score=False,
        )

    print(f"\n  {cv_folds}-fold CV results:")
    for metric in scoring:
        key = f"test_{metric}"
        scores = cv_results[key]
        print(f"    {metric}: {scores.mean():.3f} (±{scores.std():.3f})")

    result = {
        "cv_results": {
            metric: {
                "mean": float(cv_results[f"test_{metric}"].mean()),
                "std": float(cv_results[f"test_{metric}"].std()),
                "folds": cv_results[f"test_{metric}"].tolist(),
            }
            for metric in scoring
        },
        "classes": classes,
    }

    # ── Optional holdout evaluation ──────────────────────────────────────
    if test_size > 0:
        X_train, X_test, y_train, y_test = train_test_split(
            embeddings, labels,
            test_size=test_size,
            random_state=random_state,
            stratify=labels,
        )
        clf_holdout = _clone_classifier(model_type, C, max_iter, random_state, n_classes)
        clf_holdout.fit(X_train, y_train)

        y_pred = clf_holdout.predict(X_test)
        report = classification_report(y_test, y_pred, output_dict=True)
        result["holdout_report"] = report
        result["confusion_matrix"] = confusion_matrix(y_test, y_pred, labels=classes)

        # AUC
        try:
            y_prob = clf_holdout.predict_proba(X_test)
            if n_classes == 2:
                auc = roc_auc_score(y_test, y_prob[:, 1])
            else:
                auc = roc_auc_score(y_test, y_prob, multi_class="ovr")
            result["holdout_auc"] = float(auc)
            print(f"\n  Holdout AUC: {auc:.3f}")
        except Exception:
            result["holdout_auc"] = None

    # ── Train final model on all labeled data ────────────────────────────
    final_clf = _clone_classifier(model_type, C, max_iter, random_state, n_classes)
    final_clf.fit(embeddings, labels)
    result["model"] = final_clf

    print(f"\n  Final model trained on all {len(labels)} labeled samples")
    return result


def _clone_classifier(model_type, C, max_iter, random_state, n_classes):
    """Helper to create a fresh classifier instance."""
    if model_type == "logistic":
        return LogisticRegression(
            C=C,
            class_weight="balanced",
            max_iter=max_iter,
            random_state=random_state,
            solver="lbfgs",
        )
    else:
        base_svm = LinearSVC(
            C=C,
            class_weight="balanced",
            max_iter=max_iter,
            random_state=random_state,
        )
        return CalibratedClassifierCV(base_svm, cv=3)


def predict_unlabeled(
    model,
    unlabeled_embeddings: np.ndarray,
    texts: Optional[List[str]] = None,
    confidence_threshold: float = 0.0,
) -> pd.DataFrame:
    """
    Apply a trained classifier to scale predictions across an unlabeled corpus.

    Parameters
    ----------
    model : fitted classifier
        Output from train_embedding_classifier()['model'].
    unlabeled_embeddings : np.ndarray
        Shape (n_unlabeled, embedding_dim).
    texts : list of str, optional
        Original texts for reference. If provided, included in output.
    confidence_threshold : float
        Minimum max-class probability to include a prediction.
        Documents below this threshold are labeled as 'uncertain'.
        Useful for flagging low-confidence predictions for human review.

    Returns
    -------
    pd.DataFrame
        Columns: predicted_class, confidence, [class probabilities], [text]
    """
    predictions = model.predict(unlabeled_embeddings)
    probabilities = model.predict_proba(unlabeled_embeddings)
    confidence = np.max(probabilities, axis=1)

    result = pd.DataFrame({
        "predicted_class": predictions,
        "confidence": confidence,
    })

    # Add per-class probabilities
    classes = model.classes_
    for i, cls in enumerate(classes):
        result[f"prob_{cls}"] = probabilities[:, i]

    # Apply confidence threshold
    if confidence_threshold > 0:
        result.loc[confidence < confidence_threshold, "predicted_class"] = "uncertain"

    if texts is not None:
        result.insert(0, "text", texts)

    print(f"Predicted {len(result)} documents")
    print(f"  Class distribution: {result['predicted_class'].value_counts().to_dict()}")
    if confidence_threshold > 0:
        n_uncertain = (result["predicted_class"] == "uncertain").sum()
        print(f"  Below confidence threshold: {n_uncertain} ({n_uncertain/len(result):.1%})")
    print(f"  Mean confidence: {confidence.mean():.3f}")

    return result


def evaluate_label_quality(
    embeddings: np.ndarray,
    labels: np.ndarray,
    n_splits: int = 10,
    random_state: int = 42,
) -> dict:
    """
    Quick diagnostic: are the human labels learnable from the embeddings?

    Runs repeated stratified CV with a simple logistic classifier to
    estimate the ceiling performance. Low accuracy suggests either:
        (a) the construct is poorly captured by the embedding model, or
        (b) the human labels are noisy / ambiguous.

    Parameters
    ----------
    embeddings : np.ndarray
        Labeled embeddings.
    labels : np.ndarray
        Human annotations.
    n_splits : int
        Number of CV folds.

    Returns
    -------
    dict with accuracy, F1, and interpretation.
    """
    clf = LogisticRegression(
        C=1.0, class_weight="balanced", max_iter=1000,
        random_state=random_state,
    )
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    scores = cross_validate(
        clf, embeddings, labels, cv=cv,
        scoring=["accuracy", "f1_weighted"],
    )

    acc = scores["test_accuracy"].mean()
    f1 = scores["test_f1_weighted"].mean()

    interpretation = ""
    if acc >= 0.85:
        interpretation = "Strong — labels are well-captured by embeddings"
    elif acc >= 0.70:
        interpretation = "Moderate — usable but consider refining label definitions"
    elif acc >= 0.55:
        interpretation = "Weak — labels may be noisy or construct poorly captured"
    else:
        interpretation = "Near-chance — labels may not be learnable from these embeddings"

    print(f"Label quality diagnostic ({n_splits}-fold CV):")
    print(f"  Accuracy: {acc:.3f}")
    print(f"  F1 (weighted): {f1:.3f}")
    print(f"  Interpretation: {interpretation}")

    return {
        "accuracy": float(acc),
        "f1_weighted": float(f1),
        "interpretation": interpretation,
    }
