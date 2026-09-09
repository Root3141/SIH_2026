"""Evaluate 3-class fusion strategies on the shared three-detector benchmark.

Reads the same file evaluate_combined.py already produced:
    results/statistical_spatial_lstm/combined_full_results.parquet
(columns confirmed from your run: is_anomaly, anomaly_id, anomaly_type,
statistical_alert, spatial_alert, lstm_ae_alert_original).

Reports metrics from TWO angles, because a 3-class output has two
meaningful cut points:
  - "anomaly-only"        : label == ANOMALY counted as positive
                             (this is your confirmed-alert tier)
  - "suspicious-or-higher": label in {SUSPICIOUS, ANOMALY} counted as
                             positive (this is your visibility/recall tier)
Comparing both against your evaluate_combined.py numbers tells you
whether the ANOMALY tier is behaving like any_two_of_three_1x
(FPR ~0.82%, event recall ~89%) as intended.

Usage:
    python evaluate_fusion.py
"""
from __future__ import annotations

from pathlib import Path
import pandas as pd

from skyguard.fusion.engine import FusionEngine
from skyguard.fusion.strategies import CountTierStrategy, WeightedTierStrategy
from skyguard.fusion.meta_model import LogisticMetaStrategy
from skyguard.fusion.adapters import build_evidence_dict
from skyguard.fusion.config import DETECTOR_REGISTRY
from skyguard.fusion.base import SUSPICIOUS, ANOMALY

INPUT_PATH = Path("results/statistical_spatial_lstm/combined_full_results.parquet")
OUTPUT_DIR = Path("results/fusion")


def calculate_binary_metrics(pred: pd.Series, truth: pd.Series) -> dict:
    """Same TP/FP/FN/TN + precision/recall/F1/FPR shape as evaluate_combined.py."""
    tp = int((truth & pred).sum())
    fp = int((~truth & pred).sum())
    fn = int((truth & ~pred).sum())
    tn = int((~truth & ~pred).sum())

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    fpr = fp / (fp + tn) if (fp + tn) else 0.0

    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn, "precision": precision, "recall": recall, "f1": f1, "fpr": fpr}


def event_level_recall(df: pd.DataFrame, pred: pd.Series, event_col: str = "anomaly_id") -> float:
    """Fraction of ground-truth anomaly events with >=1 flagged observation."""
    mask = df["is_anomaly"]
    if not mask.any():
        return 0.0
    detected_per_event = pred[mask].groupby(df.loc[mask, event_col]).any()
    return float(detected_per_event.mean())


def evaluate_label_column(df: pd.DataFrame, label_col: str) -> dict:
    """Report both cut points for one strategy's label column."""
    truth = df["is_anomaly"]
    anomaly_only = df[label_col] == ANOMALY
    susp_or_higher = df[label_col].isin([SUSPICIOUS, ANOMALY])

    strict = calculate_binary_metrics(anomaly_only, truth)
    strict["event_recall"] = event_level_recall(df, anomaly_only)

    lenient = calculate_binary_metrics(susp_or_higher, truth)
    lenient["event_recall"] = event_level_recall(df, susp_or_higher)

    return {"anomaly_only": strict, "suspicious_or_higher": lenient}


def main():
    print(f"Loading {INPUT_PATH} ...")
    df = pd.read_parquet(INPUT_PATH)

    strategies = [
        CountTierStrategy(n_suspicious=1, n_anomaly=2),  # default: matches any_two_of_three_1x
        CountTierStrategy(n_suspicious=1, n_anomaly=3, name="count_tier_strict_3of3"),
        WeightedTierStrategy(),
    ]
    engine = FusionEngine(strategies)
    fused = engine.run(df)

    # --- Learned meta-model ---
    evidence_rows = [build_evidence_dict(row, DETECTOR_REGISTRY) for _, row in df.iterrows()]
    meta = LogisticMetaStrategy()
    meta.fit(evidence_rows, df["is_anomaly"].tolist())
    results = [meta.fuse(ev) for ev in evidence_rows]
    fused["fusion_logistic_meta_label"] = [r.label for r in results]
    fused["fusion_logistic_meta_score"] = [r.score for r in results]
    print(f"Logistic meta-model features {meta.feature_names()}, coefficients {meta.model.coef_}")

    # --- Report, matching evaluate_combined.py's metric shape ---
    all_names = [s.name for s in strategies] + ["logistic_meta"]
    rows = []
    for name in all_names:
        report = evaluate_label_column(fused, f"fusion_{name}_label")
        for view, m in report.items():
            rows.append({"strategy": name, "view": view, **m})

    summary = pd.DataFrame(rows).set_index(["strategy", "view"])
    print("\n=== FUSION STRATEGY COMPARISON (vs. your evaluate_combined.py baselines) ===")
    print(summary[["precision", "recall", "f1", "fpr", "event_recall"]].round(4))
    print(
        "\nReference from evaluate_combined.py: any_two_of_three_1x had "
        "FPR 0.8175%, event_recall 89.06%. Compare 'anomaly_only' rows above to that."
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fused.to_parquet(OUTPUT_DIR / "fusion_full_results.parquet")
    summary.to_csv(OUTPUT_DIR / "fusion_strategy_comparison.csv")
    print(f"\nSaved results to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()