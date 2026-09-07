"""Evaluate fusion strategies on the shared three-detector benchmark.

Mirrors the pattern in evaluate_statistical.py / evaluate_spatial.py /
evaluate_combined.py: load a dataframe that already has ground truth
(is_anomaly) plus each detector's alert/score columns, run fusion, and
report observation-level + event-level metrics per strategy so they're
directly comparable to the existing baselines.

Usage:
    python evaluate_fusion.py

Expects (per DEVELOPER_GUIDE.md section 20):
    results/statistical_spatial_lstm/combined_full_results.parquet
with columns: is_anomaly, anomaly_id, statistical_alert, spatial_alert,
lstm_ae_alert, lstm_ae_score, etc.

Adjust INPUT_PATH / OUTPUT_DIR to match your repo layout.
"""
from __future__ import annotations

from pathlib import Path
import pandas as pd

from skyguard.fusion.engine import FusionEngine
from skyguard.fusion.strategies import OrStrategy, AndStrategy, KOfNStrategy, WeightedScoreStrategy
from skyguard.fusion.meta_model import LogisticMetaStrategy
from skyguard.fusion.adapters import build_evidence_dict
from skyguard.fusion.config import DETECTOR_REGISTRY

INPUT_PATH = Path("results/statistical_spatial_lstm/combined_full_results.parquet")
OUTPUT_DIR = Path("results/fusion")


def calculate_binary_metrics(df: pd.DataFrame, pred_col: str, truth_col: str = "is_anomaly") -> dict:
    """Same TP/FP/FN/TN + precision/recall/F1/FPR shape as the other evaluators."""
    tp = int(((df[truth_col]) & (df[pred_col])).sum())
    fp = int(((~df[truth_col]) & (df[pred_col])).sum())
    fn = int(((df[truth_col]) & (~df[pred_col])).sum())
    tn = int(((~df[truth_col]) & (~df[pred_col])).sum())

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    fpr = fp / (fp + tn) if (fp + tn) else 0.0

    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn, "precision": precision, "recall": recall, "f1": f1, "fpr": fpr}


def event_level_recall(df: pd.DataFrame, pred_col: str, event_col: str = "anomaly_id") -> float:
    """Fraction of ground-truth anomaly events with >=1 flagged observation."""
    anomalies = df[df["is_anomaly"]]
    if anomalies.empty:
        return 0.0
    detected_per_event = anomalies.groupby(event_col)[pred_col].any()
    return float(detected_per_event.mean())


def main():
    print(f"Loading {INPUT_PATH} ...")
    df = pd.read_parquet(INPUT_PATH)

    strategies = [
        OrStrategy(),
        AndStrategy(),
        KOfNStrategy(k=2),
        WeightedScoreStrategy(threshold=0.5),
    ]
    engine = FusionEngine(strategies)
    fused = engine.run(df)

    # --- Learned meta-model: fit on this same benchmark, then fuse ---
    evidence_rows = [build_evidence_dict(row, DETECTOR_REGISTRY) for _, row in df.iterrows()]
    meta = LogisticMetaStrategy()
    meta.fit(evidence_rows, df["is_anomaly"].tolist())
    fused["fusion_logistic_meta_alert"] = [meta.fuse(ev).alert for ev in evidence_rows]
    fused["fusion_logistic_meta_score"] = [meta.fuse(ev).score for ev in evidence_rows]
    print(f"Logistic meta-model coefficients ({meta.feature_names()}): {meta.model.coef_}")

    # --- Report metrics per strategy, same shape as the other evaluators ---
    all_strategy_names = [s.name for s in strategies] + ["logistic_meta"]
    rows = []
    for name in all_strategy_names:
        alert_col = f"fusion_{name}_alert"
        m = calculate_binary_metrics(fused, alert_col)
        m["event_recall"] = event_level_recall(fused, alert_col)
        m["strategy"] = name
        rows.append(m)

    summary = pd.DataFrame(rows).set_index("strategy")
    print("\n=== FUSION STRATEGY COMPARISON ===")
    print(summary[["precision", "recall", "f1", "fpr", "event_recall"]].round(4))

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fused.to_parquet(OUTPUT_DIR / "fusion_full_results.parquet")
    summary.to_csv(OUTPUT_DIR / "fusion_strategy_comparison.csv")
    print(f"\nSaved results to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()