"""Explain the existing vote rule without reading evaluation metadata."""
import numpy as np
from skyguard.fusion.fusion import MINIMUM_VOTES


def explain_decision(row):
    votes = {}
    for name, column in [('statistical', 'statistical_alert'), ('spatial', 'spatial_alert'), ('lstm', 'lstm_ae_alert')]:
        if not isinstance(row[column], (bool, np.bool_)):
            raise ValueError(f'{column} must be boolean')
        votes[name] = bool(row[column])
    count = sum(votes.values())
    if (row['detector_votes'] != count or not isinstance(row['final_alert'], (bool, np.bool_))
            or bool(row['final_alert']) != (count >= MINIMUM_VOTES)):
        raise ValueError('Inconsistent fused decision; explanation refused')
    summary = (f'{count} of 3 detectors agreed, so the final anomaly alert was raised.'
               if row['final_alert'] else
               f'{count} of 3 detectors agreed; fewer than {MINIMUM_VOTES} votes, so no final anomaly alert was raised.')
    return dict(alert_explanation_type='deterministic_vote', alert_explanation_summary=summary,
                detector_vote_summary=votes, statistical_vote=votes['statistical'],
                spatial_vote=votes['spatial'], lstm_vote=votes['lstm'], agreement_count=count,
                agreement_required=MINIMUM_VOTES, final_alert=bool(row['final_alert']),
                final_severity=row['final_severity'], final_confidence=float(row['final_confidence']),
                confidence_meaning='detector agreement fraction, not a calibrated probability')
