# WeakLabelStudio — measured local run

Supervised gold-fitted LF weighting on synthetic numeric alerts; not an unsupervised label model.

## Results

```json
{
  "majority_vote": {
    "accuracy": 0.69,
    "precision": 0.6902654867256637,
    "recall": 0.7428571428571429,
    "f1": 0.7155963302752294,
    "average_precision": 0.8035993694247976,
    "fp": 35,
    "fn": 27,
    "cases": 200
  },
  "gold_fitted_reliability": {
    "accuracy": 0.82,
    "precision": 0.8556701030927835,
    "recall": 0.7904761904761904,
    "f1": 0.8217821782178217,
    "average_precision": 0.9124449264087842,
    "fp": 14,
    "fn": 22,
    "cases": 200
  },
  "lf_reliabilities": [
    0.8589743589743589,
    0.5859872611464968,
    0.2980132450331126
  ],
  "lf_weights": [
    1.806797346592595,
    0.3474013071534031,
    -0.8567766043417473
  ],
  "all_abstain_evaluation": 5,
  "checks": [
    "Gold development and evaluation are separate",
    "Finite fitted LF reliability weights",
    "All-abstain cases return uncertainty"
  ],
  "scope": "Supervised gold-fitted LF weighting on synthetic numeric alerts; not an unsupervised label model."
}
```

Local experiment assertions passed. This does not establish production readiness or performance on real customer data. Source fingerprints and dependency versions are in results.json.
