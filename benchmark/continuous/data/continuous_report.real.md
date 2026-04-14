# Continuous Benchmark Report

## Artifacts

- Dataset: [continuous_dataset.real.json](continuous_dataset.real.json)
- Predictions: [continuous_predictions.real.json](continuous_predictions.real.json)
- Metrics: [continuous_eval.real.json](continuous_eval.real.json)

## Overall Results

| Metric | Value |
| --- | ---: |
| Num GT samples | 60 |
| Num evaluated | 60 |
| Missing predictions | 0 |
| Exact match text | 0.0000 |
| Exact match gloss | 0.0000 |
| WER | 1.1042 |
| CER | 1.1038 |

## Breakdown by Dialect

| Dialect | GT samples | Evaluated | WER | CER | EM text | EM gloss |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| central | 3 | 3 | 1.0000 | 1.0354 | 0.0000 | 0.0000 |
| south | 2 | 2 | 1.0000 | 1.3171 | 0.0000 | 0.0000 |
| unknown | 55 | 55 | 1.1136 | 1.0998 | 0.0000 | 0.0000 |

## Breakdown by Signer

| Signer | GT samples | Evaluated | WER | CER | EM text | EM gloss |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| N | 3 | 3 | 1.0000 | 1.0354 | 0.0000 | 0.0000 |
| T | 2 | 2 | 1.0000 | 1.3171 | 0.0000 | 0.0000 |
| U | 55 | 55 | 1.1136 | 1.0998 | 0.0000 | 0.0000 |

## Notes

- The real continuous benchmark ran end-to-end without missing predictions.
- The model currently remains a baseline and is not yet accurate enough for production claims.
- Use this report together with the dataset JSON for the thesis/report appendix.