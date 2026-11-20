# 01 · Data audit

Goal: confirm the files match the brief and list every structural problem before any analysis. Nothing is fixed here, only counted.

## 1. Schema

| column | train | validation | december |
| --- | --- | --- | --- |
| load_id | object | object | — |
| pickup | object | object | object |
| delivery | object | object | object |
| pickup_lat | float64 | float64 | — |
| pickup_lon | float64 | float64 | — |
| delivery_lat | float64 | float64 | — |
| delivery_lon | float64 | float64 | — |
| distance | float64 | float64 | int64 |
| equipment | object | object | object |
| weight | float64 | float64 | int64 |
| date | datetime64[ns] | datetime64[ns] | datetime64[ns] |
| market_index | float64 | float64 | — |
| quote_signal | float64 | float64 | — |
| posted_rate | float64 | — | — |
| predicted_rate | — | — | float64 |

- In train but not validation: `['posted_rate']` (the target).
- Model inputs **missing from the December file**: `['pickup_lat', 'pickup_lon', 'delivery_lat', 'delivery_lon', 'market_index', 'quote_signal']`.

## 2. Size and date coverage

| set | rows | first date | last date | unique dates | days missing in range | loads/day mean | loads/day min | loads/day max |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| train | 48,000 | 2025-01-01 | 2025-10-31 | 304 | 0 | 157.9 | 115 | 198 |
| validation | 12,000 | 2025-11-01 | 2025-12-31 | 61 | 0 | 196.7 | 163 | 227 |
| december | 31 | 2025-12-01 | 2025-12-31 | 31 | 0 | 1.0 | 1 | 1 |

## 3. ID and file-contract checks

| check | result | detail |
| --- | --- | --- |
| train load_id unique | PASS |  |
| train load_id format TR-NNNNNN | PASS |  |
| validation load_id unique | PASS |  |
| validation load_id format TE-NNNNNN | PASS |  |
| validation ids == TE-000001..TE-012000 (what score.py expects) | PASS |  |
| template columns == [load_id, predicted_rate] | PASS |  |
| template ids == validation ids (same set) | PASS |  |
| template ids in same row order as validation | PASS |  |
| template predicted_rate all empty | PASS |  |
| December: 31 consecutive dates 2025-12-01..31 | PASS |  |
| December: fixed fields constant | PASS | pickup=Lexington, delivery=Fort Wayne, distance=360, equipment=Dry Van, weight=32000 |
| December predicted_rate all empty | PASS |  |

## 4. Missing values

| column | train missing | train % | validation missing | validation % |
| --- | --- | --- | --- | --- |
| weight | 300 | 0.62 | 165 | 1.38 |
| market_index | 374 | 0.78 | 249 | 2.08 |

Rows in train missing **both** weight and market_index: 1 (independent missingness would give ≈2.3).

## 5. Duplicates

| set | exact duplicate rows (ignoring load_id) | identical inputs | same lane+date+equipment (not an error by itself) |
| --- | --- | --- | --- |
| train | 0 | 0 | 495 |
| validation | 0 | 0 | 123 |

Validation rows whose inputs exactly match a train row: 0.

## 6. Numeric ranges

| index | min | p1 | median | p99 | max |
| --- | --- | --- | --- | --- | --- |
| distance (train) | 70.000 | 121.9 | 953.3 | 3,029 | 3,440 |
| distance (validation) | 70.000 | 119.4 | 953.6 | 3,014 | 3,326 |
| market_index (train) | 0.676 | 0.783 | 1.056 | 1.407 | 1.468 |
| market_index (validation) | 0.724 | 0.791 | 0.923 | 1.066 | 1.099 |
| posted_rate (train) | 57.220 | 327.2 | 2,031 | 5,973 | 25,533 |
| quote_signal (train) | 0.692 | 1.305 | 2.056 | 2.884 | 3.610 |
| quote_signal (validation) | 1.235 | 1.542 | 2.051 | 2.564 | 2.896 |
| weight (train) | -47,500 | 9,802 | 31,436 | 47,500 | 47,500 |
| weight (validation) | -47,500 | -24,821 | 31,193 | 47,500 | 47,500 |

## 7. Impossible or suspicious values

| rule | train | validation |
| --- | --- | --- |
| weight < 0 | 292 | 145.0 |
| weight == 0 | 0 | 0.000 |
| weight == 47,500 (suspected cap) | 1,191 | 297.0 |
| distance <= 0 | 0 | 0.000 |
| market_index <= 0 | 0 | 0.000 |
| quote_signal <= 0 | 0 | 0.000 |
| posted_rate <= 0 | 0 |  |
| pickup == delivery | 0 | 0.000 |

## 8. Categorical hygiene

| equipment (% of rows) | train | validation |
| --- | --- | --- |
| Dry Van | 56.7 | 56.5 |
| Reefer | 25.1 | 25.4 |
| Flatbed | 18.2 | 18.1 |

- Equipment levels: ['Dry Van', 'Flatbed', 'Reefer']
- City names with stray whitespace: none; case-variant clashes: none
- Cities: train 64, validation 72
- **Cities only in validation (8):** ['Allentown', 'Charlotte', 'Chicago', 'Jackson', 'Knoxville', 'Laredo', 'Norfolk', 'San Diego']
- Cities only in train: none
- Distinct lanes in train: 4,014 of 4,032 possible (median 10 loads per lane)

## 9. Coordinates

- Cities with more than one (lat, lon) across roles/files: 0 → coordinates are a fixed lookup per city.
- Extreme coordinate values below. Several cities sharing a round extreme value is the signature of clipping to a bounding box.

| coordinate | value | cities at exactly this value |
| --- | --- | --- |
| lat min | 25.50000 | Laredo |
| lat max | 44.30296 | Green Bay |
| lon min | -121.7 | San Francisco |
| lon max | -69.50000 | Boston, Providence |
