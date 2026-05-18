# Data Description

## Dataset Overview

This project uses aggregate-level hantavirus case-count time series data from the MV Hondius cruise ship. All data is aggregated to daily ship-level counts with no individually identifiable information.

## Data Files

### Raw data (`data/raw/`)

| File | Description |
|------|-------------|
| `mv_hondius_case_timeseries.csv` | Daily aggregate case counts (WHO DON reports) |
| `mv_hondius_model_ready.csv` | Model-ready variables (population, parameters) |
| `mv_hondius_repatriation_flights.csv` | Repatriation flight records (dates, volumes) |
| `mv_hondius_research_data_long_format.csv` | Long-format research data |
| `Cruise_Timeline.csv` | Event timeline (interventions, screenings, reports) |
| `Quick_Summary.csv` | Summary statistics |
| `README.csv` | Variable descriptions |

### Processed data (`data/processed/`)

| File | Description |
|------|-------------|
| `case_timeseries_standardized.csv` | Standardized daily aggregate time series (ready for training) |

## Standardized Data Fields

| Column | Type | Description |
|--------|------|-------------|
| `date` | str (YYYY-MM-DD) | Calendar date |
| `day_index` | int | Sequential day index (0-based) |
| `new_cases` | float | Daily new case count |
| `cumulative_cases` | float | Cumulative case count |
| `active_cases` | float | Estimated active cases |
| `intervention_index` | float [0, 1] | Intervention intensity index |
| `mobility_index` | float [0, 1] | Aggregate mobility index |
| `flight_volume` | float | Repatriation flight volume |
| `behavior_response_index` | float [0, 1] | Aggregate behavioral response index |
| `location` | str | Location label ("MV Hondius aggregate") |
| `source_note` | str | Data source annotation |
| `is_imputed` | bool | Whether values are imputed |
| `deaths` | float | Reported deaths |
| `population_total` | float | Estimated total population |

## Target Variable

- **Primary target**: `new_cases` (daily new case count)

## Time Range

- Start: 2026-05-04
- End: 2026-05-13 (10 days of data)
- Frequency: Daily

## Aggregation Level

- **Ship-level aggregate** (MV Hondius)
- No individual passenger data
- No person-level movement or contact data

## Privacy

This dataset contains only aggregate-level public health surveillance data. No personally identifiable information (PII), individual health records, or passenger-level data is included.

## Train/Validation/Test Split

Due to the small sample size (10 days), all data is stored in a single file:
- `data/processed/case_timeseries_standardized.csv`

Splits are created via time-ordered partitioning at training time:
- Training: earliest 80% of observations
- Validation: next portion (if available)
- Test: latest observations

No random shuffling is used; all splits respect temporal ordering.

## Feature Columns Used for Training

The default feature order for supervised windows is:
1. `new_cases`
2. `cumulative_cases`
3. `intervention_index`
4. `mobility_index`
5. `flight_volume`
6. `behavior_response_index`
7. `day_index`

## Notes

- The dataset is **very small** (10 days). This is a known limitation.
- Missing values in `flight_volume` are filled with 0.0.
- Missing intervention index values default to 0.2.
- `behavior_response_index` is derived from keyword-based coding of the event timeline.
- Imputed values are flagged with `is_imputed=True`.
