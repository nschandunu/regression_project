# Results Summary — Fish Weight Regression

**Dataset:** `Fish.csv` — 159 rows, 7 species (Bream, Roach, Whitefish, Parkki, Perch, Pike, Smelt)

**Data cleaning:** One row (Roach, index 40) had `Weight = 0`, which is physically impossible for a fish — dropped as a data error, not treated as a normal outlier. 158 rows used.

**Features:** Length1, Length2, Length3, Height, Width (all numeric), plus Species one-hot encoded (6 dummy columns, Bream as baseline)
**Target:** Weight (grams)

**Split:** 75% train / 25% test (random_state=1)

## Test-set performance
| Metric | Value |
|---|---|
| RMSE | 98.3 g |
| R² | 0.939 |

## Interpretation

The model explains 94% of weight variance on unseen fish, with a typical error of about 98g against a mean weight of ~400g — roughly 25% relative error, driven mostly by a few very large outlier fish (e.g. the ~1650g Perch) that are harder to predict precisely. The three length measurements are highly correlated with each other (they all measure roughly the same thing at different points on the fish), which shows up as `Length1`'s coefficient going *negative* even though longer fish obviously weigh more — a classic multicollinearity artifact, not a real effect. Including Species mattered: `Smelt` and `Pike` species coefficients are large, meaning body shape/density differs by species even after accounting for size, which makes biological sense (a Smelt is much slimmer than a Pike of the same length). One caveat: the model can predict slightly negative weight for very small fish, since ordinary linear regression has no floor at zero — a reminder that this model should be used within the size range it was trained on, not extrapolated to unusually tiny or huge fish.
