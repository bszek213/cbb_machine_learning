# College Basketball Game Predictions

Machine learning that predicts the outcome of any Division I college basketball game. Data are from 2015 - 2023 seasons.

## Usage

```python
python cbb_ml.py tune or python cbb_ml.py notune
```

```bash
Removed features (>=0.9 correlation): ['fta', 'opp_fg', 'opp_fta', 'opp_pf', 'def_rtg', 'fta_per_fga_pct', 'fg3a_per_fga_pct', 'ts_pct', 'stl_pct', 'blk_pct', 'efg_pct', 'tov_pct', 'orb_pct', 'ft_rate', 'opp_efg_pct', 'opp_tov_pct', 'drb_pct', 'opp_ft_rate']

### Current prediction accuracies - Random Forest
RMSE:  1.4578709886609726
R2 score:  0.985999084167356
```
### Correlation Matrix
![](https://github.com/bszek213/cbb_machine_learning/blob/dev/correlations.png)

### Feature Importances Regression
![](https://github.com/bszek213/cbb_machine_learning/blob/dev/feature_importance_random_forest.png)
## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.
