# College Basketball Game Predictions

Machine learning that predicts the outcome of any Division I college basketball game. Data are from 2015 - 2023 seasons.

## Usage

```python
python cbb_ml.py tune or python cbb_ml.py notune
```

```bash
Removed features (>=0.9 correlation): ['fta', 'fta_per_fga_pct', 'fg3a_per_fga_pct', 'ts_pct', 'stl_pct', 'blk_pct', 'efg_pct', 'tov_pct', 'orb_pct', 'ft_rate']
Number of samples: 78242

### Current prediction accuracies - Random Forest
# After 10 fold cross validation and pre-processing
Current RandomForestRegressor Parameters: {'criterion': 'squared_error', 'max_depth': 20, 'max_features': 'log2', 'min_samples_leaf': 1, 'n_estimators': 400}
RMSE:  0.7552642670840531
R2 score:  0.9965770683833447
```
### Correlation Matrix
![](https://github.com/bszek213/cbb_machine_learning/blob/dev/correlations.png)

### Feature Importances Regression
![](https://github.com/bszek213/cbb_machine_learning/blob/dev/feature_importance_random_forest.png)
## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.
