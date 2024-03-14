# College Basketball Game Predictions

Machine learning that predicts the outcome of any Division I college basketball game. Data are from 2010 - 2024 seasons. 
<!-- Currently the prediction accuracy is between 63-66% on future game outcomes.  -->
Data are from SportsReference.com

## Usage

```python
python cbb_classification.py tune or python cbb_classification.py notune
```

```bash
Removed features (>=0.9 correlation): ['fta', 'fta_per_fga_pct', 'fg3a_per_fga_pct', 'ts_pct', 'stl_pct', 'blk_pct', 'efg_pct', 'tov_pct', 'orb_pct', 'ft_rate']
dataset shape: (27973 samples, 55 features)

### Current prediction accuracies - XGBoost
# After 5 fold cross validation and pre-processing
Current XGBoost Classifier - best params:  {'colsample_bytree': 1.0, 'gamma': 0, 'learning_rate': 0.1, 'max_depth': 2, 'min_child_weight': 1, 'n_estimators': 200, 'reg_alpha': 0.01, 'reg_lambda': 0.01, 'scale_pos_weight': 1, 'subsample': 1.0}


#Classification - XGBoost
Confusion Matrix:[[1316   46]
                  [  31 1404]]
Model accuracy on test data: 0.9688952449052556

#Classificatino - DNN Keras
Final model test loss 0.07359004765748978 and accuracy 0.9760457873344421
```
### Correlation Matrix
![](https://github.com/bszek213/cbb_machine_learning/blob/dev/correlations.png)

<!-- ### Feature Importances Regression
![](https://github.com/bszek213/cbb_machine_learning/blob/dev/feature_importance_random_forest.png) -->
### Feature Importances Classification
XGBoost
![](https://github.com/bszek213/cbb_machine_learning/blob/dev/feature_importance_xgb_classifier.png)
Deep Neural Network
![](https://github.com/bszek213/cbb_machine_learning/blob/dev/SHAP_feature_importances.png)

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.
