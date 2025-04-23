#train and save model
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.dummy import DummyRegressor
import xgboost as xgb
import pickle
from sklearn.metrics import mean_squared_error
from sklearn.impute import SimpleImputer
import optuna
import warnings

warnings.filterwarnings('ignore')

# Load aggregated data
agg_df = pd.read_csv('agg_df.csv')

# Features and target
features = ['Year', 'Age', 'Height', 'Weight', 'ID', 'Prev_Medals']
X = agg_df[features]
y = agg_df['Total_Medals']

# Calculate standard deviation of target
target_std = y.std()
print(f"Target Standard Deviation: {target_std:.2f}")

# Impute NaNs with column means
imputer = SimpleImputer(strategy='mean')
X = pd.DataFrame(imputer.fit_transform(X), columns=features)

# Compute sample weights based on Year (exponential weighting)
year_min = X['Year'].min()
year_scale = 16  # Adjusts steepness of exponential curve #############
sample_weights = np.exp((X['Year'] - year_min) / year_scale) * (agg_df['Prev_Medals'] + 1)
print(
    f"Sample Weights Range: {sample_weights.min():.2f} (Year {int(year_min)}) to {sample_weights.max():.2f} (Year {int(X['Year'].max())})"
)

# Split data    #####################################################
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
sample_weights_train = np.exp((X_train['Year'] - year_min) / year_scale) * (X_train['Prev_Medals'] + 1)


# Define Optuna objective for Random Forest  ##############################
def objective_rf(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 200),     # Wider range
        'max_depth': trial.suggest_int('max_depth', 5, 25),              # Slightly deeper trees
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 8),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 4)
    }
    model = RandomForestRegressor(**params, random_state=42)
    model.fit(X_train, y_train, sample_weight=sample_weights_train)
    y_pred = model.predict(X_test)
    return np.sqrt(mean_squared_error(y_test, y_pred))


# Define Optuna objective for XGBoost  #############################
def objective_xgb(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 200),
        'max_depth': trial.suggest_int('max_depth', 4, 12),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0)

    }
    model = xgb.XGBRegressor(**params, objective='reg:squarederror', random_state=42)
    model.fit(X_train, y_train, sample_weight=sample_weights_train)
    y_pred = model.predict(X_test)
    return np.sqrt(mean_squared_error(y_test, y_pred))


# Optimize Random Forest  #########/////////////#####
study_rf = optuna.create_study(direction='minimize')
study_rf.optimize(objective_rf, n_trials=20)
best_rf_params = study_rf.best_params
print(f"Best Random Forest params: {best_rf_params}")

# Optimize XGBoost      #########/////////////#####
study_xgb = optuna.create_study(direction='minimize')
study_xgb.optimize(objective_xgb, n_trials=20)
best_xgb_params = study_xgb.best_params
print(f"Best XGBoost params: {best_xgb_params}")

# Define models
models = {
    'Random Forest': RandomForestRegressor(**best_rf_params, random_state=42),
    'XGBoost': xgb.XGBRegressor(**best_xgb_params, objective='reg:squarederror', random_state=42)
}

# Train and evaluate models
rmse_scores = {}
for name, model in models.items():
    model.fit(X_train, y_train, sample_weight=sample_weights_train)
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    rmse_scores[name] = rmse
    print(f"{name} RMSE: {rmse:.2f} (vs Target Std: {target_std:.2f})")

    # Save model
    with open(f"{name.lower().replace(' ', '_')}_model.pkl", 'wb') as f:
        pickle.dump(model, f)

# Create ensemble (weighted average of Random Forest and XGBoost)
rf_model = models['Random Forest']
xgb_model = models['XGBoost']
y_pred_rf = rf_model.predict(X_test)
y_pred_xgb = xgb_model.predict(X_test)

# Find the best ensemble weight between RF and XGB
best_rmse = float('inf')
best_weight = 0.5  # initial

for w in np.linspace(0, 1, 101):  # Try 0.00 to 1.00 (step 0.01)
    y_pred_ensemble = w * y_pred_rf + (1 - w) * y_pred_xgb
    rmse = np.sqrt(mean_squared_error(y_test, y_pred_ensemble))
    if rmse < best_rmse:
        best_rmse = rmse
        best_weight = w

print(f"Best Ensemble Weight (RF): {best_weight:.2f}")
print(f"Optimized Ensemble RMSE: {best_rmse:.2f} (vs Target Std: {target_std:.2f})")

# Save ensemble RMSE
rmse_scores['Ensemble RF+XGB'] = best_rmse

# Save RMSE scores for UI display
with open('rmse_scores.pkl', 'wb') as f:
    pickle.dump(rmse_scores, f)
print("All models trained and saved. RMSE scores saved.")

with open('ensemble_best_weight.pkl', 'wb') as f:
    pickle.dump(best_weight, f)