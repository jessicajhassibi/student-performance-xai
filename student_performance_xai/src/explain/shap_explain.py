import numpy as np
import pandas as pd
import shap

def get_shap_explainer(model, X_train):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_train)
    return explainer, shap_values

def get_shap_importance(shap_values, feature_names):
    mean_abs_shap = get_mean_abs_shap(shap_values)
    shap_importance = pd.Series(mean_abs_shap, index=feature_names).sort_values(ascending=True)
    return shap_importance

def get_mean_abs_shap(shap_values):
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    return mean_abs_shap