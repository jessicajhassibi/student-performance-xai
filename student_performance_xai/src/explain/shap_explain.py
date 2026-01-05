import shap

def get_shap_explainer(model, X_train):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_train)
    return explainer, shap_values
