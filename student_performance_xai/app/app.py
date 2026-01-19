import shap
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

from student_performance_xai.src.data.utils import load_data
from student_performance_xai.src.data.eda import univariate_analysis, bivariate_analysis
from student_performance_xai.src.data.preprocess import preprocess
from student_performance_xai.src.models.random_forest import train_model
from student_performance_xai.src.explain.shap_explain import get_shap_explainer


st.set_page_config(page_title="Student Performance XAI", layout="wide")
st.title("Student Performance XAI")

@st.cache_data # cache the data loading for better performance
def get_data():
    return load_data()

df = load_data()

###################################################

st.subheader("Dataset Preview")
st.dataframe(df.head())

###################################################
st.header("Univariate Analysis of selected categorical features")
figs = univariate_analysis(df, cols=["sex", "age", "studytime"])
for fig in figs:
    st.pyplot(fig)
    plt.close(fig)

###################################################
st.header("Bivariate Analysis")

st.subheader("Distribution of Final Grade by Study Time")
fig, ax = bivariate_analysis(df, "studytime", "G3")
ax.set_xlabel("Weekly Study Time (1 = low, 4 = high)")
ax.set_ylabel("Final Grade (G3)")
st.pyplot(fig)
plt.close(fig)

#####################################################
st.subheader("Distribution of Final Grade by sex")
fig, ax = bivariate_analysis(df, "sex", "G3")
ax.set_xlabel("Sex")
ax.set_ylabel("Final Grade (G3)")
st.pyplot(fig)
plt.close(fig)

#####################################################
st.subheader("Distribution of Final Grade by absences")
fig, ax = bivariate_analysis(df, "absences", "G3")
ax.set_xlabel("Absences")
ax.set_ylabel("Final Grade (G3)")
st.pyplot(fig)
plt.close(fig)

#####################################################
st.header("Model Training and Explanation")

st.cache_resource
def train(df):
    X, y = preprocess(df)
    return train_model(X, y)

model, X_train, X_test, y_train, y_test = train(df)

# ---- SHAP ----
explainer, shap_values = get_shap_explainer(model, X_train)

st.subheader("Global Feature Importance (SHAP)")

shap.summary_plot(shap_values, X_train, show=False)
fig = plt.gcf() # get current figure that SHAP created
if fig.axes:
    ax.set_title("Global Feature Importance (SHAP)")
st.pyplot(fig)
plt.close(fig)

######################################################
st.subheader("SHAP Dependence Plot for Weekly Study Time")

shap.dependence_plot(
    "studytime",
    shap_values,
    X_train,
    interaction_index=None,  # main effect only
    show=False
)
fig = plt.gcf()
if fig.axes:
    ax.set_title("SHAP Dependence Plot for Weekly Study Time")
st.pyplot(fig)
plt.close(fig)
