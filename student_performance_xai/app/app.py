import shap
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

from student_performance_xai.src.data.utils import load_data
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
st.header("Student Performance Overview")

fig, ax = plt.subplots()
sns.histplot(df["G3"], bins=20, kde=True, ax=ax)
ax.set_xlabel("Final Grade (G3)")
ax.set_ylabel("Number of Students")

st.pyplot(fig)

####################################################
st.subheader("Effect of Study Time on Final Grade")

fig, ax = plt.subplots()
sns.boxplot(x="studytime", y="G3", data=df, ax=ax)
ax.set_xlabel("Weekly Study Time Category")
ax.set_ylabel("Final Grade (G3)")

st.pyplot(fig)

#####################################################
st.subheader("Final Grade by Sex")

fig, ax = plt.subplots()
sns.boxplot(x="sex", y="G3", data=df, ax=ax)
ax.set_xlabel("Sex")
ax.set_ylabel("Final Grade (G3)")

st.pyplot(fig)

#####################################################
st.subheader("Interaction: Study Time and Internet Access")

fig, ax = plt.subplots()
sns.boxplot(x="studytime", y="G3", hue="internet", data=df, ax=ax)
ax.set_xlabel("Study Time")
ax.set_ylabel("Final Grade (G3)")
ax.legend(title="Internet Access")

st.pyplot(fig)

#####################################################
st.sidebar.header("Filters")
sex_filter = st.sidebar.selectbox("Select Sex", df["sex"].unique())

filtered_df = df[df["sex"] == sex_filter]

st.subheader(f"Grade Distribution for Sex = {sex_filter}")

fig, ax = plt.subplots()
sns.histplot(filtered_df["G3"], bins=15, ax=ax)
st.pyplot(fig)


#####################################################
st.header("Model Training and Explanation")

st.cache_resource
def train(df):
    X, y = preprocess(df)
    return train_model(X, y)

model, X_train, X_test, y_train, y_test = train(df)

st.subheader("Dataset Preview")
st.dataframe(df.head())

# ---- SHAP ----
st.subheader("Global Feature Importance (SHAP)")

explainer, shap_values = get_shap_explainer(model, X_train)

fig, ax = plt.subplots()
shap.summary_plot(shap_values, X_train, show=False)
st.pyplot(fig)
