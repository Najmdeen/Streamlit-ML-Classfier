import streamlit as st
import pandas as pd
import numpy as np
from sklearn import datasets
import plotly as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


st.title("Classifier Data type Sample")

st.write("Explore Different model for Classifier Dataset")

dataset = ("Wine", "Iris", "Breast Cancer")
dataset_name = st.sidebar.selectbox("Select a Dataset", dataset)

models = ("Random Forest", "SVM", "KNN", "Logistic Regression")
model_name = st.sidebar.selectbox("Select a Model", models)


def get_dataset(dataset_name):
    if dataset_name == "Wine":
        df = datasets.load_wine()
    elif dataset_name == "Iris":
        df = datasets.load_iris()
    else:
        df = datasets.load_breast_cancer()

    return df.data, df.target


feature, label = get_dataset(dataset_name)
st.write(f"Shape of Dataset: {feature.shape}")
st.write(f"Number of Classes: {len(set(label))}")


def add_param(model_name):
    params = {}
    if model_name == "Random Forest":
        n_estimators = st.sidebar.slider("n_estimators", 10, 100, step=10)
        max_depth = st.sidebar.slider("max_depth", 1, 5)
        # criterion = st.sidebar.("criterion", "gini", "entropy", "log_loss")
        params["n_estimators"] = n_estimators
        params["max_depth"] = max_depth

    elif model_name == "KNN":
        k = st.sidebar.slider("k", 1, 10, 5)
        params["k"] = k

    elif model_name == "SVM":
        c = st.sidebar.slider("C", 0.1, 10.0)
        params["C"] = c

    elif model_name == "Logistic Regression":
        c = st.sidebar.slider("C", 0.1, 10.0)
        params["C"] = c
    return params


param = add_param(model_name)


def build_model(model_name, param):
    if model_name == "Random Forest":
        clf = RandomForestClassifier(
            n_estimators=param["n_estimators"],
            max_depth=param["max_depth"],
            random_state=42,
        )

    elif model_name == "KNN":
        clf = KNeighborsClassifier(n_neighbors=param["k"])

    elif model_name == "SVM":
        clf = SVC(C=param["C"])

    else:
        clf = LogisticRegression(C=param["C"])
    return clf


model = build_model(model_name, param)

# Classifier dataset
X_train, X_test, y_train, y_test = train_test_split(
    feature, label, test_size=0.2, random_state=42
)

st.write(f"X_train, y_train, X_test, y_test: ")
st.write(X_train.shape, y_train.shape, X_test.shape, y_test.shape)

model.fit(X_train, y_train)

# Predict dataset
pred = model.predict(X_test)

accuracy = accuracy_score(y_test, pred)
st.write(f"Accurate Score: {accuracy}")
