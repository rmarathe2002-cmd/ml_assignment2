import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import (
    StandardScaler,
    OneHotEncoder,
    LabelEncoder
)
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import LinearSVC

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    roc_curve
)

# -----------------------------------
# PAGE CONFIGURATION
# -----------------------------------

st.set_page_config(
    page_title="ML Classification Dashboard",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 Machine Learning Classification Dashboard")

st.write(
    "Upload a dataset, select a classification algorithm "
    "and evaluate its performance."
)

# -----------------------------------
# FILE UPLOAD
# -----------------------------------

uploaded_file = st.file_uploader(
    "Upload your dataset",
    type=["csv", "xlsx"]
)

if uploaded_file is not None:

    try:
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)

        st.success("Dataset uploaded successfully!")

        # -----------------------------------
        # DATASET INFORMATION
        # -----------------------------------

        st.subheader("Dataset Preview")

        st.dataframe(df.head(10), use_container_width=True)

        col1, col2, col3 = st.columns(3)

        col1.metric("Total Instances", df.shape[0])
        col2.metric("Total Columns", df.shape[1])
        col3.metric("Missing Values", int(df.isnull().sum().sum()))

        st.subheader("Dataset Information")

        st.write("Number of rows:", df.shape[0])
        st.write("Number of columns:", df.shape[1])

        # -----------------------------------
        # TARGET COLUMN SELECTION
        # -----------------------------------

        st.subheader("Select Target Variable")

        target_column = st.selectbox(
            "Choose the target column",
            options=df.columns
        )

        # -----------------------------------
        # ALGORITHM SELECTION
        # -----------------------------------

        st.subheader("Select Classification Algorithm")

        algorithm = st.selectbox(
            "Choose an algorithm",
            [
                "Logistic Regression",
                "Decision Tree Classifier",
                "K-Nearest Neighbor Classifier",
                "Gaussian Naive Bayes",
                "Support Vector Machine"
            ]
        )

        # -----------------------------------
        # TEST SIZE
        # -----------------------------------

        test_size = st.slider(
            "Testing Data Percentage",
            min_value=10,
            max_value=40,
            value=20,
            step=5
        )

        # -----------------------------------
        # MODEL TRAINING
        # -----------------------------------

        if st.button("Train Model", type="primary"):

            try:

                # Remove rows with missing target values
                data = df.dropna(subset=[target_column]).copy()

                X = data.drop(columns=[target_column])
                y = data[target_column]

                # Remove columns with no values
                X = X.dropna(axis=1, how="all")

                # Check dataset requirements
                if len(X) < 20:
                    st.error("Dataset has too few instances.")
                    st.stop()

                if X.shape[1] < 1:
                    st.error("No independent features found.")
                    st.stop()

                if y.nunique() != 2:
                    st.error(
                        "This application currently supports "
                        "binary classification only. "
                        "Please select a target with exactly "
                        "two classes."
                    )
                    st.stop()

                # Encode target variable
                label_encoder = LabelEncoder()
                y = label_encoder.fit_transform(y.astype(str))

                # Display class mapping
                st.write(
                    "Class mapping:",
                    {
                        str(label): int(index)
                        for index, label in enumerate(
                            label_encoder.classes_
                        )
                    }
                )

                # Identify numerical and categorical columns
                numerical_columns = X.select_dtypes(
                    include=["number"]
                ).columns.tolist()

                categorical_columns = X.select_dtypes(
                    exclude=["number"]
                ).columns.tolist()

                # Preprocessing for numerical features
                numerical_pipeline = Pipeline([
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler())
                ])

                # Preprocessing for categorical features
                categorical_pipeline = Pipeline([
                    ("imputer", SimpleImputer(
                        strategy="most_frequent"
                    )),
                    ("encoder", OneHotEncoder(
                        handle_unknown="ignore",
                        sparse_output=False
                    ))
                ])

                # Combine preprocessing
                transformers = []

                if numerical_columns:
                    transformers.append((
                        "numerical",
                        numerical_pipeline,
                        numerical_columns
                    ))

                if categorical_columns:
                    transformers.append((
                        "categorical",
                        categorical_pipeline,
                        categorical_columns
                    ))

                preprocessor = ColumnTransformer(
                    transformers=transformers
                )

                # -----------------------------------
                # TRAIN TEST SPLIT
                # -----------------------------------

                X_train, X_test, y_train, y_test = (
                    train_test_split(
                        X,
                        y,
                        test_size=test_size / 100,
                        random_state=42,
                        stratify=y
                    )
                )

                # -----------------------------------
                # SELECT MODEL
                # -----------------------------------

                if algorithm == "Logistic Regression":

                    classifier = LogisticRegression(
                        max_iter=2000,
                        class_weight="balanced"
                    )

                elif algorithm == "Decision Tree Classifier":

                    classifier = DecisionTreeClassifier(
                        random_state=42,
                        class_weight="balanced"
                    )

                elif algorithm == "K-Nearest Neighbor Classifier":

                    classifier = KNeighborsClassifier(
                        n_neighbors=5
                    )

                elif algorithm == "Gaussian Naive Bayes":

                    classifier = GaussianNB()

                elif algorithm == "Support Vector Machine":

                    classifier = LinearSVC(
                        class_weight="balanced",
                        random_state=42,
                        max_iter=5000
                    )

                # -----------------------------------
                # CREATE COMPLETE PIPELINE
                # -----------------------------------

                model = Pipeline([
                    ("preprocessing", preprocessor),
                    ("classifier", classifier)
                ])

                # Train model
                with st.spinner("Training model..."):

                    model.fit(X_train, y_train)

                    # Predictions
                    y_pred = model.predict(X_test)

                    # -----------------------------------
                    # PREDICTION SCORES FOR AUC
                    # -----------------------------------

                    if hasattr(model, "predict_proba"):

                        y_score = model.predict_proba(
                            X_test
                        )[:, 1]

                    else:

                        y_score = model.decision_function(
                            X_test
                        )

                    # -----------------------------------
                    # EVALUATION METRICS
                    # -----------------------------------

                    accuracy = accuracy_score(
                        y_test, y_pred
                    )

                    auc = roc_auc_score(
                        y_test, y_score
                    )

                    precision = precision_score(
                        y_test,
                        y_pred,
                        zero_division=0
                    )

                    recall = recall_score(
                        y_test,
                        y_pred,
                        zero_division=0
                    )

                    f1 = f1_score(
                        y_test,
                        y_pred,
                        zero_division=0
                    )

                    mcc = matthews_corrcoef(
                        y_test, y_pred
                    )

                st.success("Model trained successfully!")

                # -----------------------------------
                # DISPLAY RESULTS
                # -----------------------------------

                st.subheader("Model Evaluation Results")

                st.write("Selected Algorithm:", algorithm)

                c1, c2, c3 = st.columns(3)

                c1.metric(
                    "Accuracy",
                    f"{accuracy:.4f}"
                )

                c2.metric(
                    "AUC Score",
                    f"{auc:.4f}"
                )

                c3.metric(
                    "Precision",
                    f"{precision:.4f}"
                )

                c4, c5, c6 = st.columns(3)

                c4.metric(
                    "Recall",
                    f"{recall:.4f}"
                )

                c5.metric(
                    "F1 Score",
                    f"{f1:.4f}"
                )

                c6.metric(
                    "MCC Score",
                    f"{mcc:.4f}"
                )

                # -----------------------------------
                # CONFUSION MATRIX
                # -----------------------------------

                st.subheader("Confusion Matrix")

                cm = confusion_matrix(
                    y_test,
                    y_pred,
                    labels=[0, 1]
                )

                fig, ax = plt.subplots(figsize=(6, 4))

                sns.heatmap(
                    cm,
                    annot=True,
                    fmt="d",
                    cmap="Blues",
                    xticklabels=label_encoder.classes_,
                    yticklabels=label_encoder.classes_,
                    ax=ax
                )

                ax.set_xlabel("Predicted")
                ax.set_ylabel("Actual")
                ax.set_title("Confusion Matrix")

                st.pyplot(fig)
                plt.close(fig)

                # -----------------------------------
                # ROC CURVE
                # -----------------------------------

                st.subheader("ROC Curve")

                fpr, tpr, thresholds = roc_curve(
                    y_test,
                    y_score
                )

                fig, ax = plt.subplots(figsize=(6, 4))

                ax.plot(
                    fpr,
                    tpr,
                    label=f"AUC = {auc:.4f}"
                )

                ax.plot(
                    [0, 1],
                    [0, 1],
                    linestyle="--"
                )

                ax.set_xlabel("False Positive Rate")
                ax.set_ylabel("True Positive Rate")
                ax.set_title("ROC Curve")
                ax.legend()

                st.pyplot(fig)
                plt.close(fig)

                # -----------------------------------
                # METRICS TABLE
                # -----------------------------------

                st.subheader("Evaluation Summary")

                results = pd.DataFrame({
                    "Evaluation Metric": [
                        "Accuracy",
                        "AUC Score",
                        "Precision",
                        "Recall",
                        "F1 Score",
                        "MCC Score"
                    ],
                    "Score": [
                        accuracy,
                        auc,
                        precision,
                        recall,
                        f1,
                        mcc
                    ]
                })

                st.dataframe(
                    results,
                    use_container_width=True,
                    hide_index=True
                )

                # Download metrics
                csv = results.to_csv(
                    index=False
                ).encode("utf-8")

                st.download_button(
                    "Download Evaluation Results",
                    data=csv,
                    file_name="model_evaluation.csv",
                    mime="text/csv"
                )

            except Exception as e:

                st.error(
                    f"Error while training model: {str(e)}"
                )

    except Exception as e:

        st.error(
            f"Unable to read dataset: {str(e)}"
        )

else:

    st.info(
        "Please upload a CSV or Excel dataset to get started."
    )