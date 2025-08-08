import streamlit as st
import pandas as pd
import numpy as np
import pickle
import io

st.set_page_config(page_title="Income Model UI", layout="centered")
st.title("Income Prediction — Model UI")
st.write("Upload a trained sklearn model (pickle) and a sample CSV (optional). The app will build input fields from the sample or from a user-provided list of features.")

# Sidebar: upload model and optional scaler
st.sidebar.header("Model & Data Upload")
model_file = st.sidebar.file_uploader("Upload model (.pkl)", type=["pkl"], help="Pickle file containing a trained sklearn model")
scaler_file = st.sidebar.file_uploader("Optional: upload scaler (.pkl)", type=["pkl"], help="If you used a scaler during training, upload it to transform inputs the same way")
sample_csv = st.sidebar.file_uploader("Optional: upload a sample CSV to infer inputs", type=["csv"])

# Option to manually provide feature names
manual_features = st.sidebar.text_area("Or paste feature names (comma separated)", value="", height=80)

# Helper to load pickle
@st.cache_data
def load_pickle(b):
    return pickle.load(b)

model = None
scaler = None
if model_file is not None:
    try:
        model = load_pickle(model_file)
        st.sidebar.success("Model loaded")
    except Exception as e:
        st.sidebar.error(f"Failed to load model: {e}")

if scaler_file is not None:
    try:
        scaler = load_pickle(scaler_file)
        st.sidebar.success("Scaler loaded")
    except Exception as e:
        st.sidebar.error(f"Failed to load scaler: {e}")

# Determine features and dtypes
features = []
dtypes = {}
sample_row = None
if sample_csv is not None:
    try:
        df_sample = pd.read_csv(sample_csv)
        if df_sample.shape[0] == 0:
            st.sidebar.warning("Sample CSV empty — please upload a CSV with at least one row.")
        else:
            # use first row as a template
            sample_row = df_sample.iloc[0]
            features = list(df_sample.columns)
            for c in features:
                dtypes[c] = str(df_sample[c].dtype)
            st.sidebar.success(f"Detected {len(features)} features from CSV")
    except Exception as e:
        st.sidebar.error(f"Failed to read CSV: {e}")

# If manual features provided and no CSV, parse them
if not features and manual_features.strip():
    parts = [p.strip() for p in manual_features.split(",") if p.strip()]
    features = parts
    for p in parts:
        dtypes[p] = "float"  # default to numeric; user can input strings
    st.sidebar.info(f"Using {len(features)} manually provided features")

if not features:
    st.info("No features defined yet. Either upload a sample CSV or paste comma-separated feature names in the sidebar.")

# Main input form
input_values = {}
if features:
    st.header("Input values")
    with st.form(key='input_form'):
        cols = st.columns(2)
        for i, feat in enumerate(features):
            col = cols[i % 2]
            # choose input widget based on dtype or example value
            example = None
            if sample_row is not None:
                example = sample_row.get(feat, None)
            # simple heuristic to choose widget type
            if example is not None and (pd.api.types.is_integer_dtype(type(example)) or pd.api.types.is_float_dtype(type(example))):
                val = col.number_input(feat, value=float(example) if not pd.isna(example) else 0.0, format="%f")
            else:
                # fallback to text input
                default = str(example) if example is not None and not pd.isna(example) else ""
                val = col.text_input(feat, value=default)
            input_values[feat] = val
        submit = st.form_submit_button("Predict")

    if submit:
        if model is None:
            st.error("Please upload a model (.pkl) in the sidebar before predicting.")
        else:
            # Build dataframe for model
            X = pd.DataFrame([input_values])
            # Try to convert numeric-like columns
            for c in X.columns:
                # attempt numeric conversion
                try:
                    X[c] = pd.to_numeric(X[c])
                except Exception:
                    X[c] = X[c].astype(object)
            # Apply scaler if provided
            try:
                if scaler is not None:
                    # scaler may expect array shape (n_features,)
                    X_numeric = X.select_dtypes(include=[np.number])
                    if X_numeric.shape[1] > 0:
                        X[X_numeric.columns] = scaler.transform(X_numeric)
                # Predict
                preds = model.predict(X)
                # If classifier with predict_proba
                proba = None
                if hasattr(model, "predict_proba"):
                    proba = model.predict_proba(X)
                st.success("Prediction complete")
                st.write("**Input used:**")
                st.table(X.T)
                st.write("**Predicted:**")
                st.write(preds)
                if proba is not None:
                    st.write("**Probabilities:**")
                    pr_df = pd.DataFrame(proba, columns=[f"class_{i}" for i in range(proba.shape[1])])
                    st.table(pr_df)
            except Exception as e:
                st.error(f"Error during prediction: {e}")

# Extra utilities
st.sidebar.markdown("---")
st.sidebar.header("Utilities")
if st.sidebar.button("Download sample CSV template"):
    # provide a minimal CSV template with feature columns (if available)
    if features:
        tmp = pd.DataFrame(columns=features)
    else:
        tmp = pd.DataFrame({"feature1":[], "feature2":[]})
    csv = tmp.to_csv(index=False).encode('utf-8')
    st.sidebar.download_button("Download CSV", data=csv, file_name="sample_template.csv", mime="text/csv")

st.markdown("---")
st.caption("Built with Streamlit — upload your model.pkl and a sample CSV to create a UI automatically.")
