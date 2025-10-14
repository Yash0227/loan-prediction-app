import os
import streamlit as st
import pandas as pd
import numpy as np
import joblib

# -------------------------------
# Load models & data
# -------------------------------
@st.cache_data
def load_all():
    # model1 = joblib.load("best_model_model1.pkl")
    # model2 = joblib.load("model2_best_regressor.pkl")
    # features_model2 = joblib.load("model2_feature_names.pkl")

    # df_raw = pd.read_csv("original_dataset.csv")
    # df_processed = pd.read_csv("df_final_features_v2.csv")
    # df_model2 = pd.read_csv("df_model2_features_with_id.csv")  # Corrected file
    BASE_PATH = os.path.dirname(__file__)  # notebooks/

    model1 = joblib.load(os.path.join(BASE_PATH, "best_model_model1.pkl"))
    model2 = joblib.load(os.path.join(BASE_PATH, "model2_best_regressor.pkl"))
    features_model2 = joblib.load(os.path.join(BASE_PATH, "model2_feature_names.pkl"))

    df_raw = pd.read_csv(os.path.join(BASE_PATH, "original_dataset.csv"))
    df_processed = pd.read_csv(os.path.join(BASE_PATH, "df_final_features_v2.csv"))
    df_model2 = pd.read_csv(os.path.join(BASE_PATH, "df_model2_features_with_id.csv"))

    if 'loan_id' not in df_processed.columns:
        df_processed = pd.concat([
            df_raw[['loan_id']].reset_index(drop=True),
            df_processed.reset_index(drop=True)
        ], axis=1)
    else:
        df_processed = pd.merge(
            df_raw[['loan_id']], df_processed,
            left_index=True, right_index=True, how='inner'
        )

    return model1, model2, features_model2, df_raw, df_processed, df_model2

model1, model2, features_model2, df_raw, df_processed, df_model2 = load_all()

# -------------------------------
# Streamlit UI Setup
# -------------------------------
st.set_page_config(page_title="Loan Prediction", layout="wide")

st.markdown("""
<style>
.section-title {
    color: white !important;
}
</style>
""", unsafe_allow_html=True)


st.markdown("<div class='section-title'>🏦 Loan Prediction Web App</div>", unsafe_allow_html=True)
st.write("Enter a Loan ID to get approval prediction and estimated loan amount if approved.")

# -------------------------------
# Responsive layout toggle
# -------------------------------
layout_mode = st.radio("Choose layout:", options=["Stacked (Vertical)", "Side by Side"], index=0)

if layout_mode == "Side by Side":
    col1, col2 = st.columns([1.5, 1])

    with col1:
        st.markdown("<div class='big-font'>📋 Original Dataset</div>", unsafe_allow_html=True)
        st.dataframe(df_raw, height=600, use_container_width=True)

    with col2:
        loan_id_input = st.text_input("🔢 Enter Loan ID", "")

        if loan_id_input:
            if not loan_id_input.isdigit():
                st.error("⚠️ Please enter a valid numeric Loan ID.")
            else:
                loan_id = int(loan_id_input)
                row1 = df_processed[df_processed['loan_id'] == loan_id]

                if row1.empty:
                    st.error(f"❌ Loan ID {loan_id} not found.")
                else:
                    X1 = row1[model1.feature_names_in_].apply(pd.to_numeric, errors='coerce').fillna(0)
                    pred_status = model1.predict(X1)[0]

                    try:
                        prob = model1.predict_proba(X1)[0][0]
                    except:
                        prob = None

                    status = {0: "✅ Approved", 1: "❌ Rejected"}.get(pred_status, "Unknown")

                    st.markdown("<hr>", unsafe_allow_html=True)
                    st.markdown("<div class='section-title'>🔍 Prediction Result</div>", unsafe_allow_html=True)
                    st.markdown(f"<div class='big-font'>Loan ID: <b>{loan_id}</b></div>", unsafe_allow_html=True)
                    st.markdown(f"<div class='big-font'>Status: <b>{status}</b></div>", unsafe_allow_html=True)

                    if prob is not None:
                        st.markdown(f"<div class='big-font'>Approval Probability: <b>{prob:.2%}</b></div>", unsafe_allow_html=True)

                    if pred_status == 0:
                        row2 = df_model2[df_model2['loan_id'] == loan_id]

                        if row2.empty:
                            st.warning("⚠️ Loan ID not found in Model 2 dataset.")
                        elif row2.shape[0] > 1:
                            st.warning(f"⚠️ Multiple matches found for Loan ID {loan_id}.")
                        else:
                            X2 = row2[features_model2].apply(pd.to_numeric, errors='coerce').fillna(0)
                            loan_log = model2.predict(X2)[0]
                            loan_amount = np.expm1(loan_log)

                            st.markdown(f"<div class='big-font'>💰 Predicted Loan Amount: <b>₹{loan_amount:,.2f}</b></div>", unsafe_allow_html=True)
                    else:
                        st.info("Loan amount prediction is not applicable (loan was rejected).")

else:
    # Stacked layout: Dataset then input/prediction
    st.markdown("<div class='big-font'>📋 Original Dataset</div>", unsafe_allow_html=True)
    st.dataframe(df_raw, height=400, use_container_width=True)

    st.markdown("<hr>", unsafe_allow_html=True)

    loan_id_input = st.text_input("🔢 Enter Loan ID", "")

    if loan_id_input:
        if not loan_id_input.isdigit():
            st.error("⚠️ Please enter a valid numeric Loan ID.")
        else:
            loan_id = int(loan_id_input)
            row1 = df_processed[df_processed['loan_id'] == loan_id]

            if row1.empty:
                st.error(f"❌ Loan ID {loan_id} not found.")
            else:
                X1 = row1[model1.feature_names_in_].apply(pd.to_numeric, errors='coerce').fillna(0)
                pred_status = model1.predict(X1)[0]

                try:
                    prob = model1.predict_proba(X1)[0][0]
                except:
                    prob = None

                status = {0: "✅ Approved", 1: "❌ Rejected"}.get(pred_status, "Unknown")

                st.markdown("<hr>", unsafe_allow_html=True)
                st.markdown("<div class='section-title'>🔍 Prediction Result</div>", unsafe_allow_html=True)
                st.markdown(f"<div class='big-font'>Loan ID: <b>{loan_id}</b></div>", unsafe_allow_html=True)
                st.markdown(f"<div class='big-font'>Status: <b>{status}</b></div>", unsafe_allow_html=True)

                if prob is not None:
                    st.markdown(f"<div class='big-font'>Approval Probability: <b>{prob:.2%}</b></div>", unsafe_allow_html=True)

                if pred_status == 0:
                    row2 = df_model2[df_model2['loan_id'] == loan_id]

                    if row2.empty:
                        st.warning("⚠️ Loan ID not found in Model 2 dataset.")
                    elif row2.shape[0] > 1:
                        st.warning(f"⚠️ Multiple matches found for Loan ID {loan_id}.")
                    else:
                        X2 = row2[features_model2].apply(pd.to_numeric, errors='coerce').fillna(0)
                        loan_log = model2.predict(X2)[0]
                        loan_amount = np.expm1(loan_log)

                        st.markdown(f"<div class='big-font'>💰 Predicted Loan Amount: <b>₹{loan_amount:,.2f}</b></div>", unsafe_allow_html=True)
                else:
                    st.info("Loan amount prediction is not applicable (loan was rejected).")
