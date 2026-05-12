import streamlit as st
import requests

st.set_page_config(
    page_title="Credit Risk Prediction",
    page_icon="💳",
    layout="wide"
)

st.title("💳 Credit Lending Risk Analysis")
st.markdown("### Customer Credit Risk Prediction System")
st.write("Fill in the applicant credit details below to predict the approval category.")

FASTAPI_URL = "http://127.0.0.1:8000/predict"

with st.container():

    col1, col2 = st.columns(2)

    with col1:

        st.subheader("📊 Credit Behaviour")

        enq_L3m = st.number_input(
            "Enquiries in Last 3 Months",
            min_value=0,
            max_value=100,
            value=2,
            help="Total number of credit enquiries made in the last 3 months."
        )

        Age_Oldest_TL = st.number_input(
            "Age of Oldest Tradeline (Months)",
            min_value=0,
            max_value=600,
            value=120,
            help="Age of the oldest credit account or tradeline in months."
        )

        num_times_delinquent = st.number_input(
            "Number of Delinquencies",
            min_value=0,
            max_value=100,
            value=1,
            help="Total number of delinquent payment events."
        )

        pct_PL_enq_L6m_of_ever = st.slider(
            "Personal Loan Enquiry Percentage (Last 6 Months)",
            min_value=0.0,
            max_value=100.0,
            value=25.0,
            help="Percentage of personal loan enquiries in the last 6 months out of all enquiries ever made."
        )

        num_std_6mts = st.number_input(
            "Standard Accounts in Last 6 Months",
            min_value=0,
            max_value=100,
            value=3,
            help="Number of standard credit accounts in the last 6 months."
        )

        num_std = st.number_input(
            "Total Standard Accounts",
            min_value=0,
            max_value=500,
            value=5,
            help="Total number of standard credit accounts."
        )

        num_deliq_12mts = st.number_input(
            "Delinquent Accounts in Last 12 Months",
            min_value=0,
            max_value=100,
            value=1,
            help="Number of delinquent accounts in the last 12 months."
        )

        max_recent_level_of_deliq = st.number_input(
            "Maximum Recent Delinquency Level",
            min_value=0,
            max_value=10,
            value=2,
            help="Maximum severity level of recent delinquency."
        )

        Age_Newest_TL = st.number_input(
            "Age of Newest Tradeline (Months)",
            min_value=0,
            max_value=240,
            value=12,
            help="Age of the most recently opened tradeline in months."
        )

    with col2:

        st.subheader("🏦 Credit Utilization")

        num_times_60p_dpd = st.number_input(
            "60+ Days Past Due Count",
            min_value=0,
            max_value=100,
            value=0,
            help="Number of times payments were delayed by more than 60 days."
        )

        tot_enq = st.number_input(
            "Total Credit Enquiries",
            min_value=0,
            max_value=500,
            value=10,
            help="Total number of credit enquiries."
        )

        PL_enq_L6m = st.number_input(
            "Personal Loan Enquiries in Last 6 Months",
            min_value=0,
            max_value=100,
            value=2,
            help="Number of personal loan enquiries made in the last 6 months."
        )

        time_since_recent_payment_missing = st.number_input(
            "Months Since Recent Missing Payment",
            min_value=0,
            max_value=240,
            value=8,
            help="Time since the most recent missed payment in months."
        )

        pct_tl_open_L12M = st.slider(
            "Tradelines Opened in Last 12 Months (%)",
            min_value=0.0,
            max_value=100.0,
            value=35.0,
            help="Percentage of tradelines opened in the last 12 months."
        )

        Tot_TL_closed_L12M = st.number_input(
            "Tradelines Closed in Last 12 Months",
            min_value=0,
            max_value=100,
            value=2,
            help="Total number of tradelines closed in the last 12 months."
        )

        Home_TL = st.number_input(
            "Home Loan Tradelines",
            min_value=0,
            max_value=50,
            value=1,
            help="Number of home loan tradelines."
        )

        pct_tl_open_L6M = st.slider(
            "Tradelines Opened in Last 6 Months (%)",
            min_value=0.0,
            max_value=100.0,
            value=20.0,
            help="Percentage of tradelines opened in the last 6 months."
        )

        pct_closed_tl = st.slider(
            "Closed Tradelines Percentage (%)",
            min_value=0.0,
            max_value=100.0,
            value=40.0,
            help="Percentage of closed tradelines out of total tradelines."
        )

st.divider()

if st.button("🚀 Predict Credit Risk", use_container_width=True):

    payload = {
        "enq_L3m": enq_L3m,
        "Age_Oldest_TL": Age_Oldest_TL,
        "num_times_delinquent": num_times_delinquent,
        "pct_PL_enq_L6m_of_ever": pct_PL_enq_L6m_of_ever,
        "num_std_6mts": num_std_6mts,
        "num_std": num_std,
        "num_deliq_12mts": num_deliq_12mts,
        "max_recent_level_of_deliq": max_recent_level_of_deliq,
        "Age_Newest_TL": Age_Newest_TL,
        "num_times_60p_dpd": num_times_60p_dpd,
        "tot_enq": tot_enq,
        "PL_enq_L6m": PL_enq_L6m,
        "time_since_recent_payment_missing": time_since_recent_payment_missing,
        "pct_tl_open_L12M": pct_tl_open_L12M,
        "Tot_TL_closed_L12M": Tot_TL_closed_L12M,
        "Home_TL": Home_TL,
        "pct_tl_open_L6M": pct_tl_open_L6M,
        "pct_closed_tl": pct_closed_tl
    }

    try:

        response = requests.post(
            FASTAPI_URL,
            json=payload
        )

        if response.status_code == 200:

            prediction = response.json()["prediction"]

            st.success(f"✅ Predicted Credit Approval Category: {prediction}")

        else:

            st.error(f"❌ API Error: {response.text}")

    except Exception as e:

        st.error(f"❌ Connection Error: {str(e)}")