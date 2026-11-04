
import streamlit as st
import pandas as pd
import joblib

st.set_page_config(
    page_title="Marketing Intelligence",
    page_icon="📊",
    layout="wide"
)

st.markdown("""
<style>
.stApp {
    background-color: #F1F5F9;
}

h1, h2, h3 {
    color: #142D4E;
}

[data-testid="stSidebar"] {
    background-color: #142D4E;
}

[data-testid="stSidebar"] * {
    color: white;
}

[data-testid="stMetric"] {
    background-color: white;
    padding: 18px;
    border-radius: 12px;
    border: 1px solid #E2E8F0;
}

.stButton > button {
    background-color: #142D4E;
    color: white;
    border-radius: 8px;
    width: 100%;
    height: 45px;
}

</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_models():
    revenue_model = joblib.load("revenue_prediction_model.pkl")

    classification_model = joblib.load("profit_status_model.pkl")

    return revenue_model, classification_model


revenue_model, classification_model = load_models()

st.sidebar.title("📊 Campaign Intelligence")

page = st.sidebar.radio(
    "Navigation",
    ["Overview","Revenue Prediction","Profit/Loss Prediction"])


if page == "Overview":

    file_path=r"C:\Users\Dhanalakshmi\Desktop\GUVI PRACTICE\Projects\Project 3\Cleaned\marketing_cleaned.csv"

    df=pd.read_csv(file_path)
    st.header("📊 Cleaned Marketing Dataset Report")

    st.write("Explore the cleaned marketing campaign dataset.")

    # Dataset summary
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Total Records", df.shape[0])

    with col2:
        st.metric("Total Columns", df.shape[1])

    with col3:
        st.metric("Missing Values", df.isnull().sum().sum())

    st.subheader("Marketing campaign Dataset")

    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True
    )

    # Download option
    csv = df.to_csv(index=False).encode("utf-8")

    st.download_button(
        "📥 Download Cleaned Dataset",
        data=csv,
        file_name="marketing_cleaned.csv",
        mime="text/csv"
    )


    st.title("Marketing Campaign Dashboard")

    st.caption("Monitor and predict marketing campaign performance")

    col1, col2 = st.columns(2)

    with col1:
        st.metric("Prediction Model","Revenue","Random Forest Regressor")

    with col2:
        st.metric("Classification Model","Profit / Loss","Random Forest Classifier")

    st.subheader("Prediction Modules")

    col1, col2 = st.columns(2)

    with col1:
        st.info("💰 Revenue Prediction")
        st.write("Estimate campaign revenue using campaign details.")

    with col2:
        st.info("📈 Profit/Loss Classification")
        st.write(
            "Identify whether a campaign is predicted to "
            "generate profit or loss."
        )


elif page == "Revenue Prediction":

    st.title("Revenue Prediction")

    st.write("Enter campaign details to estimate revenue.")

    with st.form("revenue_form"):

        col1, col2 = st.columns(2)

        with col1:
            duration = st.number_input("Duration", value=1.0)

            impressions = st.number_input("Impressions", value=0.0)

            clicks = st.number_input("Clicks", value=0.0)

            leads = st.number_input("Leads", value=0.0)

        with col2:

            roi = st.number_input("ROI", value=0.0)

            conversions = st.number_input("Conversions", value=0.0)

            acquisition_cost = st.number_input("Acquisition Cost", value=0.0)

            engagement_score = st.number_input("Engagement Score", value=0.0)

        submit = st.form_submit_button("Predict Revenue")

    if submit:

        input_data = pd.DataFrame({
            "Duration": [duration],
            "Impressions": [impressions],
            "Clicks": [clicks],
            "Leads": [leads],
            "ROI":[roi],
            "Conversions": [conversions],
            "Acquisition_Cost": [acquisition_cost],
            "Engagement_Score": [engagement_score]
        })

        # Match exact training features
        expected = list(revenue_model.feature_names_in_)

        missing = set(expected) - set(input_data.columns)

        if missing:
            st.error(
                f"Missing model features: {list(missing)}"
            )
        else:
            input_data = input_data[expected]

            prediction = revenue_model.predict(input_data)[0]

            st.success("Revenue prediction completed!")

            st.metric(
                "Predicted Revenue",
                f"{prediction:,.2f}"
            )

            chart_data = pd.DataFrame({
                "Category": [
                    "Predicted Revenue",
                    "Acquisition Cost"
                ],
                "Amount": [
                    prediction,
                    acquisition_cost
                ]
            })

            st.subheader("Revenue Analysis")

            st.bar_chart(
                chart_data.set_index("Category"),
                color="#2563EB"
            )
elif page == "Profit/Loss Prediction":

    st.title("Profit / Loss Classification")

    st.write(
        "Enter campaign details to predict profitability."
    )

    with st.form("classification_form"):

        col1, col2 = st.columns(2)

        with col1:
            duration = st.number_input(
                "Duration", value=1.0,
                key="class_duration"
            )

            impressions = st.number_input(
                "Impressions", value=0.0,
                key="class_impressions"
            )

            clicks = st.number_input(
                "Clicks", value=0.0,
                key="class_clicks"
            )

            leads = st.number_input(
                "Leads", value=0.0,
                key="class_leads"
            )

        with col2:
            conversions = st.number_input(
                "Conversions", value=0.0,
                key="class_conversions"
            )

            acquisition_cost = st.number_input(
                "Acquisition Cost", value=0.0,
                key="class_cost"
            )

            roi = st.number_input(
                "ROI", value=0.0,
                key="class_roi"
            )

            engagement_score = st.number_input(
                "Engagement Score", value=0.0,
                key="class_engagement"
            )

        submit = st.form_submit_button(
            "Predict Profit Status"
        )

    if submit:

        input_data = pd.DataFrame({
            "Duration": [duration],
            "Impressions": [impressions],
            "Clicks": [clicks],
            "Leads": [leads],
            "Conversions": [conversions],
            "Acquisition_Cost": [acquisition_cost],
            "ROI": [roi],
            "Engagement_Score": [engagement_score]
        })

        expected = list(
            classification_model.feature_names_in_
        )

        missing = set(expected) - set(input_data.columns)

        if missing:
            st.error(
                f"Missing model features: {list(missing)}"
            )
        else:
            input_data = input_data[expected]

            prediction = classification_model.predict(
                input_data
            )[0]

            if prediction == "Profit":
                st.success("Predicted Status: PROFIT")
            else:
                st.error("Predicted Status: LOSS")

            if hasattr(
                classification_model, "predict_proba"
            ):
                probabilities = (
                    classification_model.predict_proba(
                        input_data
                    )[0]
                )

                probability_df = pd.DataFrame({
                    "Status": classification_model.classes_,
                    "Probability": probabilities * 100
                })

                st.subheader("Prediction Probability")

                st.bar_chart(
                    probability_df.set_index("Status"),
                    color="#16845B"
                )
