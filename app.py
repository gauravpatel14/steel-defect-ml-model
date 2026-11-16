import json
from pathlib import Path

import joblib
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ---------------- PAGE CONFIG ----------------

st.set_page_config(
    page_title="Steel Defect AI",
    page_icon="🏭",
    layout="wide"
)

MODEL_FILE = Path("steel_defect_model.pkl")
METRICS_FILE = Path("model_metrics.json")

FEATURES = [
    "X_Minimum", "X_Maximum", "Y_Minimum", "Y_Maximum",
    "Pixels_Areas", "X_Perimeter", "Y_Perimeter",
    "Sum_of_Luminosity", "Minimum_of_Luminosity",
    "Maximum_of_Luminosity", "Length_of_Conveyer",
    "TypeOfSteel_A300", "TypeOfSteel_A400",
    "Steel_Plate_Thickness", "Edges_Index", "Empty_Index",
    "Square_Index", "Outside_X_Index", "Edges_X_Index",
    "Edges_Y_Index", "Outside_Global_Index", "LogOfAreas",
    "Log_X_Index", "Log_Y_Index", "Orientation_Index",
    "Luminosity_Index", "SigmoidOfAreas"
]

DEFECTS = [
    "Pastry", "Z_Scratch", "K_Scatch", "Stains",
    "Dirtiness", "Bumps", "Other_Faults"
]

# ---------------- LOAD FILES ----------------

@st.cache_resource
def load_model():
    return joblib.load(MODEL_FILE)


def load_metrics():
    if not METRICS_FILE.exists():
        return None

    with open(METRICS_FILE, "r", encoding="utf-8") as file:
        return json.load(file)


if not MODEL_FILE.exists():
    st.error(
        "Model file not found. Keep steel_defect_model.pkl "
        "in the same folder as app.py."
    )
    st.stop()

model = load_model()
metrics = load_metrics()

# ---------------- DESIGN ----------------

st.markdown("""
<style>
.stApp {
    background-color: #0b1220;
    color: #f8fafc;
}

[data-testid="stSidebar"] {
    background-color: #111827;
    border-right: 1px solid #263449;
}

.hero {
    padding: 28px;
    border: 1px solid #263449;
    border-radius: 18px;
    background: linear-gradient(120deg, #1b304a, #101a2a);
    margin-bottom: 28px;
}

.hero h1 {
    color: #ffffff;
    font-size: 36px;
    margin-bottom: 8px;
}

.hero p {
    color: #b7c7d9;
    font-size: 16px;
    margin-bottom: 0;
}

div[data-testid="stMetric"] {
    background: #111d2e;
    border: 1px solid #263449;
    padding: 18px;
    border-radius: 14px;
}

div[data-testid="stMetric"] label {
    color: #b7c7d9;
}

div[data-testid="stMetricValue"] {
    color: #ffffff;
}

.section-card {
    background: #111d2e;
    border: 1px solid #263449;
    padding: 18px;
    border-radius: 14px;
    margin-bottom: 14px;
}

.small-note {
    color: #aab8ca;
    font-size: 13px;
}

.stButton button[kind="primary"] {
    background-color: #2563eb;
    border: none;
    color: white;
    border-radius: 9px;
    padding: 10px 20px;
}

a {
    color: #80bfff !important;
}
</style>
""", unsafe_allow_html=True)

# ---------------- SIDEBAR ----------------

with st.sidebar:
    st.markdown("## 🏭 Steel Defect AI")
    st.caption("Machine Learning Project")
    st.divider()

    page = st.radio(
        "Navigation",
        [
            "Dashboard",
            "Predict Defect",
            "Model Performance",
            "About Project"
        ],
        label_visibility="visible"
    )

    st.divider()
    st.caption("Academic Machine Learning Project")
    st.caption("Selected Model: Random Forest")

# ---------------- COMMON HEADER ----------------

st.markdown("""
<div class="hero">
    <h1>Steel Plate Defect Detection</h1>
    <p>Machine-learning based classification of steel plate surface defects</p>
</div>
""", unsafe_allow_html=True)


# =====================================================
# DASHBOARD
# =====================================================

if page == "Dashboard":

    st.subheader("Project Dashboard")

    col1, col2, col3 = st.columns(3)

    col1.metric("Dataset Records", "1,941")
    col2.metric("Input Features", "27")
    col3.metric("Defect Categories", "7")

    st.markdown("### What does this application do?")

    st.write(
        "This application uses a trained Machine Learning model "
        "to identify the type of defect present on a steel plate "
        "using its measurement values."
    )

    st.info(
        "Upload a CSV file to predict steel plate defects. "
        "One row gives one prediction, while multiple rows give batch predictions."
    )

    st.markdown("### Defect Categories")

    defect_cols = st.columns(4)

    for index, defect in enumerate(DEFECTS):
        defect_cols[index % 4].markdown(
            f"<div class='section-card'>{defect}</div>",
            unsafe_allow_html=True
        )

    st.markdown("### Selected Model")

    st.success(
        "Random Forest achieved the highest test accuracy "
        "among the three models tested."
    )

    if metrics:
        rf = next(
            (
                item for item in metrics.get("model_comparison", [])
                if item.get("Model") == "Random Forest"
            ),
            None
        )

        if rf:
            st.metric(
                "Random Forest Test Accuracy",
                f"{rf['Accuracy'] * 100:.2f}%"
            )


# =====================================================
# PREDICT DEFECT
# =====================================================

elif page == "Predict Defect":

    st.subheader("Predict Steel Plate Defects")

    st.write(
        "Upload a CSV file containing the 27 measurement columns "
        "to classify steel plate defects."
    )

    st.caption(
        "One row = Single prediction  |  Multiple rows = Batch prediction"
    )

    template = pd.DataFrame(columns=FEATURES)

    st.download_button(
        label="⬇️ Download CSV Template",
        data=template.to_csv(index=False).encode("utf-8"),
        file_name="steel_defect_template.csv",
        mime="text/csv"
    )

    uploaded_file = st.file_uploader(
        "Upload your CSV file",
        type=["csv"]
    )

    if uploaded_file is not None:

        try:
            data = pd.read_csv(uploaded_file)
            

            missing_columns = [
                feature for feature in FEATURES
                if feature not in data.columns
            ]

            if missing_columns:
                st.error(
                    "Your CSV is missing these columns: "
                    + ", ".join(missing_columns)
                )

            elif data.empty:
                st.warning("The uploaded CSV has no data rows.")

            else:
                input_data = data[FEATURES].apply(
                    pd.to_numeric,
                    errors="coerce"
                )

                if input_data.isnull().any().any():
                    st.error(
                        "Some values are empty or non-numeric. "
                        "Please correct your CSV and upload it again."
                    )

                else:
                    st.success(
                        f"{len(input_data)} row(s) are ready for prediction."
                    )

                    with st.expander("Preview uploaded data"):
                        st.dataframe(
                            input_data.head(),
                            use_container_width=True
                        )

                    if st.button(
                        "🔍 Predict Defect",
                        type="primary",
                        use_container_width=True
                    ):

                        predictions = model.predict(input_data)

                        results = pd.DataFrame({
                            "Predicted Defect": predictions
                        })

                        st.markdown("### Prediction Results")

                        if len(results) == 1:
                            st.success(
                                f"Predicted Defect: **{predictions[0]}**"
                            )

                        st.dataframe(
                            results,
                            use_container_width=True,
                            hide_index=True
                        )

                        counts = (
                            results["Predicted Defect"]
                            .value_counts()
                            .rename_axis("Defect Type")
                            .reset_index(name="Count")
                        )

                        left, right = st.columns(2)

                        with left:
                            st.markdown("#### Prediction Summary")
                            st.dataframe(
                                counts,
                                use_container_width=True,
                                hide_index=True
                            )

                        with right:
                            fig = px.bar(
                                counts,
                                x="Defect Type",
                                y="Count",
                                title="Predictions by Defect Type",
                                text="Count"
                            )
                            fig.update_layout(
                                template="plotly_dark",
                                paper_bgcolor="#0b1220",
                                plot_bgcolor="#111d2e"
                            )
                            st.plotly_chart(
                                fig,
                                use_container_width=True
                            )

                        st.download_button(
                            "⬇️ Download Prediction Results",
                            data=results.to_csv(index=False).encode("utf-8"),
                            file_name="steel_defect_predictions.csv",
                            mime="text/csv"
                        )

        except Exception as error:
            st.error(f"Could not process the CSV file: {error}")


# =====================================================
# MODEL PERFORMANCE
# =====================================================

elif page == "Model Performance":

    st.subheader("Model Performance Comparison")

    if metrics is None:
        st.warning(
            "model_metrics.json is missing. "
            "Place it in the same folder as app.py."
        )

    else:
        comparison = pd.DataFrame(
            metrics.get("model_comparison", [])
        )

        required_columns = [
            "Model", "Accuracy", "Precision", "Recall", "F1 Score"
        ]

        if not all(col in comparison.columns for col in required_columns):
            st.error("The model_metrics.json file has an incomplete comparison table.")

        else:
            st.markdown("### Performance Summary")

            display_table = comparison[required_columns].copy()

            for column in required_columns[1:]:
                display_table[column] = (
                    display_table[column] * 100
                ).round(2).astype(str) + "%"

            st.dataframe(
                display_table,
                use_container_width=True,
                hide_index=True
            )

            st.markdown("### Metric Comparison")

            chart_data = comparison.melt(
                id_vars="Model",
                value_vars=[
                    "Accuracy", "Precision", "Recall", "F1 Score"
                ],
                var_name="Metric",
                value_name="Score"
            )

            fig = px.bar(
                chart_data,
                x="Model",
                y="Score",
                color="Metric",
                barmode="group",
                text=chart_data["Score"].apply(
                    lambda value: f"{value * 100:.1f}%"
                ),
                title="Comparison of All Three Models"
            )

            fig.update_yaxes(
                range=[0, 1],
                tickformat=".0%"
            )

            fig.update_layout(
                template="plotly_dark",
                paper_bgcolor="#0b1220",
                plot_bgcolor="#111d2e",
                legend_title="Evaluation Metric"
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

            best_model = metrics.get("best_model", "Not available")

            st.success(
                f"Best Model based on test accuracy: {best_model}"
            )

            st.markdown("### Best Model — Random Forest")

            rf = next(
                (
                    item for item in metrics.get("model_comparison", [])
                    if item.get("Model") == best_model
                ),
                None
            )

            if rf:
                m1, m2, m3, m4 = st.columns(4)

                m1.metric("Accuracy", f"{rf['Accuracy'] * 100:.2f}%")
                m2.metric("Precision", f"{rf['Precision'] * 100:.2f}%")
                m3.metric("Recall", f"{rf['Recall'] * 100:.2f}%")
                m4.metric("F1 Score", f"{rf['F1 Score'] * 100:.2f}%")

            # Confusion Matrix
            if metrics.get("confusion_matrix") and metrics.get("class_labels"):

                st.markdown("### Confusion Matrix — Random Forest")

                st.caption(
                    "Rows show actual defect types. Columns show predicted "
                    "defect types. Values on the diagonal are correct predictions."
                )

                labels = metrics["class_labels"]

                matrix = pd.DataFrame(
                    metrics["confusion_matrix"],
                    index=labels,
                    columns=labels
                )

                fig = px.imshow(
                    matrix,
                    text_auto=True,
                    aspect="auto",
                    x=labels,
                    y=labels,
                    color_continuous_scale="Blues",
                    labels={
                        "x": "Predicted Defect",
                        "y": "Actual Defect",
                        "color": "Number of Records"
                    }
                )

                fig.update_layout(
                    template="plotly_dark",
                    paper_bgcolor="#0b1220",
                    plot_bgcolor="#111d2e"
                )

                st.plotly_chart(
                    fig,
                    use_container_width=True
                )

                st.info(
                    "The confusion matrix above belongs to the selected "
                    "Random Forest model, not to all three models."
                )


# =====================================================
# ABOUT PROJECT
# =====================================================

elif page == "About Project":

    st.subheader("About This Project")

    st.write(
        "This project uses Supervised Machine Learning classification "
        "to predict the type of surface defect in a steel plate from "
        "its numerical measurement values."
    )

    st.markdown("### Problem Statement")

    st.write(
        "Manual inspection of steel plate surfaces can take time. "
        "This project demonstrates how Machine Learning can classify "
        "steel plate defects using measurement data."
    )

    st.markdown("### Dataset Information")

    info1, info2, info3 = st.columns(3)

    info1.metric("Total Records", "1,941")
    info2.metric("Input Features", "27")
    info3.metric("Defect Classes", "7")

    st.markdown("### Defect Categories")

    st.write(
        "Pastry, Z_Scratch, K_Scatch, Stains, "
        "Dirtiness, Bumps and Other_Faults."
    )

    st.markdown("### Algorithms Tested")

    st.write("""
    - Decision Tree
    - Random Forest
    - Support Vector Machine (SVM)
    """)

    st.markdown("### Selected Model")

    st.success("Random Forest — Highest test accuracy among the three models.")

    st.markdown("### Dataset Sources")

    st.markdown(
        "[Original Dataset — UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/198/steel+plates+faults)"
    )

    st.markdown(
        "[Kaggle Dataset — Faulty Steel Plates](https://www.kaggle.com/datasets/uciml/faulty-steel-plates)"
    )

    st.markdown("### Project Limitations")

    st.write(
        "The application is an academic demonstration. "
        "Predictions depend on the quality and correctness of the uploaded "
        "measurement data. The model's performance is based on the test "
        "dataset used during evaluation."
    )

    st.markdown("### Future Scope")

    st.write(
        "The project can be extended in the future with more steel plate "
        "inspection data and integration with industrial inspection systems."
    )

    st.divider()

    st.caption(
        "Academic Machine Learning Project | Steel Plate Defect Detection"
    )