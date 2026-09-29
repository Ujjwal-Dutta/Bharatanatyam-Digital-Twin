import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import plotly.express as px
import plotly.graph_objects as go

# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Bharatanatyam Digital Twin",
    page_icon="💃",
    layout="wide"
)

# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(
    BASE_DIR, "models", "final_svm_bhangi_classifier.pkl"
)

REFERENCE_PATH = os.path.join(
    BASE_DIR, "data", "bhangi_reference_profiles.csv"
)

# ============================================================
# RESEARCH FEATURE DEFINITIONS
# ============================================================

FEATURE_COLUMNS = [
    "left_elbow_angle",
    "right_elbow_angle",
    "left_shoulder_angle",
    "right_shoulder_angle",
    "left_hip_angle",
    "right_hip_angle",
    "left_knee_angle",
    "right_knee_angle",
    "shoulder_width",
    "hip_width",
    "knee_distance",
    "ankle_distance",
    "knee_angle_difference",
    "hip_angle_difference",
    "elbow_angle_difference",
    "shoulder_angle_difference"
]

ANGLE_FEATURES = set(
    FEATURE_COLUMNS[:8] + FEATURE_COLUMNS[12:]
)

# ============================================================
# RESEARCH RESULT FILES
# ============================================================

VIDEO_FILES = {
    "classification": "video_bhangi_classification.csv",
    "temporal": "video_temporal_bhangi_predictions.csv",
    "transitions": "video_bhangi_transition_summary.csv",
    "stability": "video_temporal_stability.csv",
    "duration": "video_bhangi_duration_summary.csv",
    "posture": "video_posture_deviation_analysis.csv",
    "risk_index": "video_musculoskeletal_risk_index.csv",
    "risk_category": "video_musculoskeletal_risk_categories.csv",
    "state": "digital_twin_state_data.csv",
    "twin_summary": "digital_twin_temporal_summary.csv",
    "twin_transitions": "digital_twin_transition_summary.csv",
    "twin_risk_transitions": "digital_twin_risk_transition_summary.csv"
}

# ============================================================
# LOAD FUNCTIONS
# ============================================================

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        st.error(
            "SVM model not found. Please check "
            "models/final_svm_bhangi_classifier.pkl."
        )
        st.stop()

    return joblib.load(MODEL_PATH)


@st.cache_data
def load_reference_profiles():
    if not os.path.exists(REFERENCE_PATH):
        st.error(
            "Bhangi reference profiles file not found. "
            "Please check data/bhangi_reference_profiles.csv."
        )
        st.stop()

    return pd.read_csv(REFERENCE_PATH)


@st.cache_data
def load_csv(filename):
    path = os.path.join(BASE_DIR, "data", filename)

    if not os.path.exists(path):
        return None

    try:
        return pd.read_csv(path)
    except Exception as exc:
        st.warning(f"Could not read {filename}: {exc}")
        return None


model = load_model()
reference_df = load_reference_profiles()

video_data = {
    key: load_csv(filename)
    for key, filename in VIDEO_FILES.items()
}

# ============================================================
# GENERAL HELPERS
# ============================================================

def clean_class_name(name):
    return str(name).replace(" Augmented", "").strip()


def find_col(df, candidates):
    if df is None or df.empty:
        return None

    lookup = {
        str(column).strip().lower(): column
        for column in df.columns
    }

    for candidate in candidates:
        if candidate.lower() in lookup:
            return lookup[candidate.lower()]

    return None


def standardize_video_keys(df):
    """
    Standardizes common video/frame column names without
    changing the actual research values.
    """
    if df is None or df.empty:
        return None

    out = df.copy()

    video_col = find_col(
        out,
        ["video", "video_name", "source_video", "video_id"]
    )

    frame_col = find_col(
        out,
        ["frame_number", "frame", "frame_id"]
    )

    if video_col and video_col != "video":
        out = out.rename(columns={video_col: "video"})

    if frame_col and frame_col != "frame_number":
        out = out.rename(columns={frame_col: "frame_number"})

    if "video" in out.columns:
        out["video"] = out["video"].astype(str)

    if "frame_number" in out.columns:
        out["frame_number"] = pd.to_numeric(
            out["frame_number"],
            errors="coerce"
        )

    return out


def find_reference_row(predicted_class):
    class_col = find_col(
        reference_df,
        ["class", "bhangi", "label"]
    )

    if class_col is None:
        return None

    exact = reference_df[
        reference_df[class_col].astype(str)
        == str(predicted_class)
    ]

    if not exact.empty:
        return exact.iloc[0]

    target = clean_class_name(predicted_class)

    matches = reference_df[
        reference_df[class_col]
        .astype(str)
        .apply(clean_class_name)
        == target
    ]

    if not matches.empty:
        return matches.iloc[0]

    return None


def safe_normalize(value, minimum, maximum):
    if not np.isfinite(value) or maximum <= minimum:
        return 0.0

    normalized = (
        (value - minimum)
        / (maximum - minimum)
    ) * 100.0

    return float(np.clip(normalized, 0, 100))


def category_text(value):
    if pd.isna(value):
        return "Unknown"

    text = str(value)

    if "Very High" in text:
        return "Very High Risk"

    if "High" in text:
        return "High Risk"

    if "Moderate" in text:
        return "Moderate Risk"

    if "Low" in text:
        return "Low Risk"

    return text


# ============================================================
# MANUAL RISK CALCULATION
# ============================================================

def calculate_manual_risk(input_values, reference_values):

    knee_angle_deviation = np.mean([
        abs(
            input_values["left_knee_angle"]
            - reference_values["left_knee_angle"]
        ),
        abs(
            input_values["right_knee_angle"]
            - reference_values["right_knee_angle"]
        )
    ])

    hip_angle_deviation = np.mean([
        abs(
            input_values["left_hip_angle"]
            - reference_values["left_hip_angle"]
        ),
        abs(
            input_values["right_hip_angle"]
            - reference_values["right_hip_angle"]
        )
    ])

    knee_asymmetry = abs(
        input_values["left_knee_angle"]
        - input_values["right_knee_angle"]
    )

    hip_asymmetry = abs(
        input_values["left_hip_angle"]
        - input_values["right_hip_angle"]
    )

    shoulder_asymmetry = abs(
        input_values["left_shoulder_angle"]
        - input_values["right_shoulder_angle"]
    )

    elbow_asymmetry = abs(
        input_values["left_elbow_angle"]
        - input_values["right_elbow_angle"]
    )

    overall_asymmetry = np.mean([
        knee_asymmetry,
        hip_asymmetry,
        shoulder_asymmetry,
        elbow_asymmetry
    ])

    angle_features = [
        "left_elbow_angle",
        "right_elbow_angle",
        "left_shoulder_angle",
        "right_shoulder_angle",
        "left_hip_angle",
        "right_hip_angle",
        "left_knee_angle",
        "right_knee_angle"
    ]

    distance_features = [
        "shoulder_width",
        "hip_width",
        "knee_distance",
        "ankle_distance"
    ]

    symmetry_features = [
        "knee_angle_difference",
        "hip_angle_difference",
        "elbow_angle_difference",
        "shoulder_angle_difference"
    ]

    angle_deviations = []

    for feature in angle_features:
        reference = reference_values[feature]
        value = input_values[feature]

        if abs(reference) > 1e-9:
            deviation = (
                abs(value - reference)
                / abs(reference)
            ) * 100
        else:
            deviation = abs(value - reference)

        angle_deviations.append(
            min(deviation, 100)
        )

    distance_deviations = []

    for feature in distance_features:
        reference = reference_values[feature]
        value = input_values[feature]

        if abs(reference) > 1e-9:
            deviation = (
                abs(value - reference)
                / abs(reference)
            ) * 100
        else:
            deviation = abs(value - reference) * 100

        distance_deviations.append(
            min(deviation, 100)
        )

    symmetry_deviations = []

    for feature in symmetry_features:
        deviation = abs(
            input_values[feature]
            - reference_values[feature]
        )

        symmetry_deviations.append(
            min(deviation, 100)
        )

    angle_deviation_index = float(
        np.mean(angle_deviations)
    )

    distance_deviation_index = float(
        np.mean(distance_deviations)
    )

    symmetry_deviation_index = float(
        np.mean(symmetry_deviations)
    )

    overall_posture_deviation = float(
        np.mean([
            angle_deviation_index,
            distance_deviation_index,
            symmetry_deviation_index
        ])
    )

    lower_limb_raw = np.mean([
        knee_angle_deviation,
        hip_angle_deviation
    ])

    asymmetry_raw = overall_asymmetry

    posture_raw = np.mean([
        overall_posture_deviation,
        angle_deviation_index,
        distance_deviation_index,
        symmetry_deviation_index
    ])

    lower_limb_risk = safe_normalize(
        lower_limb_raw,
        0,
        60
    )

    asymmetry_risk = safe_normalize(
        asymmetry_raw,
        0,
        60
    )

    posture_deviation_risk = safe_normalize(
        posture_raw,
        0,
        100
    )

    musculoskeletal_risk_index = float(
        np.clip(
            np.mean([
                lower_limb_risk,
                asymmetry_risk,
                posture_deviation_risk
            ]),
            0,
            100
        )
    )

    if musculoskeletal_risk_index < 25:
        risk_category = "Low Risk"
    elif musculoskeletal_risk_index < 50:
        risk_category = "Moderate Risk"
    elif musculoskeletal_risk_index < 75:
        risk_category = "High Risk"
    else:
        risk_category = "Very High Risk"

    return {
        "knee_angle_deviation": knee_angle_deviation,
        "hip_angle_deviation": hip_angle_deviation,
        "knee_asymmetry": knee_asymmetry,
        "hip_asymmetry": hip_asymmetry,
        "shoulder_asymmetry": shoulder_asymmetry,
        "elbow_asymmetry": elbow_asymmetry,
        "overall_asymmetry": overall_asymmetry,
        "angle_deviation_index": angle_deviation_index,
        "distance_deviation_index": distance_deviation_index,
        "symmetry_deviation_index": symmetry_deviation_index,
        "overall_posture_deviation": overall_posture_deviation,
        "lower_limb_risk": lower_limb_risk,
        "asymmetry_risk": asymmetry_risk,
        "posture_deviation_risk": posture_deviation_risk,
        "musculoskeletal_risk_index": musculoskeletal_risk_index,
        "risk_category": risk_category
    }


# ============================================================
# HEADER
# ============================================================

st.title("💃 Bharatanatyam Digital Twin")

st.subheader(
    "AI-Driven Bhangi Analysis and "
    "Musculoskeletal Risk Assessment"
)

st.caption(
    "Research prototype using the trained RBF-SVM classifier, "
    "biomechanical analysis, video results and Digital Twin state data."
)

with st.sidebar:

    st.header("About the System")

    st.write("**Input:** 16 biomechanical features")
    st.write("**ML Model:** RBF SVM")
    st.write("**Bhangi Classes:** 9")
    st.write("**Risk Categories:** 4")
    st.write("**Research Test Accuracy:** 83.03%")

    st.caption(
        "The musculoskeletal risk index is a "
        "project-defined posture-based screening "
        "measure and is not a clinical diagnosis."
    )


# ============================================================
# TABS
# ============================================================

tab_manual, tab_video, tab_twin, tab_results = st.tabs([
    "🧮 Manual Analysis",
    "🎥 Video Analysis",
    "🧍 Digital Twin",
    "📊 Research Results"
])


# ============================================================
# TAB 1 — MANUAL ANALYSIS
# ============================================================

with tab_manual:

    st.header("1. Manual Bhangi Analysis")

    st.write(
        "Enter the 16 biomechanical features used "
        "in the research pipeline."
    )

    defaults = {
        "left_elbow_angle": 140.0,
        "right_elbow_angle": 145.0,
        "left_shoulder_angle": 100.0,
        "right_shoulder_angle": 100.0,
        "left_hip_angle": 90.0,
        "right_hip_angle": 90.0,
        "left_knee_angle": 80.0,
        "right_knee_angle": 80.0,
        "shoulder_width": 0.25,
        "hip_width": 0.15,
        "knee_distance": 0.40,
        "ankle_distance": 0.35,
        "knee_angle_difference": 10.0,
        "hip_angle_difference": 10.0,
        "elbow_angle_difference": 15.0,
        "shoulder_angle_difference": 15.0
    }

    labels = {
        "left_elbow_angle": "Left Elbow Angle",
        "right_elbow_angle": "Right Elbow Angle",
        "left_shoulder_angle": "Left Shoulder Angle",
        "right_shoulder_angle": "Right Shoulder Angle",
        "left_hip_angle": "Left Hip Angle",
        "right_hip_angle": "Right Hip Angle",
        "left_knee_angle": "Left Knee Angle",
        "right_knee_angle": "Right Knee Angle",
        "shoulder_width": "Shoulder Width",
        "hip_width": "Hip Width",
        "knee_distance": "Knee Distance",
        "ankle_distance": "Ankle Distance",
        "knee_angle_difference": "Knee Angle Difference",
        "hip_angle_difference": "Hip Angle Difference",
        "elbow_angle_difference": "Elbow Angle Difference",
        "shoulder_angle_difference": "Shoulder Angle Difference"
    }

    values = {}

    columns = st.columns(4)

    for index, feature in enumerate(FEATURE_COLUMNS):

        with columns[index % 4]:

            if feature in ANGLE_FEATURES:

                values[feature] = st.number_input(
                    labels[feature],
                    min_value=0.0,
                    max_value=180.0,
                    value=defaults[feature],
                    step=0.1,
                    key=f"manual_{feature}"
                )

            else:

                values[feature] = st.number_input(
                    labels[feature],
                    min_value=0.0,
                    max_value=2.0,
                    value=defaults[feature],
                    step=0.01,
                    key=f"manual_{feature}"
                )

    if st.button(
        "🔍 ANALYZE POSTURE",
        type="primary",
        use_container_width=True
    ):

        input_df = pd.DataFrame(
            [values],
            columns=FEATURE_COLUMNS
        )

        prediction = str(
            model.predict(input_df)[0]
        )

        reference_row = find_reference_row(
            prediction
        )

        if reference_row is None:

            st.error(
                "Reference profile for the predicted "
                "Bhangi could not be found."
            )

        else:

            reference_values = {
                feature: float(
                    reference_row[feature]
                )
                for feature in FEATURE_COLUMNS
            }

            risk = calculate_manual_risk(
                values,
                reference_values
            )

            st.subheader("Analysis Result")

            c1, c2, c3, c4 = st.columns(4)

            c1.metric(
                "Predicted Bhangi",
                clean_class_name(prediction)
            )

            c2.metric(
                "Posture Deviation",
                f"{risk['overall_posture_deviation']:.2f}"
            )

            c3.metric(
                "Musculoskeletal Risk Index",
                f"{risk['musculoskeletal_risk_index']:.2f}"
            )

            c4.metric(
                "Risk Category",
                risk["risk_category"]
            )

            st.subheader("Risk Components")

            r1, r2, r3 = st.columns(3)

            r1.metric(
                "Lower-Limb Risk",
                f"{risk['lower_limb_risk']:.2f}"
            )

            r2.metric(
                "Asymmetry Risk",
                f"{risk['asymmetry_risk']:.2f}"
            )

            r3.metric(
                "Posture-Deviation Risk",
                f"{risk['posture_deviation_risk']:.2f}"
            )

            st.subheader("Left–Right Symmetry")

            symmetry_df = pd.DataFrame({
                "Measure": [
                    "Knee Asymmetry",
                    "Hip Asymmetry",
                    "Shoulder Asymmetry",
                    "Elbow Asymmetry"
                ],
                "Value": [
                    risk["knee_asymmetry"],
                    risk["hip_asymmetry"],
                    risk["shoulder_asymmetry"],
                    risk["elbow_asymmetry"]
                ]
            })

            st.dataframe(
                symmetry_df.round(3),
                use_container_width=True,
                hide_index=True
            )

            st.subheader(
                "Input vs Bhangi Reference Profile"
            )

            comparison_df = pd.DataFrame({
                "Feature": FEATURE_COLUMNS,
                "Input Value": [
                    values[f]
                    for f in FEATURE_COLUMNS
                ],
                "Reference Value": [
                    reference_values[f]
                    for f in FEATURE_COLUMNS
                ]
            })

            comparison_df[
                "Absolute Difference"
            ] = (
                comparison_df["Input Value"]
                - comparison_df["Reference Value"]
            ).abs()

            st.dataframe(
                comparison_df.round(4),
                use_container_width=True,
                hide_index=True
            )

            st.subheader(
                "Biomechanical Feature Comparison"
            )

            chart = go.Figure()

            chart.add_trace(
                go.Bar(
                    name="Input",
                    x=FEATURE_COLUMNS,
                    y=[
                        values[f]
                        for f in FEATURE_COLUMNS
                    ]
                )
            )

            chart.add_trace(
                go.Bar(
                    name="Reference",
                    x=FEATURE_COLUMNS,
                    y=[
                        reference_values[f]
                        for f in FEATURE_COLUMNS
                    ]
                )
            )

            chart.update_layout(
                barmode="group",
                height=500,
                xaxis_title="Biomechanical Feature",
                yaxis_title="Value",
                xaxis_tickangle=-45
            )

            st.plotly_chart(
                chart,
                use_container_width=True
            )

            with st.expander(
                "Model Information"
            ):

                st.write(
                    "Classifier: RBF SVM"
                )

                st.write(
                    "Number of Bhangi classes: 9"
                )

                st.write(
                    "Research test accuracy: 83.03%"
                )

                if hasattr(
                    model,
                    "decision_function"
                ):

                    try:

                        scores = model.decision_function(
                            input_df
                        )

                        st.write(
                            "Maximum SVM decision score: "
                            f"{float(np.max(scores)):.4f}"
                        )

                        st.caption(
                            "The SVM decision score is not "
                            "a probability."
                        )

                    except Exception:
                        pass


# ============================================================
# TAB 2 — VIDEO ANALYSIS
# ============================================================

with tab_video:

    st.header("2. Video Analysis")

    st.write(
        "Research results from the three analyzed "
        "Bharatanatyam videos."
    )

    classification = standardize_video_keys(
        video_data["classification"]
    )

    posture = standardize_video_keys(
        video_data["posture"]
    )

    risk_index = standardize_video_keys(
        video_data["risk_index"]
    )

    risk_category = standardize_video_keys(
        video_data["risk_category"]
    )

    temporal = standardize_video_keys(
        video_data["temporal"]
    )

    transitions = video_data["transitions"]
    stability = video_data["stability"]
    duration = video_data["duration"]

    if classification is None:

        st.warning(
            "video_bhangi_classification.csv was not "
            "found in the data folder."
        )

    else:

        if "video" not in classification.columns:

            st.error(
                "The classification file does not contain "
                "a recognizable video column."
            )

        else:

            video_options = sorted(
                classification["video"]
                .dropna()
                .astype(str)
                .unique()
            )

            selected_video = st.selectbox(
                "Select Video",
                video_options
            )

            selected_classification = (
                classification[
                    classification["video"].astype(str)
                    == selected_video
                ]
                .copy()
            )

            selected_posture = None

            if posture is not None and "video" in posture.columns:

                selected_posture = posture[
                    posture["video"].astype(str)
                    == selected_video
                ].copy()

            selected_risk = None

            if risk_index is not None and "video" in risk_index.columns:

                selected_risk = risk_index[
                    risk_index["video"].astype(str)
                    == selected_video
                ].copy()

            selected_category = None

            if (
                risk_category is not None
                and "video" in risk_category.columns
            ):

                selected_category = risk_category[
                    risk_category["video"].astype(str)
                    == selected_video
                ].copy()

            # ------------------------------------------------
            # KEY METRICS
            # ------------------------------------------------

            frame_count = len(
                selected_classification
            )

            mri_col = find_col(
                selected_risk,
                [
                    "musculoskeletal_risk_index",
                    "mri",
                    "risk_index"
                ]
            )

            posture_col = find_col(
                selected_posture,
                [
                    "overall_posture_deviation",
                    "posture_deviation"
                ]
            )

            mean_mri = np.nan
            max_mri = np.nan

            if (
                selected_risk is not None
                and mri_col is not None
            ):

                mri_values = pd.to_numeric(
                    selected_risk[mri_col],
                    errors="coerce"
                ).dropna()

                if not mri_values.empty:

                    mean_mri = float(
                        mri_values.mean()
                    )

                    max_mri = float(
                        mri_values.max()
                    )

            mean_posture = np.nan

            if (
                selected_posture is not None
                and posture_col is not None
            ):

                posture_values = pd.to_numeric(
                    selected_posture[posture_col],
                    errors="coerce"
                ).dropna()

                if not posture_values.empty:

                    mean_posture = float(
                        posture_values.mean()
                    )

            k1, k2, k3, k4 = st.columns(4)

            k1.metric(
                "Analyzed Frames",
                f"{frame_count:,}"
            )

            k2.metric(
                "Mean MRI",
                f"{mean_mri:.2f}"
                if np.isfinite(mean_mri)
                else "N/A"
            )

            k3.metric(
                "Maximum MRI",
                f"{max_mri:.2f}"
                if np.isfinite(max_mri)
                else "N/A"
            )

            k4.metric(
                "Mean Posture Deviation",
                f"{mean_posture:.2f}"
                if np.isfinite(mean_posture)
                else "N/A"
            )

            # ------------------------------------------------
            # BHANGI DISTRIBUTION
            # ------------------------------------------------

            bhangi_col = find_col(
                selected_classification,
                [
                    "predicted_bhangi",
                    "bhangi",
                    "predicted_class",
                    "class"
                ]
            )

            if bhangi_col is not None:

                st.subheader(
                    "Bhangi Distribution"
                )

                bhangi_counts = (
                    selected_classification[
                        bhangi_col
                    ]
                    .astype(str)
                    .map(clean_class_name)
                    .value_counts()
                    .reset_index()
                )

                bhangi_counts.columns = [
                    "Bhangi",
                    "Frame Count"
                ]

                fig = px.bar(
                    bhangi_counts,
                    x="Bhangi",
                    y="Frame Count",
                    title=(
                        "Bhangi Distribution — "
                        f"{selected_video}"
                    )
                )

                fig.update_layout(
                    xaxis_tickangle=-35
                )

                st.plotly_chart(
                    fig,
                    use_container_width=True
                )

            # ------------------------------------------------
            # RISK DISTRIBUTION
            # ------------------------------------------------

            category_col = find_col(
                selected_category,
                [
                    "risk_category",
                    "category"
                ]
            )

            if category_col is not None:

                st.subheader(
                    "Risk Category Distribution"
                )

                risk_counts = (
                    selected_category[
                        category_col
                    ]
                    .astype(str)
                    .map(category_text)
                    .value_counts()
                    .reset_index()
                )

                risk_counts.columns = [
                    "Risk Category",
                    "Frame Count"
                ]

                fig = px.pie(
                    risk_counts,
                    names="Risk Category",
                    values="Frame Count",
                    title=(
                        "Risk Distribution — "
                        f"{selected_video}"
                    )
                )

                st.plotly_chart(
                    fig,
                    use_container_width=True
                )

            # ------------------------------------------------
            # TEMPORAL RISK TREND
            # ------------------------------------------------

            if (
                selected_posture is not None
                and selected_risk is not None
                and posture_col is not None
                and mri_col is not None
                and "frame_number" in selected_posture.columns
                and "frame_number" in selected_risk.columns
            ):

                st.subheader(
                    "Temporal Risk and Posture Trend"
                )

                p = selected_posture[
                    ["frame_number", posture_col]
                ].copy()

                r = selected_risk[
                    ["frame_number", mri_col]
                ].copy()

                p[posture_col] = pd.to_numeric(
                    p[posture_col],
                    errors="coerce"
                )

                r[mri_col] = pd.to_numeric(
                    r[mri_col],
                    errors="coerce"
                )

                trend = p.merge(
                    r,
                    on="frame_number",
                    how="inner"
                ).dropna()

                trend = trend.sort_values(
                    "frame_number"
                )

                if not trend.empty:

                    fig = go.Figure()

                    fig.add_trace(
                        go.Scatter(
                            x=trend["frame_number"],
                            y=trend[posture_col],
                            name="Posture Deviation"
                        )
                    )

                    fig.add_trace(
                        go.Scatter(
                            x=trend["frame_number"],
                            y=trend[mri_col],
                            name="Musculoskeletal Risk Index"
                        )
                    )

                    fig.update_layout(
                        height=450,
                        xaxis_title="Frame Number",
                        yaxis_title="Value"
                    )

                    st.plotly_chart(
                        fig,
                        use_container_width=True
                    )

            # ------------------------------------------------
            # TEMPORAL BHANGI SEQUENCE
            # ------------------------------------------------

            temporal_bhangi_col = find_col(
                temporal,
                [
                    "predicted_bhangi",
                    "bhangi",
                    "predicted_class",
                    "class"
                ]
            )

            if (
                temporal is not None
                and temporal_bhangi_col is not None
                and "video" in temporal.columns
                and "frame_number" in temporal.columns
            ):

                selected_temporal = temporal[
                    temporal["video"].astype(str)
                    == selected_video
                ].copy()

                if not selected_temporal.empty:

                    st.subheader(
                        "Temporal Bhangi Sequence"
                    )

                    sequence = selected_temporal[
                        [
                            "frame_number",
                            temporal_bhangi_col
                        ]
                    ].dropna().copy()

                    sequence[
                        "Bhangi"
                    ] = sequence[
                        temporal_bhangi_col
                    ].astype(str).map(
                        clean_class_name
                    )

                    sequence = sequence.sort_values(
                        "frame_number"
                    )

                    classes = list(
                        dict.fromkeys(
                            sequence["Bhangi"].tolist()
                        )
                    )

                    class_to_number = {
                        value: index
                        for index, value
                        in enumerate(classes)
                    }

                    sequence["Code"] = (
                        sequence["Bhangi"]
                        .map(class_to_number)
                    )

                    fig = px.scatter(
                        sequence,
                        x="frame_number",
                        y="Code",
                        color="Bhangi",
                        title=(
                            "Temporal Bhangi Sequence — "
                            f"{selected_video}"
                        ),
                        labels={
                            "frame_number": "Frame Number",
                            "Code": "Bhangi"
                        }
                    )

                    fig.update_traces(
                        marker={"size": 6}
                    )

                    fig.update_yaxes(
                        tickmode="array",
                        tickvals=list(
                            range(len(classes))
                        ),
                        ticktext=classes
                    )

                    st.plotly_chart(
                        fig,
                        use_container_width=True
                    )

            # ------------------------------------------------
            # TRANSITIONS / STABILITY
            # ------------------------------------------------

            if transitions is not None:

                transition_video_col = find_col(
                    transitions,
                    [
                        "video",
                        "video_name",
                        "source_video"
                    ]
                )

                if transition_video_col:

                    selected_transitions = transitions[
                        transitions[
                            transition_video_col
                        ].astype(str)
                        == selected_video
                    ]

                else:

                    selected_transitions = transitions

                if not selected_transitions.empty:

                    st.subheader(
                        "Bhangi Transition Summary"
                    )

                    st.dataframe(
                        selected_transitions,
                        use_container_width=True,
                        hide_index=True
                    )

            if stability is not None:

                stability_video_col = find_col(
                    stability,
                    [
                        "video",
                        "video_name",
                        "source_video"
                    ]
                )

                if stability_video_col:

                    selected_stability = stability[
                        stability[
                            stability_video_col
                        ].astype(str)
                        == selected_video
                    ]

                else:

                    selected_stability = stability

                if not selected_stability.empty:

                    st.subheader(
                        "Temporal Stability"
                    )

                    st.dataframe(
                        selected_stability,
                        use_container_width=True,
                        hide_index=True
                    )

            # ------------------------------------------------
            # SAMPLE
            # ------------------------------------------------

            st.subheader(
                "Video Classification Sample"
            )

            display_cols = [
                column
                for column in [
                    "video",
                    "frame_number",
                    bhangi_col
                ]
                if column is not None
                and column in selected_classification.columns
            ]

            if display_cols:

                st.dataframe(
                    selected_classification[
                        display_cols
                    ].head(100),
                    use_container_width=True,
                    hide_index=True
                )

            st.info(
                "Video Bhangi labels are SVM predictions on "
                "extracted video frames. Video classification "
                "accuracy is not claimed because manual "
                "frame-level ground truth was not established."
            )


# ============================================================
# TAB 3 — DIGITAL TWIN
# ============================================================

with tab_twin:

    st.header("3. Digital Twin State")

    st.write(
        "Select a video and frame to inspect the corresponding "
        "Digital Twin state generated by the research pipeline."
    )

    twin = standardize_video_keys(
        video_data["state"]
    )

    if twin is None:

        st.warning(
            "digital_twin_state_data.csv was not found "
            "in the data folder."
        )

    elif "video" not in twin.columns:

        st.error(
            "The Digital Twin state file does not contain "
            "a recognizable video column."
        )

    elif "frame_number" not in twin.columns:

        st.error(
            "The Digital Twin state file does not contain "
            "frame_number."
        )

    else:

        twin_videos = sorted(
            twin["video"]
            .dropna()
            .astype(str)
            .unique()
        )

        twin_video = st.selectbox(
            "Digital Twin Video",
            twin_videos,
            key="digital_twin_video"
        )

        twin_selected = twin[
            twin["video"].astype(str)
            == twin_video
        ].copy()

        twin_selected[
            "frame_number"
        ] = pd.to_numeric(
            twin_selected["frame_number"],
            errors="coerce"
        )

        twin_selected = twin_selected.dropna(
            subset=["frame_number"]
        ).sort_values(
            "frame_number"
        )

        minimum_frame = int(
            twin_selected[
                "frame_number"
            ].min()
        )

        maximum_frame = int(
            twin_selected[
                "frame_number"
            ].max()
        )

        selected_frame = st.slider(
            "Select Frame",
            min_value=minimum_frame,
            max_value=maximum_frame,
            value=minimum_frame,
            step=1
        )

        nearest_index = (
            twin_selected[
                "frame_number"
            ]
            - selected_frame
        ).abs().idxmin()

        row = twin_selected.loc[
            nearest_index
        ]

        bhangi_col = find_col(
            twin_selected,
            [
                "predicted_bhangi",
                "bhangi",
                "predicted_class",
                "class"
            ]
        )

        posture_col = find_col(
            twin_selected,
            [
                "overall_posture_deviation",
                "posture_deviation"
            ]
        )

        mri_col = find_col(
            twin_selected,
            [
                "musculoskeletal_risk_index",
                "mri",
                "risk_index"
            ]
        )

        risk_col = find_col(
            twin_selected,
            [
                "risk_category",
                "category"
            ]
        )

        quality_col = find_col(
            twin_selected,
            [
                "quality",
                "quality_label"
            ]
        )

        c1, c2, c3, c4 = st.columns(4)

        c1.metric(
            "Frame",
            int(row["frame_number"])
        )

        c2.metric(
            "Predicted Bhangi",
            clean_class_name(
                row[bhangi_col]
            )
            if bhangi_col
            and pd.notna(row[bhangi_col])
            else "N/A"
        )

        c3.metric(
            "MRI",
            f"{float(row[mri_col]):.2f}"
            if mri_col
            and pd.notna(row[mri_col])
            else "N/A"
        )

        c4.metric(
            "Risk",
            category_text(
                row[risk_col]
            )
            if risk_col
            else "N/A"
        )

        if posture_col:

            st.metric(
                "Posture Deviation",
                f"{float(row[posture_col]):.2f}"
                if pd.notna(row[posture_col])
                else "N/A"
            )

        if quality_col:

            st.write(
                f"**Frame Quality:** {row[quality_col]}"
            )

        # ----------------------------------------------------
        # PRE-RENDERED ANALYTICS FRAME
        # ----------------------------------------------------

        image_dir = os.path.join(
            BASE_DIR,
            "images",
            "analytics_frames"
        )

        image_candidates = [
            f"{clean_class_name(twin_video)}_digital_twin.png",
            f"{twin_video}_digital_twin.png"
        ]

        image_path = None

        for filename in image_candidates:

            candidate = os.path.join(
                image_dir,
                filename
            )

            if os.path.exists(candidate):

                image_path = candidate
                break

        if image_path:

            st.image(
                image_path,
                caption=(
                    "Digital Twin Analytics Frame — "
                    f"{twin_video}"
                ),
                use_container_width=True
            )

        else:

            st.info(
                "No pre-rendered Digital Twin image was "
                "found in images/analytics_frames. "
                "The numerical Digital Twin state is still available."
            )

        st.subheader(
            "Selected Digital Twin State"
        )

        state_display = (
            row.to_frame("Value")
            .reset_index()
        )

        state_display.columns = [
            "Field",
            "Value"
        ]

        st.dataframe(
            state_display.head(100),
            use_container_width=True,
            hide_index=True
        )


# ============================================================
# TAB 4 — RESEARCH RESULTS
# ============================================================

with tab_results:

    st.header("4. Research Results")

    st.write(
        "Key measured results from the completed "
        "research pipeline."
    )

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Original Posture Images",
        "1,731"
    )

    c2.metric(
        "Final Landmark Records",
        "1,649"
    )

    c3.metric(
        "Bhangi Classes",
        "9"
    )

    c4.metric(
        "SVM Test Accuracy",
        "83.03%"
    )

    st.subheader(
        "Model Performance"
    )

    performance_df = pd.DataFrame({
        "Metric": [
            "Accuracy",
            "Balanced Accuracy",
            "Macro Precision",
            "Macro Recall",
            "Macro F1"
        ],
        "Value (%)": [
            83.03,
            81.06,
            85.60,
            81.06,
            82.90
        ]
    })

    st.dataframe(
        performance_df,
        use_container_width=True,
        hide_index=True
    )

    # --------------------------------------------------------
    # VIDEO-WISE RISK
    # --------------------------------------------------------

    risk_summary_source = standardize_video_keys(
        video_data["risk_index"]
    )

    if (
        risk_summary_source is not None
        and "video" in risk_summary_source.columns
    ):

        risk_value_col = find_col(
            risk_summary_source,
            [
                "musculoskeletal_risk_index",
                "mri",
                "risk_index"
            ]
        )

        if risk_value_col:

            risk_summary = risk_summary_source.copy()

            risk_summary[
                risk_value_col
            ] = pd.to_numeric(
                risk_summary[
                    risk_value_col
                ],
                errors="coerce"
            )

            summary = (
                risk_summary
                .dropna(
                    subset=[risk_value_col]
                )
                .groupby("video")[
                    risk_value_col
                ]
                .agg(
                    Analyzed_Frames="count",
                    Mean_MRI="mean",
                    Maximum_MRI="max"
                )
                .reset_index()
            )

            summary.columns = [
                "Video",
                "Analyzed Frames",
                "Mean MRI",
                "Maximum MRI"
            ]

            st.subheader(
                "Video-Wise Musculoskeletal Risk"
            )

            st.dataframe(
                summary.round(2),
                use_container_width=True,
                hide_index=True
            )

            fig = px.bar(
                summary,
                x="Video",
                y="Mean MRI",
                title="Mean Musculoskeletal Risk Index by Video"
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

    # --------------------------------------------------------
    # RESEARCH NOTES
    # --------------------------------------------------------

    st.subheader(
        "Research Notes"
    )

    st.markdown(
        """
- The final posture-image dataset contains **1,649 usable landmark records** from **1,731 images**.
- The final classifier is an **RBF SVM** trained on the 16 engineered biomechanical features.
- Test accuracy was **83.03%** on the untouched stratified test set.
- The video pipeline analyzed extracted frames from **Alarippu, Kathanakuthala and Sakshi**.
- Video Bhangi labels are model predictions because manual frame-level ground truth was not established.
- The musculoskeletal risk index is a **project-defined posture-based screening index**, not a clinically validated injury probability or medical diagnosis.
- The current dashboard presents completed offline research outputs and does not claim real-time latency benchmarking.
"""
    )

    st.divider()

    st.caption(
        "Bharatanatyam Digital Twin — Research Prototype"
    )
