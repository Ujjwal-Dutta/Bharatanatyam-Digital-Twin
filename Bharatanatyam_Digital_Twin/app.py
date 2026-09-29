import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
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
# PROJECT PATHS
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "final_svm_bhangi_classifier.pkl"
)

REFERENCE_PATH = os.path.join(
    BASE_DIR,
    "data",
    "bhangi_reference_profiles.csv"
)

NORMALIZATION_PATH = os.path.join(
    BASE_DIR,
    "data",
    "musculoskeletal_risk_normalization_parameters.csv"
)


# ============================================================
# 16 BIOMECHANICAL FEATURES
# Same feature set used in the research pipeline
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


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():

    if not os.path.exists(MODEL_PATH):
        st.error(
            "SVM model not found.\n\n"
            "Please check:\n"
            "models/final_svm_bhangi_classifier.pkl"
        )
        st.stop()

    try:
        return joblib.load(MODEL_PATH)

    except Exception as e:
        st.error(f"Could not load SVM model: {e}")
        st.stop()


# ============================================================
# LOAD REFERENCE PROFILES
# ============================================================

@st.cache_data
def load_reference_profiles():

    if not os.path.exists(REFERENCE_PATH):
        st.error(
            "Bhangi reference profile file not found.\n\n"
            "Please check:\n"
            "data/bhangi_reference_profiles.csv"
        )
        st.stop()

    try:
        df = pd.read_csv(REFERENCE_PATH)

    except Exception as e:
        st.error(f"Could not read reference profiles: {e}")
        st.stop()

    if "class" not in df.columns:
        st.error(
            "The reference profile CSV does not contain "
            "the required 'class' column."
        )
        st.stop()

    missing_features = [
        feature
        for feature in FEATURE_COLUMNS
        if feature not in df.columns
    ]

    if missing_features:
        st.error(
            "The reference profile CSV is missing these features:\n\n"
            + ", ".join(missing_features)
        )
        st.stop()

    return df


# ============================================================
# LOAD RISK NORMALIZATION PARAMETERS
# ============================================================

@st.cache_data
def load_normalization_parameters():

    if not os.path.exists(NORMALIZATION_PATH):

        st.warning(
            "Risk normalization parameter file was not found. "
            "Fallback normalization will be used."
        )

        return None

    try:
        return pd.read_csv(NORMALIZATION_PATH)

    except Exception as e:

        st.warning(
            f"Could not read normalization parameters: {e}. "
            "Fallback normalization will be used."
        )

        return None


# ============================================================
# LOAD ALL RESEARCH ASSETS
# ============================================================

model = load_model()
reference_df = load_reference_profiles()
normalization_df = load_normalization_parameters()


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def clean_class_name(name):

    return str(name).replace(" Augmented", "").strip()


# ------------------------------------------------------------
# FIND Bhangi reference profile
# ------------------------------------------------------------

def find_reference_row(predicted_class):

    predicted_class = str(predicted_class)

    # Exact match
    exact = reference_df[
        reference_df["class"].astype(str) == predicted_class
    ]

    if len(exact) > 0:
        return exact.iloc[0]

    # Match after removing "Augmented"
    clean_name = clean_class_name(predicted_class)

    candidates = reference_df[
        reference_df["class"]
        .astype(str)
        .apply(clean_class_name)
        == clean_name
    ]

    if len(candidates) > 0:
        return candidates.iloc[0]

    return None


# ------------------------------------------------------------
# Find normalization feature column
# ------------------------------------------------------------

def find_normalization_feature_column():

    if normalization_df is None:
        return None

    possible_columns = [
        "feature",
        "feature_name",
        "risk_feature",
        "variable",
        "Feature",
        "Feature_Name"
    ]

    for column in possible_columns:

        if column in normalization_df.columns:
            return column

    return None


# ------------------------------------------------------------
# Get 5th and 95th percentile
# ------------------------------------------------------------

def get_normalization_range(feature_name, fallback_values):

    # --------------------------------------------------------
    # Try saved research normalization parameters first
    # --------------------------------------------------------

    if normalization_df is not None:

        feature_column = find_normalization_feature_column()

        if feature_column is not None:

            matches = normalization_df[
                normalization_df[feature_column]
                .astype(str)
                .str.strip()
                .str.lower()
                == str(feature_name).strip().lower()
            ]

            if len(matches) > 0:

                row = matches.iloc[0]

                lower_columns = [
                    "p05",
                    "P05",
                    "5th_percentile",
                    "5th Percentile",
                    "lower",
                    "Lower",
                    "min",
                    "Min",
                    "q05",
                    "Q05"
                ]

                upper_columns = [
                    "p95",
                    "P95",
                    "95th_percentile",
                    "95th Percentile",
                    "upper",
                    "Upper",
                    "max",
                    "Max",
                    "q95",
                    "Q95"
                ]

                lower = None
                upper = None

                for column in lower_columns:

                    if column in normalization_df.columns:

                        value = row[column]

                        if pd.notna(value):

                            lower = float(value)
                            break

                for column in upper_columns:

                    if column in normalization_df.columns:

                        value = row[column]

                        if pd.notna(value):

                            upper = float(value)
                            break

                if (
                    lower is not None
                    and upper is not None
                    and upper > lower
                ):

                    return lower, upper

    # --------------------------------------------------------
    # Fallback
    # --------------------------------------------------------

    values = np.asarray(
        fallback_values,
        dtype=float
    )

    values = values[np.isfinite(values)]

    if len(values) == 0:
        return 0.0, 1.0

    lower = float(np.percentile(values, 5))
    upper = float(np.percentile(values, 95))

    if upper <= lower:
        upper = lower + 1e-9

    return lower, upper


# ------------------------------------------------------------
# Normalize value to 0–100
# ------------------------------------------------------------

def normalize_to_100(
    value,
    lower,
    upper
):

    if upper <= lower:
        return 0.0

    normalized = (
        (float(value) - lower)
        /
        (upper - lower)
    ) * 100.0

    return float(
        np.clip(
            normalized,
            0,
            100
        )
    )


# ------------------------------------------------------------
# Normalize using research parameters
# ------------------------------------------------------------

def research_normalize(
    feature_name,
    value,
    fallback_values
):

    lower, upper = get_normalization_range(
        feature_name,
        fallback_values
    )

    score = normalize_to_100(
        value,
        lower,
        upper
    )

    return score, lower, upper


# ============================================================
# RISK CALCULATION
# ============================================================

def calculate_risk(
    input_values,
    reference_values
):

    """
    Project-defined posture-based musculoskeletal
    screening index.

    Components:
        1. Lower-limb risk
        2. Bilateral asymmetry risk
        3. Posture-deviation risk

    The final MRI is the mean of these three components.

    This is NOT a clinically validated injury probability.
    """


    # ========================================================
    # 1. KNEE AND HIP ANGLE DEVIATION
    # ========================================================

    knee_deviations = [

        abs(
            input_values["left_knee_angle"]
            -
            reference_values["left_knee_angle"]
        ),

        abs(
            input_values["right_knee_angle"]
            -
            reference_values["right_knee_angle"]
        )
    ]

    hip_deviations = [

        abs(
            input_values["left_hip_angle"]
            -
            reference_values["left_hip_angle"]
        ),

        abs(
            input_values["right_hip_angle"]
            -
            reference_values["right_hip_angle"]
        )
    ]

    knee_angle_deviation = float(
        np.mean(knee_deviations)
    )

    hip_angle_deviation = float(
        np.mean(hip_deviations)
    )


    # ========================================================
    # 2. BILATERAL ASYMMETRY
    # ========================================================

    knee_asymmetry = abs(
        input_values["left_knee_angle"]
        -
        input_values["right_knee_angle"]
    )

    hip_asymmetry = abs(
        input_values["left_hip_angle"]
        -
        input_values["right_hip_angle"]
    )

    shoulder_asymmetry = abs(
        input_values["left_shoulder_angle"]
        -
        input_values["right_shoulder_angle"]
    )

    elbow_asymmetry = abs(
        input_values["left_elbow_angle"]
        -
        input_values["right_elbow_angle"]
    )

    overall_asymmetry = float(
        np.mean([
            knee_asymmetry,
            hip_asymmetry,
            shoulder_asymmetry,
            elbow_asymmetry
        ])
    )


    # ========================================================
    # 3. REFERENCE-RELATIVE FEATURE DEVIATIONS
    # ========================================================

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


    # --------------------------------------------------------
    # Angle deviation
    # --------------------------------------------------------

    angle_deviations = []

    for feature in angle_features:

        reference = float(
            reference_values[feature]
        )

        value = float(
            input_values[feature]
        )

        if abs(reference) > 1e-9:

            deviation = (
                abs(value - reference)
                /
                abs(reference)
            ) * 100.0

        else:

            deviation = abs(value - reference)

        angle_deviations.append(
            min(deviation, 100.0)
        )


    # --------------------------------------------------------
    # Distance deviation
    # --------------------------------------------------------

    distance_deviations = []

    for feature in distance_features:

        reference = float(
            reference_values[feature]
        )

        value = float(
            input_values[feature]
        )

        if abs(reference) > 1e-9:

            deviation = (
                abs(value - reference)
                /
                abs(reference)
            ) * 100.0

        else:

            deviation = abs(value - reference) * 100.0

        distance_deviations.append(
            min(deviation, 100.0)
        )


    # --------------------------------------------------------
    # Symmetry deviation
    # --------------------------------------------------------

    symmetry_deviations = []

    for feature in symmetry_features:

        reference = float(
            reference_values[feature]
        )

        value = float(
            input_values[feature]
        )

        deviation = abs(
            value - reference
        )

        symmetry_deviations.append(
            min(deviation, 100.0)
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


    # ========================================================
    # 4. OVERALL POSTURE DEVIATION
    # ========================================================

    overall_posture_deviation = float(
        np.mean([
            angle_deviation_index,
            distance_deviation_index,
            symmetry_deviation_index
        ])
    )


    # ========================================================
    # 5. LOWER-LIMB RISK
    # ========================================================

    lower_limb_raw = float(
        np.mean([
            knee_angle_deviation,
            hip_angle_deviation
        ])
    )

    # Use saved research normalization when available
    lower_limb_risk_values = []

    knee_risk, knee_p05, knee_p95 = research_normalize(
        "knee_angle_deviation",
        knee_angle_deviation,
        knee_deviations
    )

    hip_risk, hip_p05, hip_p95 = research_normalize(
        "hip_angle_deviation",
        hip_angle_deviation,
        hip_deviations
    )

    lower_limb_risk = float(
        np.mean([
            knee_risk,
            hip_risk
        ])
    )


    # ========================================================
    # 6. ASYMMETRY RISK
    # ========================================================

    asymmetry_risk, asym_p05, asym_p95 = research_normalize(
        "overall_asymmetry",
        overall_asymmetry,
        [
            knee_asymmetry,
            hip_asymmetry,
            shoulder_asymmetry,
            elbow_asymmetry
        ]
    )


    # ========================================================
    # 7. POSTURE-DEVIATION RISK
    # ========================================================

    posture_component_values = [
        overall_posture_deviation,
        angle_deviation_index,
        distance_deviation_index,
        symmetry_deviation_index
    ]

    posture_risk_values = []

    posture_feature_names = [
        "overall_posture_deviation",
        "angle_deviation_index",
        "distance_deviation_index",
        "symmetry_deviation_index"
    ]

    for feature_name, value in zip(
        posture_feature_names,
        posture_component_values
    ):

        score, _, _ = research_normalize(
            feature_name,
            value,
            posture_component_values
        )

        posture_risk_values.append(score)

    posture_deviation_risk = float(
        np.mean(posture_risk_values)
    )


    # ========================================================
    # 8. FINAL MUSCULOSKELETAL RISK INDEX
    # ========================================================

    musculoskeletal_risk_index = float(
        np.mean([
            lower_limb_risk,
            asymmetry_risk,
            posture_deviation_risk
        ])
    )

    musculoskeletal_risk_index = float(
        np.clip(
            musculoskeletal_risk_index,
            0,
            100
        )
    )


    # ========================================================
    # 9. RISK CATEGORY
    # ========================================================

    if musculoskeletal_risk_index < 25:

        risk_category = "Low Risk"

    elif musculoskeletal_risk_index < 50:

        risk_category = "Moderate Risk"

    elif musculoskeletal_risk_index < 75:

        risk_category = "High Risk"

    else:

        risk_category = "Very High Risk"


    # ========================================================
    # RETURN RESULTS
    # ========================================================

    return {

        "knee_angle_deviation":
            knee_angle_deviation,

        "hip_angle_deviation":
            hip_angle_deviation,

        "knee_asymmetry":
            knee_asymmetry,

        "hip_asymmetry":
            hip_asymmetry,

        "shoulder_asymmetry":
            shoulder_asymmetry,

        "elbow_asymmetry":
            elbow_asymmetry,

        "overall_asymmetry":
            overall_asymmetry,

        "angle_deviation_index":
            angle_deviation_index,

        "distance_deviation_index":
            distance_deviation_index,

        "symmetry_deviation_index":
            symmetry_deviation_index,

        "overall_posture_deviation":
            overall_posture_deviation,

        "lower_limb_raw":
            lower_limb_raw,

        "lower_limb_risk":
            lower_limb_risk,

        "asymmetry_risk":
            asymmetry_risk,

        "posture_deviation_risk":
            posture_deviation_risk,

        "musculoskeletal_risk_index":
            musculoskeletal_risk_index,

        "risk_category":
            risk_category,

        "knee_p05":
            knee_p05,

        "knee_p95":
            knee_p95,

        "hip_p05":
            hip_p05,

        "hip_p95":
            hip_p95,

        "asymmetry_p05":
            asym_p05,

        "asymmetry_p95":
            asym_p95
    }


# ============================================================
# HEADER
# ============================================================

st.title(
    "💃 Bharatanatyam Digital Twin"
)

st.subheader(
    "AI-Driven Bhangi Analysis and "
    "Musculoskeletal Risk Assessment"
)

st.markdown(
    """
This interactive dashboard uses the trained **RBF SVM
Bhangi classifier** and the biomechanical feature framework
developed in the research project.

Enter the 16 biomechanical posture features and click
**Analyze Posture**.
"""
)

st.divider()


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("About the System")

    st.write(
        """
        **Input:** 16 biomechanical features

        **ML Model:** RBF SVM

        **Bhangi Classes:** 9

        **Risk Categories:** 4

        **Research Test Accuracy:** 83.03%

        The musculoskeletal risk score is a
        project-defined posture-based screening
        index and is not a clinical diagnosis.
        """
    )

    st.divider()

    st.caption(
        "Research prototype — Bharatanatyam "
        "Digital Twin project"
    )


# ============================================================
# INPUT SECTION
# ============================================================

st.header("1. Enter Biomechanical Features")

st.caption(
    "Enter values according to the feature definitions "
    "used in the research pipeline."
)


# ============================================================
# ROW 1 — UPPER BODY ANGLES
# ============================================================

col1, col2, col3, col4 = st.columns(4)

with col1:

    left_elbow_angle = st.number_input(
        "Left Elbow Angle",
        min_value=0.0,
        max_value=180.0,
        value=140.0,
        step=0.1
    )

with col2:

    right_elbow_angle = st.number_input(
        "Right Elbow Angle",
        min_value=0.0,
        max_value=180.0,
        value=145.0,
        step=0.1
    )

with col3:

    left_shoulder_angle = st.number_input(
        "Left Shoulder Angle",
        min_value=0.0,
        max_value=180.0,
        value=100.0,
        step=0.1
    )

with col4:

    right_shoulder_angle = st.number_input(
        "Right Shoulder Angle",
        min_value=0.0,
        max_value=180.0,
        value=100.0,
        step=0.1
    )


# ============================================================
# ROW 2 — HIP AND KNEE ANGLES
# ============================================================

col1, col2, col3, col4 = st.columns(4)

with col1:

    left_hip_angle = st.number_input(
        "Left Hip Angle",
        min_value=0.0,
        max_value=180.0,
        value=90.0,
        step=0.1
    )

with col2:

    right_hip_angle = st.number_input(
        "Right Hip Angle",
        min_value=0.0,
        max_value=180.0,
        value=90.0,
        step=0.1
    )

with col3:

    left_knee_angle = st.number_input(
        "Left Knee Angle",
        min_value=0.0,
        max_value=180.0,
        value=80.0,
        step=0.1
    )

with col4:

    right_knee_angle = st.number_input(
        "Right Knee Angle",
        min_value=0.0,
        max_value=180.0,
        value=80.0,
        step=0.1
    )


# ============================================================
# ROW 3 — DISTANCE FEATURES
# ============================================================

col1, col2, col3, col4 = st.columns(4)

with col1:

    shoulder_width = st.number_input(
        "Shoulder Width",
        min_value=0.0,
        max_value=2.0,
        value=0.25,
        step=0.01
    )

with col2:

    hip_width = st.number_input(
        "Hip Width",
        min_value=0.0,
        max_value=2.0,
        value=0.15,
        step=0.01
    )

with col3:

    knee_distance = st.number_input(
        "Knee Distance",
        min_value=0.0,
        max_value=2.0,
        value=0.40,
        step=0.01
    )

with col4:

    ankle_distance = st.number_input(
        "Ankle Distance",
        min_value=0.0,
        max_value=2.0,
        value=0.35,
        step=0.01
    )


# ============================================================
# ROW 4 — DIFFERENCE FEATURES
# ============================================================

col1, col2, col3, col4 = st.columns(4)

with col1:

    knee_angle_difference = st.number_input(
        "Knee Angle Difference",
        min_value=0.0,
        max_value=180.0,
        value=10.0,
        step=0.1
    )

with col2:

    hip_angle_difference = st.number_input(
        "Hip Angle Difference",
        min_value=0.0,
        max_value=180.0,
        value=10.0,
        step=0.1
    )

with col3:

    elbow_angle_difference = st.number_input(
        "Elbow Angle Difference",
        min_value=0.0,
        max_value=180.0,
        value=15.0,
        step=0.1
    )

with col4:

    shoulder_angle_difference = st.number_input(
        "Shoulder Angle Difference",
        min_value=0.0,
        max_value=180.0,
        value=15.0,
        step=0.1
    )


# ============================================================
# ANALYSIS BUTTON
# ============================================================

st.divider()

analyze = st.button(
    "🔍 ANALYZE POSTURE",
    type="primary",
    use_container_width=True
)


# ============================================================
# ANALYSIS
# ============================================================

if analyze:

    # ========================================================
    # CREATE INPUT
    # ========================================================

    input_values = {

        "left_elbow_angle":
            left_elbow_angle,

        "right_elbow_angle":
            right_elbow_angle,

        "left_shoulder_angle":
            left_shoulder_angle,

        "right_shoulder_angle":
            right_shoulder_angle,

        "left_hip_angle":
            left_hip_angle,

        "right_hip_angle":
            right_hip_angle,

        "left_knee_angle":
            left_knee_angle,

        "right_knee_angle":
            right_knee_angle,

        "shoulder_width":
            shoulder_width,

        "hip_width":
            hip_width,

        "knee_distance":
            knee_distance,

        "ankle_distance":
            ankle_distance,

        "knee_angle_difference":
            knee_angle_difference,

        "hip_angle_difference":
            hip_angle_difference,

        "elbow_angle_difference":
            elbow_angle_difference,

        "shoulder_angle_difference":
            shoulder_angle_difference
    }


    input_df = pd.DataFrame(
        [input_values],
        columns=FEATURE_COLUMNS
    )


    # ========================================================
    # SVM PREDICTION
    # ========================================================

    try:

        prediction = model.predict(
            input_df
        )[0]

    except Exception as e:

        st.error(
            f"SVM prediction failed: {e}"
        )

        st.stop()


    predicted_bhangi = str(
        prediction
    )


    # ========================================================
    # SVM DECISION SCORE
    # ========================================================

    decision_score = None

    if hasattr(model, "decision_function"):

        try:

            scores = model.decision_function(
                input_df
            )

            scores = np.asarray(scores)

            if scores.ndim == 1:

                decision_score = float(
                    np.max(scores)
                )

            else:

                decision_score = float(
                    np.max(scores[0])
                )

        except Exception:

            decision_score = None


    # ========================================================
    # FIND REFERENCE PROFILE
    # ========================================================

    reference_row = find_reference_row(
        predicted_bhangi
    )

    if reference_row is None:

        st.error(
            "Reference profile for the predicted "
            "Bhangi could not be found."
        )

        st.stop()


    reference_values = {}

    for feature in FEATURE_COLUMNS:

        reference_values[feature] = float(
            reference_row[feature]
        )


    # ========================================================
    # CALCULATE RISK
    # ========================================================

    risk_results = calculate_risk(
        input_values,
        reference_values
    )


    # ========================================================
    # RESULT HEADER
    # ========================================================

    st.divider()

    st.header(
        "2. Analysis Result"
    )


    result_col1, result_col2, result_col3 = st.columns(3)

    with result_col1:

        st.metric(
            "Predicted Bhangi",
            clean_class_name(
                predicted_bhangi
            )
        )

    with result_col2:

        st.metric(
            "Posture Deviation",
            f"{risk_results['overall_posture_deviation']:.2f}"
        )

    with result_col3:

        st.metric(
            "Musculoskeletal Risk Index",
            f"{risk_results['musculoskeletal_risk_index']:.2f}"
        )


    # ========================================================
    # RISK CATEGORY
    # ========================================================

    risk_category = risk_results[
        "risk_category"
    ]


    if risk_category == "Low Risk":

        st.success(
            f"Risk Category: {risk_category}"
        )

    elif risk_category == "Moderate Risk":

        st.info(
            f"Risk Category: {risk_category}"
        )

    elif risk_category == "High Risk":

        st.warning(
            f"Risk Category: {risk_category}"
        )

    else:

        st.error(
            f"Risk Category: {risk_category}"
        )


    # ========================================================
    # RISK COMPONENTS
    # ========================================================

    st.subheader(
        "Risk Components"
    )

    c1, c2, c3 = st.columns(3)

    with c1:

        st.metric(
            "Lower-Limb Risk",
            f"{risk_results['lower_limb_risk']:.2f}"
        )

    with c2:

        st.metric(
            "Asymmetry Risk",
            f"{risk_results['asymmetry_risk']:.2f}"
        )

    with c3:

        st.metric(
            "Posture-Deviation Risk",
            f"{risk_results['posture_deviation_risk']:.2f}"
        )


    # ========================================================
    # POSTURE METRICS
    # ========================================================

    st.subheader(
        "Posture Analysis"
    )

    p1, p2, p3 = st.columns(3)

    with p1:

        st.metric(
            "Angle Deviation",
            f"{risk_results['angle_deviation_index']:.2f}"
        )

    with p2:

        st.metric(
            "Distance Deviation",
            f"{risk_results['distance_deviation_index']:.2f}"
        )

    with p3:

        st.metric(
            "Symmetry Deviation",
            f"{risk_results['symmetry_deviation_index']:.2f}"
        )


    # ========================================================
    # ASYMMETRY
    # ========================================================

    st.subheader(
        "Left–Right Symmetry"
    )

    symmetry_df = pd.DataFrame({

        "Measure": [

            "Knee Asymmetry",
            "Hip Asymmetry",
            "Shoulder Asymmetry",
            "Elbow Asymmetry"

        ],

        "Value": [

            risk_results[
                "knee_asymmetry"
            ],

            risk_results[
                "hip_asymmetry"
            ],

            risk_results[
                "shoulder_asymmetry"
            ],

            risk_results[
                "elbow_asymmetry"
            ]
        ]
    })


    st.dataframe(
        symmetry_df.round(4),
        use_container_width=True,
        hide_index=True
    )


    # ========================================================
    # REFERENCE COMPARISON
    # ========================================================

    st.subheader(
        "Input vs Bhangi Reference Profile"
    )

    comparison_df = pd.DataFrame({

        "Feature":
            FEATURE_COLUMNS,

        "Input Value": [

            input_values[
                feature
            ]

            for feature in FEATURE_COLUMNS
        ],

        "Reference Value": [

            reference_values[
                feature
            ]

            for feature in FEATURE_COLUMNS
        ]
    })


    comparison_df[
        "Absolute Difference"
    ] = (

        comparison_df[
            "Input Value"
        ]
        -
        comparison_df[
            "Reference Value"
        ]

    ).abs()


    st.dataframe(
        comparison_df.round(4),
        use_container_width=True,
        hide_index=True
    )


    # ========================================================
    # FEATURE COMPARISON CHART
    # ========================================================

    st.subheader(
        "Biomechanical Feature Comparison"
    )

    chart = go.Figure()


    chart.add_trace(
        go.Bar(

            name="Input",

            x=FEATURE_COLUMNS,

            y=[
                input_values[
                    feature
                ]

                for feature in FEATURE_COLUMNS
            ]
        )
    )


    chart.add_trace(
        go.Bar(

            name="Reference",

            x=FEATURE_COLUMNS,

            y=[
                reference_values[
                    feature
                ]

                for feature in FEATURE_COLUMNS
            ]
        )
    )


    chart.update_layout(

        barmode="group",

        height=500,

        xaxis_title=
            "Biomechanical Feature",

        yaxis_title=
            "Value",

        xaxis_tickangle=-45,

        legend_title=
            "Comparison"
    )


    st.plotly_chart(
        chart,
        use_container_width=True
    )


    # ========================================================
    # NORMALIZATION INFORMATION
    # ========================================================

    with st.expander(
        "Risk Normalization Information"
    ):

        st.write(
            """
            Risk components are normalized to a 0–100
            scale. When available, the saved research
            5th and 95th percentile normalization
            parameters are used.
            """
        )

        normalization_info = pd.DataFrame({

            "Parameter": [

                "Knee deviation P05",
                "Knee deviation P95",
                "Hip deviation P05",
                "Hip deviation P95",
                "Asymmetry P05",
                "Asymmetry P95"

            ],

            "Value": [

                risk_results["knee_p05"],
                risk_results["knee_p95"],

                risk_results["hip_p05"],
                risk_results["hip_p95"],

                risk_results["asymmetry_p05"],
                risk_results["asymmetry_p95"]
            ]
        })


        st.dataframe(
            normalization_info.round(4),
            use_container_width=True,
            hide_index=True
        )


    # ========================================================
    # MODEL INFORMATION
    # ========================================================

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

        if decision_score is not None:

            st.write(
                f"Maximum SVM decision score: "
                f"{decision_score:.4f}"
            )

        st.caption(
            "The SVM decision score is not a probability."
        )


    # ========================================================
    # LIMITATION / DISCLAIMER
    # ========================================================

    st.caption(
        """
        **Important:** The musculoskeletal risk index is a
        project-defined posture-based screening measure.
        It is not a clinically validated injury probability
        or medical diagnosis. The current manual-input mode
        requires the 16 biomechanical features as input.
        """
    )
