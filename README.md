# 🩰 AI-Driven Digital Twin for Bharatanatyam Bhangi Analysis

### 🎭 Real-Time Posture Analysis & Musculoskeletal Risk Assessment Using Computer Vision

An AI-driven **Digital Twin framework for Bharatanatyam dancers** that combines human pose estimation, biomechanical feature extraction, machine learning, posture deviation analysis, and posture-based musculoskeletal risk assessment.

🔗 **Live Streamlit Dashboard:**
https://bharatanatyam-digital-twin.streamlit.app/

---

## 📌 Project Overview

Bharatanatyam involves complex postures, joint movements, body alignment, symmetry, and repeated movements. Incorrect or excessive postural deviations may contribute to musculoskeletal stress during training.

This project develops a computer-vision-based analytical framework that:

* 🧍 Extracts human body landmarks from Bharatanatyam images and videos
* 📐 Calculates joint angles, body alignment, symmetry, and biomechanical features
* 🤖 Classifies Bharatanatyam **Bhangi postures using Machine Learning**
* 🔍 Compares observed postures with reference Bhangi profiles
* 📊 Quantifies posture deviation
* 🦴 Estimates a **project-defined musculoskeletal risk index**
* 🎭 Creates an analytical **Digital Twin representation** of the dancer
* ⏱️ Performs temporal Bhangi and risk analysis from dance videos
* 📈 Provides an interactive Streamlit dashboard for visualization

---

## 🎯 Research Objectives

### Objective 1 — Dataset & Pose Analysis

Develop a dataset-based computer vision pipeline for Bharatanatyam posture analysis and human pose landmark extraction.

### Objective 2 — Biomechanical Feature Engineering

Extract meaningful biomechanical features including:

* Joint angles
* Shoulder and hip alignment
* Knee alignment
* Body symmetry
* Joint-to-joint distances
* Left-right angular differences

### Objective 3 — Bhangi Classification 🤖

Develop a machine-learning model to classify selected Bharatanatyam Bhangi postures.

The final **RBF SVM classifier** achieved:

**83.03% Test Accuracy**

on the held-out test dataset.

### Objective 4 — Posture Deviation Analysis 📐

Compare observed postures with Bhangi reference profiles to identify deviations in:

* Joint angles
* Body alignment
* Symmetry
* Relative distances
* Overall posture characteristics

### Objective 5 — Musculoskeletal Risk Assessment 🦴

Develop a project-defined posture-based risk assessment framework using:

* Lower-limb deviations
* Postural deviations
* Left-right asymmetry
* Alignment characteristics

The system categorizes analyzed frames into:

* 🟢 Low Risk
* 🟡 Moderate Risk
* 🟠 High Risk
* 🔴 Very High Risk

### Objective 6 — Digital Twin 🎭

Develop an analytical Digital Twin that combines:

**Pose → Bhangi → Posture Deviation → Risk Index → Risk Category**

to provide a unified representation of the dancer's analyzed state.

---

## 🧠 System Workflow

```text
                    🎥 Dance Images / Videos
                              │
                              ▼
                    🧍 Pose Estimation
                              │
                              ▼
                    📍 Body Landmarks
                              │
                              ▼
                 📐 Feature Engineering
                              │
             ┌────────────────┴────────────────┐
             ▼                                 ▼
      🤖 Bhangi Classification          📊 Reference Profiles
             │                                 │
             └──────────────┬──────────────────┘
                            ▼
                  🔍 Posture Deviation
                            │
                            ▼
                  🦴 Risk Assessment
                            │
                            ▼
                  🎭 Digital Twin State
                            │
                            ▼
                  📈 Interactive Dashboard
```

---

## 🛠️ Technologies Used

### 👁️ Computer Vision

* MediaPipe Pose Landmarker
* Human Pose Estimation
* Body Landmark Detection
* Image and Video Processing

### 🤖 Machine Learning

* Support Vector Machine (SVM)
* RBF Kernel
* Logistic Regression
* Random Forest
* K-Nearest Neighbors
* StandardScaler
* Stratified Train/Test Split
* 5-Fold Cross Validation

### 📐 Biomechanical Analysis

* Joint angle calculation
* Alignment analysis
* Left-right symmetry analysis
* Distance-based features
* Posture deviation analysis

### 🦴 Risk Analysis

* Lower-limb risk
* Asymmetry risk
* Posture deviation risk
* Musculoskeletal Risk Index (MRI)
* Risk categorization

### 📊 Dashboard

* Streamlit
* Plotly
* Pandas
* NumPy

### 💻 Development

* Python
* Google Colab
* GitHub
* Streamlit Community Cloud

---

## 🩰 Bhangi Classes

The system analyzes **9 Bharatanatyam Bhangi classes**:

1. Ardhamandalam
2. Bramha
3. Garuda
4. Muzhumandi
5. Nagabandham
6. Nataraj
7. Prenkhana
8. Samapadam
9. Swastika

---

## 📊 Dataset & Pose Extraction

The initial posture-image dataset contained:

**1,731 images**

After pose extraction, confidence-based recovery, rotation recovery, and quality filtering:

**1,649 final landmark records**

were retained for feature engineering and machine-learning analysis.

### Pose Extraction Performance

| Metric                   | Result |
| ------------------------ | -----: |
| Original Images          |  1,731 |
| Successful Final Records |  1,649 |
| Final Detection Rate     | 95.27% |
| Bhangi Classes           |      9 |

Rotation recovery was also applied to improve landmark extraction for images where the original orientation produced unsuccessful detection.

---

## 🤖 Machine Learning Results

Multiple classification algorithms were evaluated.

| Model               | Test Accuracy |
| ------------------- | ------------: |
| Logistic Regression |        72.42% |
| SVM                 |    **83.03%** |
| Random Forest       |        79.70% |
| KNN                 |        79.70% |

The final **RBF SVM model** was selected for the dashboard.

### Final SVM Results

* 🎯 Test Accuracy: **83.03%**
* ⚖️ Balanced Accuracy: **81.06%**
* 📌 Macro Precision: **85.60%**
* 📌 Macro Recall: **81.06%**
* 📌 Macro F1-Score: **82.90%**

---

## 🎥 Video Analysis

Three Bharatanatyam performance videos were analyzed:

* 🩰 Alarippu
* 🩰 Kathanakuthula
* 🩰 Sakshi

The videos were sampled at **5 frames per second (FPS)** for temporal analysis.

### Video Pose Extraction

| Metric                    | Result |
| ------------------------- | -----: |
| Total Video Frames        |  5,155 |
| Successfully Detected     |  5,057 |
| Detection Rate            | 98.10% |
| Acceptable Quality Frames |  3,368 |

The selected frames were further analyzed for Bhangi classification, posture deviation, temporal transitions, and musculoskeletal risk.

---

## 🦴 Musculoskeletal Risk Assessment

The project defines a posture-based **Musculoskeletal Risk Index (MRI)** using three major components:

```text
Lower-Limb Risk
       +
Asymmetry Risk
       +
Posture Deviation Risk
       │
       ▼
Musculoskeletal Risk Index
       │
       ▼
Risk Category
```

### Risk Categories

| MRI Range | Category     |
| --------- | ------------ |
| < 25      | 🟢 Low       |
| 25 – < 50 | 🟡 Moderate  |
| 50 – < 75 | 🟠 High      |
| ≥ 75      | 🔴 Very High |

### Overall Video Analysis

* Minimum MRI: **7.86**
* Maximum MRI: **78.91**
* Mean MRI: **44.42**

### Video-wise Mean MRI

| Video          | Mean MRI |
| -------------- | -------: |
| Alarippu       |    36.89 |
| Kathanakuthula |    45.58 |
| Sakshi         |    46.64 |

---

## 🎭 Digital Twin

The Digital Twin combines the outputs of the complete analytical pipeline into a unified state representation.

Each analyzed frame contains information such as:

```text
Video
Frame Number
Pose Quality
Predicted Bhangi
Posture Deviation
Angle Deviation
Distance Deviation
Symmetry Deviation
Musculoskeletal Risk Index
Risk Category
```

This enables frame-level and temporal analysis of the dancer's posture.

---

## 📊 Interactive Dashboard

The project includes a Streamlit dashboard with four major sections:

### 1️⃣ Manual Analysis

Users can provide biomechanical feature values and obtain:

* 🤖 Predicted Bhangi
* 📊 Reference comparison
* 📐 Posture deviation
* 🦴 Risk index
* 🚦 Risk category
* 📈 Interactive visualizations

### 2️⃣ Video Analysis

Provides:

* Bhangi distribution
* Temporal Bhangi sequence
* Bhangi transitions
* Temporal stability
* Posture deviation
* Musculoskeletal risk trends
* Video-wise analysis

### 3️⃣ Digital Twin State

Provides frame-level Digital Twin information including:

* 🩰 Predicted Bhangi
* 📐 Posture deviation
* 🦴 Musculoskeletal Risk Index
* 🚦 Risk category
* 🧍 Digital Twin visualization

### 4️⃣ Research Results

Displays important research outcomes including:

* Dataset statistics
* Landmark extraction results
* Bhangi classification performance
* Video analysis
* Risk assessment results
* Research visualizations

---

## 🌐 Live Dashboard

🚀 **Try the Interactive Dashboard:**

👉 https://bharatanatyam-digital-twin.streamlit.app/

---

## 📁 Repository Structure

```text
Bharatanatyam_Digital_Twin/
│
├── 📄 app.py
├── 📄 requirements.txt
├── 📓 Bharatanatyam_Digital_Twin.ipynb
│
├── 🤖 models/
│   └── final_svm_bhangi_classifier.pkl
│
├── 📊 data/
│   ├── bhangi_reference_profiles.csv
│   ├── engineered_biomechanical_features.csv
│   ├── bhangi_musculoskeletal_risk_features.csv
│   ├── bhangi_musculoskeletal_risk_normalized.csv
│   ├── musculoskeletal_risk_normalization_parameters.csv
│   ├── video_musculoskeletal_risk_index.csv
│   ├── video_musculoskeletal_risk_categories.csv
│   ├── video_bhangi_classification.csv
│   ├── video_temporal_bhangi_predictions.csv
│   ├── video_posture_deviation_analysis.csv
│   └── digital_twin_state_data.csv
│
└── 🖼️ images/
    ├── analytical frames/
    │   ├── Alarippu_digital_twin.png
    │   ├── Kathanakuthula_digital_twin.png
    │   └── Sakshi_digital_twin.png
    │
    └── final_dashboard/
        ├── Alarippu_final_digital_twin_dashboard.png
        ├── Kathanakuthula_final_digital_twin_dashboard.png
        └── Sakshi_final_digital_twin_dashboard.png
```

---

## 📓 Research Notebook

The repository also contains the complete research notebook:

**`Bharatanatyam_Digital_Twin.ipynb`**

The notebook documents the major stages of the project, including:

* Dataset processing
* Pose extraction
* Rotation recovery
* Feature engineering
* Machine-learning model development
* Video analysis
* Posture deviation analysis
* Musculoskeletal risk assessment
* Digital Twin development
* Research visualizations

---

## ✨ Key Contributions

### 🔹 AI-Based Bhangi Classification

Machine-learning-based classification of 9 Bharatanatyam Bhangi postures.

### 🔹 Biomechanical Feature Analysis

Uses joint angles, alignment, distances, and symmetry to characterize Bharatanatyam postures.

### 🔹 Posture Deviation Quantification

Provides a numerical comparison between observed postures and Bhangi reference profiles.

### 🔹 Posture-Based Risk Screening

Introduces a project-defined framework for estimating musculoskeletal risk from posture-related characteristics.

### 🔹 Temporal Dance Analysis

Analyzes Bhangi sequences, transitions, stability, and risk changes across dance videos.

### 🔹 Analytical Digital Twin

Integrates posture, classification, deviation, and risk information into a unified dancer-state representation.

### 🔹 Interactive Decision-Support Dashboard

Provides an accessible interface for exploring the analytical results.

---

## ⚠️ Important Limitations

This project should be interpreted within the scope of its research methodology.

* 🩺 The musculoskeletal risk index is a **project-defined posture-based screening measure**, not a clinically validated injury prediction system.
* 🎥 Video Bhangi labels are generated by the trained SVM model; frame-level video ground-truth annotations were not available for calculating video classification accuracy.
* 🎭 The Digital Twin is an **analytical and visual posture Digital Twin**, rather than a complete physical musculoskeletal simulation.
* ⚡ The current video pipeline uses offline **5 FPS analysis**. Real-time/live-stream latency has not been formally benchmarked.
* 🔁 Movement repetition and exposure duration are analyzed separately and are not directly integrated into the current MRI calculation.

---

## 🚀 Future Enhancements

Future versions can extend the system with:

* 📹 Live webcam-based posture analysis
* ⚡ Real-time inference and latency benchmarking
* 🧠 Temporal deep-learning models such as LSTM/Transformer
* 🦴 More detailed biomechanical modeling
* 📊 Larger and more diverse Bharatanatyam datasets
* 👩‍⚕️ Validation with dance experts and healthcare professionals
* 🩺 Clinically validated musculoskeletal assessment protocols
* 🎭 Frame-synchronized Digital Twin visualizations
* 📱 Deployment as a mobile or web-based training assistant

---

## 👨‍💻 Project

**AI-Driven Digital Twin for Real-Time Bharatanatyam Bhangi Analysis and Musculoskeletal Risk Assessment Using Computer Vision**

### 🔬 Research Areas

`Computer Vision` • `Human Pose Estimation` • `Machine Learning` • `Biomechanics` • `Digital Twin` • `Decision Support System` • `Bharatanatyam` • `Musculoskeletal Risk Assessment`

---

## ⭐ Project Highlights

🩰 **9** Bharatanatyam Bhangi Classes
🖼️ **1,731** Original Posture Images
📍 **1,649** Final Landmark Records
🤖 **83.03%** SVM Test Accuracy
🎥 **5,155** Video Frames Analyzed
📊 **3,368** Quality-Filtered Video Frames
🎭 **Digital Twin** Analytical Framework
📈 **Interactive Streamlit Dashboard**

---

### 🌟 Built with Python, Computer Vision, Machine Learning & Bharatanatyam Research

**Developed as an M.Tech research project.**
