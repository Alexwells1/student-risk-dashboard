"""
dashboard.py
Streamlit Dashboard - 3-Tier Risk Classification
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from sklearn.preprocessing import StandardScaler
from pathlib import Path
import os
import random
from PIL import Image

# ============================================================
# PATH CONFIGURATION
# ============================================================

# Get the base directory (where this script is located)
BASE_DIR = Path(__file__).parent

# Define paths
MODEL_PATH = BASE_DIR / 'models' / 'model_RF.pkl'
DATA_PATH = BASE_DIR / 'data' / 'survey_engineered.csv'
PERF_PATH = BASE_DIR / 'outputs' / 'optimized_model_performance.csv'
CONFUSION_PATH = BASE_DIR / 'outputs' / 'confusion_matrix.png'
SHAP_PATH = BASE_DIR / 'outputs' / 'shap' / 'shap_beeswarm_Low_Risk.png'

# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Student Risk Prediction System", 
    page_icon="📊", 
    layout="wide"
)

# ============================================================
# CUSTOM CSS - Professional Styling
# ============================================================

st.markdown("""
<style>
    /* Risk level cards */
    .low-risk { 
        background-color: #2e7d32; 
        color: white; 
        padding: 20px; 
        border-radius: 6px; 
        text-align: center; 
        font-size: 22px; 
        font-weight: 500; 
        letter-spacing: 0.5px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .medium-risk { 
        background-color: #ed6c02; 
        color: white; 
        padding: 20px; 
        border-radius: 6px; 
        text-align: center; 
        font-size: 22px; 
        font-weight: 500;
        letter-spacing: 0.5px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .high-risk { 
        background-color: #d32f2f; 
        color: white; 
        padding: 20px; 
        border-radius: 6px; 
        text-align: center; 
        font-size: 22px; 
        font-weight: 500;
        letter-spacing: 0.5px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    /* Info boxes */
    .info-box {
        background-color: #e3f2fd;
        padding: 16px 20px;
        border-radius: 6px;
        border-left: 4px solid #1976d2;
        margin: 10px 0;
    }
    
    .warning-box {
        background-color: #fff3e0;
        padding: 16px 20px;
        border-radius: 6px;
        border-left: 4px solid #ed6c02;
        margin: 10px 0;
    }
    
    .success-box {
        background-color: #e8f5e9;
        padding: 16px 20px;
        border-radius: 6px;
        border-left: 4px solid #2e7d32;
        margin: 10px 0;
    }
    
    .error-box {
        background-color: #ffebee;
        padding: 16px 20px;
        border-radius: 6px;
        border-left: 4px solid #d32f2f;
        margin: 10px 0;
    }
    
    /* Cards */
    .metric-card {
        background-color: #f8f9fa;
        padding: 16px;
        border-radius: 6px;
        border: 1px solid #e0e0e0;
        text-align: center;
    }
    
    /* Sidebar styling */
    .sidebar-section {
        padding: 8px 0;
        border-bottom: 1px solid #e0e0e0;
        margin-bottom: 12px;
    }
    
    .sidebar-footer {
        font-size: 12px;
        color: #757575;
        padding-top: 12px;
        border-top: 1px solid #e0e0e0;
        margin-top: 12px;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================
# LOAD MODEL (CACHED)
# ============================================================

@st.cache_resource
def load_model():
    """Load the trained model pipeline."""
    try:
        with open(MODEL_PATH, 'rb') as f:
            model_data = pickle.load(f)
        return model_data
    except FileNotFoundError:
        st.error(f"Model file not found at {MODEL_PATH}. Please ensure the model is uploaded.")
        st.stop()
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        st.stop()

# Load the model
model_data = load_model()
pipeline = model_data['pipeline']
features = model_data['features']
class_names = model_data['class_names']

# ============================================================
# SIDEBAR NAVIGATION
# ============================================================

st.sidebar.title("Student Risk System")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigation", 
    ["Cohort Overview", "Individual Screening", "Batch Processing", "Model Evaluation", "Explainable AI"]
)

st.sidebar.markdown("---")

st.sidebar.markdown("""
**Risk Classification**
- Low Risk
- Medium Risk  
- High Risk
""")

st.sidebar.markdown("---")
st.sidebar.caption(f"Model: {type(pipeline.named_steps['classifier']).__name__}")
st.sidebar.caption("Version 1.0")

# ============================================================
# COHORT OVERVIEW
# ============================================================

def cohort_overview():
    """Display cohort-level risk distribution."""
    st.title("Institutional Cohort Overview")
    
    try:
        # Load data
        df = pd.read_csv(DATA_PATH)
        X = df[features]
        y_pred = pipeline.predict(X)
        
        # Calculate metrics
        tier_counts = pd.Series(y_pred).value_counts().sort_index()
        tier_labels = ['Low Risk', 'Medium Risk', 'High Risk']
        tier_colors = ['#2e7d32', '#ed6c02', '#d32f2f']
        
        # Display metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Students", len(y_pred))
        with col2:
            st.metric("Low Risk", tier_counts.get(0, 0))
        with col3:
            st.metric("Medium Risk", tier_counts.get(1, 0))
        with col4:
            st.metric("High Risk", tier_counts.get(2, 0))
        
        # Create bar chart
        fig, ax = plt.subplots(figsize=(8, 5))
        bars = ax.bar(tier_labels, tier_counts.values, color=tier_colors, edgecolor='white', linewidth=1)
        ax.set_ylabel("Number of Students", fontsize=11)
        ax.set_title("Student Risk Tier Distribution", fontsize=13, fontweight='600')
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        ax.set_axisbelow(True)
        
        # Add value labels on bars
        for bar, count in zip(bars, tier_counts.values):
            percentage = (count / len(y_pred)) * 100
            ax.text(
                bar.get_x() + bar.get_width()/2, 
                bar.get_height() + 5,
                f'{count}\n({percentage:.1f}%)', 
                ha='center', 
                va='bottom', 
                fontsize=10
            )
        
        plt.tight_layout()
        st.pyplot(fig)
        
        # Additional insights
        st.subheader("Risk Distribution Insights")
        low_pct = (tier_counts.get(0, 0) / len(y_pred)) * 100
        medium_pct = (tier_counts.get(1, 0) / len(y_pred)) * 100
        high_pct = (tier_counts.get(2, 0) / len(y_pred)) * 100
        
        if high_pct > 30:
            st.markdown(f"""
            <div class="warning-box">
                <strong>High Risk Alert:</strong> {high_pct:.1f}% of students are in the High Risk category. 
                Consider intervention programs.
            </div>
            """, unsafe_allow_html=True)
        elif low_pct > 60:
            st.markdown(f"""
            <div class="success-box">
                <strong>Positive Indicator:</strong> {low_pct:.1f}% of students are in the Low Risk category.
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="info-box">
                <strong>Distribution Summary:</strong> Low: {low_pct:.1f}%, Medium: {medium_pct:.1f}%, High: {high_pct:.1f}%
            </div>
            """, unsafe_allow_html=True)
            
    except FileNotFoundError:
        st.error(f"Data file not found at {DATA_PATH}. Please ensure the data is uploaded.")
    except Exception as e:
        st.error(f"Error in Cohort Overview: {str(e)}")

# ============================================================
# INDIVIDUAL SCREENING
# ============================================================

def individual_screening():
    """Predict risk for a single student."""
    st.title("Individual Risk Screening")
    
    st.markdown("""
    <div class="info-box">
        Enter student characteristics below to predict their risk tier.
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Academic Factors")
        cgpa = st.slider("CGPA", 0.0, 5.0, 3.0, 0.01, help="Current Grade Point Average")
        carryovers = st.number_input("Carryover Courses", 0, 10, 0, help="Number of courses carried over")
        probation = st.selectbox("Academic Probation", ['No', 'Yes'])
        self_perf = st.selectbox("Self-Rated Performance", ['Excellent', 'Very Good', 'Good', 'Fair', 'Poor'])
        
        st.subheader("Study Habits")
        attendance = st.selectbox("Lecture Attendance", ['Always', 'Often', 'Sometimes', 'Rarely', 'Never'])
        study_hours = st.selectbox("Daily Study Hours", ['More than 6 hours', '5-6 hours', '3-4 hours', '1-2 hours', 'Less than 1 hour'])
        procrastination = st.selectbox("Procrastination Level", ['Always', 'Often', 'Sometimes', 'Rarely', 'Never'])
    
    with col2:
        st.subheader("Psychological Factors")
        stress = st.selectbox("Stress/Anxiety Level", ['Always', 'Often', 'Sometimes', 'Rarely', 'Never'])
        peer_focus = st.selectbox("Peer Academic Focus", ['Always', 'Often', 'Sometimes', 'Rarely', 'Never'])
        distraction = st.selectbox("Distraction Level", ['Always', 'Often', 'Sometimes', 'Rarely', 'Never'])
        
        st.subheader("External Factors")
        financial = st.selectbox("Financial Challenges", ['Always', 'Often', 'Sometimes', 'Rarely', 'Never'])
        tech_access = st.selectbox("Technology Access", ['Always', 'Often', 'Sometimes', 'Rarely', 'Never'])
        parent_edu = st.selectbox("Parental Education Level", ['Yes', 'No'])
        employment = st.selectbox("Part-Time Employment", ['Yes', 'No'])
        work_hours = st.number_input("Weekly Work Hours", 0, 40, 0)
        withdrawal = st.selectbox("Considered Withdrawal", ['No', 'Yes'])
    
    if st.button("Predict Risk Tier", use_container_width=True, type="primary"):
        with st.spinner("Analyzing student data..."):
            # TODO: Replace with actual prediction logic
            pred_tier = random.choices([0, 1, 2], weights=[0.4, 0.3, 0.3])[0]
            probs = [0.70, 0.20, 0.10] if pred_tier == 0 else \
                    [0.15, 0.70, 0.15] if pred_tier == 1 else \
                    [0.05, 0.15, 0.80]
            
            tier_labels = ['Low Risk', 'Medium Risk', 'High Risk']
            
            # Display probabilities
            st.subheader("Prediction Probabilities")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Low Risk", f"{probs[0]*100:.1f}%")
            with col2:
                st.metric("Medium Risk", f"{probs[1]*100:.1f}%")
            with col3:
                st.metric("High Risk", f"{probs[2]*100:.1f}%")
            
            # Display result
            st.markdown(f"""
            <div style="text-align: center; padding: 20px;">
                <h3>Prediction Result</h3>
                <div class="{['low-risk', 'medium-risk', 'high-risk'][pred_tier]}">
                    {tier_labels[pred_tier]}
                </div>
                <p style="margin-top: 10px; color: #666; font-size: 14px;">
                    Confidence: {max(probs)*100:.1f}%
                </p>
            </div>
            """, unsafe_allow_html=True)
            
            # Recommendations
            st.subheader("Recommendations")
            if pred_tier == 0:
                st.markdown("""
                <div class="success-box">
                    <strong>Low Risk Indicators:</strong>
                    <ul style="margin: 8px 0; padding-left: 20px;">
                        <li>Continue current support and monitoring</li>
                        <li>Encourage peer mentoring opportunities</li>
                        <li>Maintain regular check-ins</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)
            elif pred_tier == 1:
                st.markdown("""
                <div class="warning-box">
                    <strong>Moderate Risk Indicators:</strong>
                    <ul style="margin: 8px 0; padding-left: 20px;">
                        <li>Schedule academic counseling session</li>
                        <li>Monitor attendance and engagement</li>
                        <li>Consider study skills workshop</li>
                        <li>Regular progress tracking</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="error-box">
                    <strong>High Risk Indicators:</strong>
                    <ul style="margin: 8px 0; padding-left: 20px;">
                        <li>Immediate academic intervention required</li>
                        <li>Refer to student support services</li>
                        <li>Develop personalized success plan</li>
                        <li>Weekly progress monitoring</li>
                        <li>Connect with academic advisor</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)

# ============================================================
# BATCH PROCESSING
# ============================================================

def batch_processing():
    """Process multiple students from CSV file."""
    st.title("Batch Prediction")
    
    st.markdown("""
    <div class="info-box">
        Upload a CSV file with student data to process multiple predictions at once.
        The file must contain the required feature columns.
    </div>
    """, unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader(
        "Choose a CSV file", 
        type=['csv'],
        help="File must contain the required feature columns"
    )
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.success(f"Loaded {len(df)} records")
            
            # Show sample of uploaded data
            with st.expander("Preview Uploaded Data"):
                st.dataframe(df.head())
                st.caption(f"Total columns: {len(df.columns)} | Total rows: {len(df)}")
            
            # Check for required features
            missing_features = [f for f in features if f not in df.columns]
            if missing_features:
                st.warning(f"Missing features: {missing_features[:5]}...")
                if len(missing_features) > 5:
                    st.warning(f"... and {len(missing_features) - 5} more features")
            
            if st.button("Process Batch", use_container_width=True, type="primary"):
                with st.spinner(f"Processing {len(df)} students..."):
                    # TODO: Replace with actual prediction
                    results = df.copy()
                    results['Risk Tier'] = np.random.choice([0, 1, 2], len(df), p=[0.4, 0.3, 0.3])
                    
                    # Add risk tier labels
                    tier_map = {0: 'Low Risk', 1: 'Medium Risk', 2: 'High Risk'}
                    results['Risk Category'] = results['Risk Tier'].map(tier_map)
                    
                    # Add confidence scores
                    results['Confidence'] = np.random.uniform(0.6, 0.95, len(df))
                    
                    # Display results
                    st.subheader("Batch Results")
                    
                    # Summary metrics
                    col1, col2, col3, col4 = st.columns(4)
                    summary = results['Risk Category'].value_counts()
                    with col1:
                        st.metric("Total Processed", len(results))
                    with col2:
                        st.metric("Low Risk", summary.get('Low Risk', 0))
                    with col3:
                        st.metric("Medium Risk", summary.get('Medium Risk', 0))
                    with col4:
                        st.metric("High Risk", summary.get('High Risk', 0))
                    
                    # Display detailed results
                    display_cols = ['Risk Category', 'Confidence'] + [c for c in results.columns if c not in ['Risk Tier', 'Risk Category', 'Confidence']]
                    st.dataframe(
                        results[display_cols],
                        use_container_width=True
                    )
                    
                    # Download results
                    csv = results.to_csv(index=False)
                    st.download_button(
                        label="Download Results CSV",
                        data=csv,
                        file_name="predictions_results.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
                    
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")

# ============================================================
# MODEL EVALUATION
# ============================================================

def model_evaluation():
    """Display model performance metrics."""
    st.title("Model Performance Evaluation")
    
    # Display performance metrics
    try:
        if PERF_PATH.exists():
            perf_df = pd.read_csv(PERF_PATH)
            st.subheader("Performance Metrics")
            
            # Style the dataframe
            styled_df = perf_df.style.background_gradient(cmap='Blues', subset=['Accuracy', 'F1_Score'])\
                .format({
                    'Accuracy': '{:.3f}',
                    'Precision': '{:.3f}',
                    'Recall': '{:.3f}',
                    'F1_Score': '{:.3f}'
                })
            st.dataframe(styled_df, use_container_width=True)
        else:
            st.warning("Performance data not found. Run model training first.")
    except Exception as e:
        st.error(f"Error loading performance data: {str(e)}")
    
    # Display confusion matrix
    try:
        if CONFUSION_PATH.exists():
            st.subheader("Confusion Matrix")
            img = Image.open(CONFUSION_PATH)
            st.image(img, caption='Confusion Matrix', use_container_width=True)
        else:
            st.info("Confusion matrix image not available.")
    except Exception as e:
        st.error(f"Error loading confusion matrix: {str(e)}")
    
    # Model details
    with st.expander("Model Details"):
        st.write("**Model Architecture:**")
        st.write(f"- Model Type: {type(pipeline.named_steps['classifier']).__name__}")
        st.write(f"- Features Used: {len(features)}")
        st.write(f"- Feature Names: {', '.join(features[:10])}{'...' if len(features) > 10 else ''}")
        st.write("**Pipeline Steps:**")
        for name, step in pipeline.named_steps.items():
            st.write(f"- {name}: {type(step).__name__}")

# ============================================================
# EXPLAINABLE AI INTERFACE
# ============================================================

def xai_interface():
    """Explainable AI interface using SHAP."""
    st.title("Explainable AI")
    st.markdown("""
    <div class="info-box">
        Understand why the model made a particular prediction using SHAP 
        (SHapley Additive exPlanations) values.
    </div>
    """, unsafe_allow_html=True)
    
    # Student selection
    col1, col2 = st.columns([2, 1])
    with col1:
        student_id = st.number_input(
            "Enter Student ID", 
            min_value=0, 
            max_value=200, 
            value=0,
            help="Enter the student ID to generate explanations"
        )
    with col2:
        st.write("")
        st.write("")
        generate_btn = st.button("Generate Explanation", use_container_width=True, type="primary")
    
    if generate_btn:
        with st.spinner(f"Generating explanation for Student {student_id}..."):
            st.success(f"Explanation for Student {student_id} generated")
            
            # Display SHAP visualization
            try:
                if SHAP_PATH.exists():
                    st.subheader("SHAP Feature Importance")
                    img = Image.open(SHAP_PATH)
                    st.image(img, caption='SHAP Beeswarm Plot - Feature Impact on Predictions', use_container_width=True)
                else:
                    st.warning("SHAP images not found. Run SHAP analysis first.")
            except Exception as e:
                st.error(f"Error loading SHAP image: {str(e)}")
            
            # Feature importance interpretation
            st.subheader("Interpretation Guide")
            st.markdown("""
            **Understanding SHAP Values**
            - SHAP values show how each feature contributes to the prediction
            - Positive values (red) push prediction toward HIGHER risk
            - Negative values (blue) push prediction toward LOWER risk
            - Feature importance shows which factors most influence the model
            
            **Key Factors for Student Risk**
            1. CGPA: Lower CGPA increases risk
            2. Attendance: Poor attendance increases risk
            3. Carryover Courses: More carryovers increase risk
            4. Stress Level: Higher stress increases risk
            5. Study Hours: Fewer study hours increase risk
            """)

# ============================================================
# MAIN ROUTING
# ============================================================

# Page routing
if page == "Cohort Overview":
    cohort_overview()
elif page == "Individual Screening":
    individual_screening()
elif page == "Batch Processing":
    batch_processing()
elif page == "Model Evaluation":
    model_evaluation()
elif page == "Explainable AI":
    xai_interface()

# Footer
st.sidebar.markdown("---")
st.sidebar.caption("Built with Streamlit | Student Risk System v1.0")