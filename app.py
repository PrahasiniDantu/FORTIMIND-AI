import streamlit as st
import pandas as pd
import numpy as np
import time
import json

# Setup Page
st.set_page_config(page_title="FortiMind AI", page_icon="🛡️", layout="wide")

# Custom CSS for Background and Glassmorphism
st.markdown("""
<style>
/* Full screen background */
[data-testid="stAppViewContainer"] {
    background-image: linear-gradient(135deg, rgba(30,60,114,0.7), rgba(42,82,152,0.7)), url("https://images.unsplash.com/photo-1620712943543-bcc4688e7485?q=80&w=2000&auto=format&fit=crop");
    background-size: cover;
    background-position: center;
    background-attachment: fixed;
}
/* Transparent header */
[data-testid="stHeader"] {
    background: transparent;
}
/* Sidebar styling */
[data-testid="stSidebar"] {
    background-color: rgba(15, 23, 42, 0.85);
    backdrop-filter: blur(15px);
}
/* Global text color adjustment for dark theme */
.css-1g8v9l0, .css-10tr5yj, .css-1qg05tj, .stMarkdown, .stText, h1, h2, h3, p, label {
    color: #F8FAFC !important;
}

/* Glassmorphism Cards */
.glass-card {
    background: rgba(255, 255, 255, 0.08);
    backdrop-filter: blur(12px);
    -webkit-backdrop-filter: blur(12px);
    border-radius: 12px;
    border: 1px solid rgba(255, 255, 255, 0.15);
    padding: 24px;
    margin-bottom: 20px;
    color: #ffffff;
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.2);
    transition: transform 0.2s ease-in-out;
}
.glass-card:hover {
    transform: translateY(-2px);
    background: rgba(255, 255, 255, 0.12);
}
.glass-card-high { border-left: 6px solid #FF4B4B; }
.glass-card-mod { border-left: 6px solid #FFA500; }
.glass-card-low { border-left: 6px solid #00C853; }

/* Gradient text */
.gradient-text {
    background: linear-gradient(90deg, #6EE7B7 0%, #3B82F6 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-weight: 800;
    font-size: 3.5rem;
    margin-bottom: 0.2rem;
    padding-bottom: 0.2rem;
}
.subtitle-text {
    font-size: 1.2rem;
    color: #cbd5e1;
    margin-bottom: 2rem;
    font-weight: 300;
}
.footer-text {
    text-align: center;
    color: #94a3b8;
    margin-top: 50px;
    font-size: 0.9rem;
}
</style>
""", unsafe_allow_html=True)

# Session State for History
if "history" not in st.session_state:
    st.session_state.history = []

# --- Sidebar ---
with st.sidebar:
    st.title("🛡️ FortiMind AI")
    st.markdown("---")
    
    st.subheader("🕰️ Recent Analyses")
    if len(st.session_state.history) == 0:
        st.info("No history yet.")
    else:
        for idx, item in enumerate(reversed(st.session_state.history[-3:])):
            with st.expander(f"{item['filename']} - {item['time']}"):
                st.write(f"**Target:** {item['target']}")
                st.write(f"**Max Disparity:** {item['max_bias']:.2f}")
                st.write(f"**Top Risk:** {item['top_feature']}")
    
    st.markdown("---")
    st.subheader("💡 AI Insights")
    insights = [
        "AI bias can impact hiring, loans, and healthcare decisions.",
        "Balanced datasets lead to more reliable predictions.",
        "Transparency improves trust in AI systems."
    ]
    for insight in insights:
        st.markdown(f"> {insight}")

# --- Main UI ---
st.markdown('<div class="gradient-text">FortiMind AI</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle-text">Intelligent auditing for fair and trustworthy AI decisions</div>', unsafe_allow_html=True)

# Greeting
if 'greeted' not in st.session_state:
    st.toast("👋 Welcome to FortiMind AI! Let’s analyze your data intelligently.")
    st.session_state.greeted = True

uploaded_file = st.file_uploader("Upload your dataset", type=["csv", "xlsx", "xls", "json", "txt"])

if uploaded_file is not None:
    # 1. Load Data
    try:
        filename = uploaded_file.name
        if filename.endswith(".csv") or filename.endswith(".txt"):
            df = pd.read_csv(uploaded_file)
        elif filename.endswith((".xlsx", ".xls")):
            df = pd.read_excel(uploaded_file)
        elif filename.endswith(".json"):
            df = pd.read_json(uploaded_file)
        else:
            st.error("Unsupported file format.")
            st.stop()
            
        with st.expander("Dataset Preview", expanded=False):
            st.dataframe(df.head(10), use_container_width=True)
            
        # 2. Ask for Target Column
        target_col = st.selectbox("Select Target Column (Outcome to analyze for bias)", options=["-- Select --"] + list(df.columns))
        
        if target_col != "-- Select --":
            # 3. Dynamic Messages
            msg_placeholder = st.empty()
            messages = [
                "Analyzing patterns in your data...",
                "Detecting hidden disparities...",
                "AI systems can inherit bias from data patterns...",
                "Fairness is critical in decision-making systems...",
                "Generating meaningful insights..."
            ]
            
            for i in range(5):
                msg_placeholder.info(f"🤖 {messages[i]}")
                time.sleep(0.5)
            msg_placeholder.empty()

            # 4. Target Processing
            analysis_df = df.copy()
            target_series = analysis_df[target_col]
            
            # Convert target to binary
            is_valid_target = False
            if pd.api.types.is_numeric_dtype(target_series):
                median_val = target_series.median()
                analysis_df['__target_binary__'] = (target_series > median_val).astype(int)
                target_desc = f"Numeric (threshold > median: {median_val:.2f})"
                is_valid_target = True
            else:
                unique_vals = target_series.dropna().unique()
                if len(unique_vals) == 2:
                    # Convert to 0 and 1
                    val0, val1 = unique_vals[0], unique_vals[1]
                    analysis_df['__target_binary__'] = (target_series == val1).astype(int)
                    target_desc = f"Categorical ({val1} vs {val0})"
                    is_valid_target = True
                else:
                    st.error(f"Target column must be numeric or a binary categorical variable. '{target_col}' has {len(unique_vals)} unique values.")
            
            if is_valid_target:
                # 5. Bias Detection Logic
                results = []
                features = [c for c in analysis_df.columns if c not in [target_col, '__target_binary__']]
                
                for feature in features:
                    feat_series = analysis_df[feature]
                    
                    if feat_series.isnull().all():
                        continue
                        
                    # Skip identifiers / text
                    if feat_series.nunique() > 50 and pd.api.types.is_object_dtype(feat_series):
                        continue
                        
                    # Binning for numeric
                    if pd.api.types.is_numeric_dtype(feat_series) and feat_series.nunique() > 5:
                        try:
                            # Try quantiles
                            binned = pd.qcut(feat_series, q=3, labels=["Low", "Medium", "High"], duplicates='drop')
                        except ValueError:
                            # Fallback to cut
                            binned = pd.cut(feat_series, bins=3, labels=["Low", "Medium", "High"])
                        group_col = binned
                    else:
                        group_col = feat_series.fillna("Unknown").astype(str)
                    
                    # Calculate rates
                    rates = analysis_df.groupby(group_col)['__target_binary__'].mean()
                    rates = rates.dropna()
                    if len(rates) > 1:
                        max_rate = rates.max()
                        min_rate = rates.min()
                        disparity = max_rate - min_rate
                        
                        risk = "Low 🟢"
                        css_class = "glass-card-low"
                        if disparity > 0.5:
                            risk = "High Risk 🔴"
                            css_class = "glass-card-high"
                        elif disparity > 0.2:
                            risk = "Moderate 🟠"
                            css_class = "glass-card-mod"
                            
                        results.append({
                            "feature": feature,
                            "disparity": disparity,
                            "risk": risk,
                            "css_class": css_class,
                            "rates": rates.to_dict()
                        })
                
                # Sort by disparity descending
                results = sorted(results, key=lambda x: x["disparity"], reverse=True)
                
                # Update History
                if results:
                    top_feature = results[0]["feature"]
                    max_bias = results[0]["disparity"]
                else:
                    top_feature = "None"
                    max_bias = 0.0
                    
                # Prevent duplicate consecutive history entries
                current_time = time.strftime("%H:%M:%S")
                if not st.session_state.history or st.session_state.history[-1]['filename'] != filename or st.session_state.history[-1]['target'] != target_col:
                    st.session_state.history.append({
                        "filename": filename,
                        "time": current_time,
                        "target": target_col,
                        "max_bias": max_bias,
                        "top_feature": top_feature
                    })

                # 6. Render Results
                st.markdown("### 📊 Bias Analysis Results")
                st.markdown(f"**Target Variable Setup:** {target_desc}")
                
                if not results:
                    st.info("No suitable features found for bias analysis.")
                else:
                    # Highlight most biased
                    most_biased = results[0]
                    st.markdown(f"""
                    <div class="glass-card {most_biased['css_class']}">
                        <h3 style="margin-top:0;">⚠️ Top Bias Risk: {most_biased['feature']}</h3>
                        <p style="font-size: 1.2rem;">Disparity: <b>{most_biased['disparity']:.3f}</b> | Risk Level: <b>{most_biased['risk']}</b></p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    st.markdown("#### Feature Breakdown")
                    cols = st.columns(2)
                    for i, res in enumerate(results):
                        if i == 0: continue # Skip first as it is highlighted above
                        
                        rates_html = "".join([f"<li>{k}: {v:.2%}</li>" for k,v in res['rates'].items()])
                        
                        card_html = f"""
                        <div class="glass-card {res['css_class']}">
                            <h4 style="margin-top:0;">{res['feature']}</h4>
                            <p>Disparity: <b>{res['disparity']:.3f}</b> <br> Risk Level: {res['risk']}</p>
                            <details>
                                <summary>View Group Rates</summary>
                                <ul>{rates_html}</ul>
                            </details>
                        </div>
                        """
                        cols[(i-1) % 2].markdown(card_html, unsafe_allow_html=True)

                    # 7. AI Explanation
                    st.markdown("### 🧠 AI Explanation")
                    
                    explanation_text = f"""
                    **Why does this bias exist?**  
                    In this dataset, the feature **{most_biased['feature']}** showed the highest disparity ({most_biased['disparity']:.2f}) across its groups. 
                    AI systems learn directly from historical patterns. If certain groups historically experience different outcomes, the AI will internalize and potentially amplify this behavior, leading to systemic bias.
                    
                    **What does it mean?**  
                    A disparity of {most_biased['disparity']:.2f} means there is a significant difference in the likelihood of the target outcome depending solely on a person's {most_biased['feature']}. This represents a **{most_biased['risk'].split()[0]}** risk of unfairness if a model is trained on this data without intervention.
                    
                    **How to fix it?**  
                    - **Data Collection:** Ensure balanced representation across all groups in {most_biased['feature']}.
                    - **Preprocessing:** Apply resampling techniques or fairness-aware algorithms (like reweighing) to mitigate the disparity before training.
                    - **Exclusion:** If {most_biased['feature']} is a protected attribute (e.g., race, gender) and not strictly necessary for the task, consider omitting it, though proxy variables may still leak this information.
                    """
                    
                    st.markdown(f"""
                    <div class="glass-card" style="border-left: 6px solid #3B82F6;">
                        {explanation_text}
                    </div>
                    """, unsafe_allow_html=True)

    except Exception as e:
        st.error(f"An error occurred: {str(e)}")

# Footer
st.markdown('<div class="footer-text">Making AI decisions transparent and accountable</div>', unsafe_allow_html=True)
