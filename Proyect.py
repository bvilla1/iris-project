import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_score,
                             recall_score, f1_score, confusion_matrix)
from sklearn.preprocessing import StandardScaler

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Iris Species Classifier",
    page_icon="🌸",
    layout="wide"
)

# ─────────────────────────────────────────────
# ESTILOS CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #f8f9fa; }
    .stMetric {
        background-color: white;
        border-radius: 12px;
        padding: 15px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
    }
    h1 { color: #2d6a4f; }
    h2, h3 { color: #40916c; }
    .prediction-box {
        background: linear-gradient(135deg, #d8f3dc, #b7e4c7);
        border-radius: 15px;
        padding: 25px;
        text-align: center;
        font-size: 1.5rem;
        font-weight: bold;
        color: #1b4332;
        margin: 15px 0;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
    }
    .section-header {
        background: linear-gradient(90deg, #2d6a4f, #52b788);
        color: white;
        padding: 10px 20px;
        border-radius: 8px;
        margin-bottom: 15px;
    }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# CARGAR Y PREPARAR DATOS
# ─────────────────────────────────────────────
@st.cache_data
def load_and_train():
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df['species'] = [iris.target_names[t] for t in iris.target]
    df['target'] = iris.target

    X = df[iris.feature_names]
    y = df['target']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train_sc, y_train)
    y_pred = model.predict(X_test_sc)

    metrics = {
        "Accuracy":  round(accuracy_score(y_test, y_pred), 4),
        "Precision": round(precision_score(y_test, y_pred, average='weighted'), 4),
        "Recall":    round(recall_score(y_test, y_pred, average='weighted'), 4),
        "F1 Score":  round(f1_score(y_test, y_pred, average='weighted'), 4),
    }
    cm = confusion_matrix(y_test, y_pred)
    return model, scaler, df, iris.feature_names, iris.target_names, metrics, cm

model, scaler, df, feature_names, target_names, metrics, cm = load_and_train()

# ─────────────────────────────────────────────
# TÍTULO
# ─────────────────────────────────────────────
st.markdown("<h1 style='text-align:center'>🌸 Iris Species Classification Dashboard</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center; color:#555'>Data Mining Final Project — Universidad de la Costa</p>", unsafe_allow_html=True)
st.markdown("---")

# ─────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["📊 Model Metrics", "🔍 Predict Species", "📈 Data Exploration"])

# ══════════════════════════════════════════════
# TAB 1 — MÉTRICAS
# ══════════════════════════════════════════════
with tab1:
    st.markdown("<div class='section-header'><h3 style='color:white;margin:0'>Model Performance — Random Forest</h3></div>", unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)
    icons = ["🎯", "🔬", "📡", "⚖️"]
    for col, (name, val), icon in zip([col1,col2,col3,col4], metrics.items(), icons):
        col.metric(f"{icon} {name}", f"{val*100:.2f}%")

    st.markdown("---")

    # Confusion Matrix
    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("Confusion Matrix")
        fig_cm, ax = plt.subplots(figsize=(5,4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Greens',
                    xticklabels=target_names, yticklabels=target_names, ax=ax)
        ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
        ax.set_title("Confusion Matrix")
        st.pyplot(fig_cm)

    with col_b:
        st.subheader("Feature Importance")
        importances = model.feature_importances_
        feat_df = pd.DataFrame({
            'Feature': [f.replace(' (cm)','') for f in feature_names],
            'Importance': importances
        }).sort_values('Importance', ascending=True)
        fig_imp = px.bar(feat_df, x='Importance', y='Feature', orientation='h',
                         color='Importance', color_continuous_scale='Greens',
                         title="Which features matter most?")
        fig_imp.update_layout(showlegend=False)
        st.plotly_chart(fig_imp, use_container_width=True)

    st.markdown("---")
    st.subheader("ℹ️ Why Random Forest?")
    st.info("""
    **Random Forest** was chosen because:
    - ✅ Works great with small datasets like Iris (150 samples)
    - ✅ Handles multi-class classification natively
    - ✅ Resistant to overfitting (uses many decision trees)
    - ✅ Provides feature importance scores
    - ✅ No strong assumptions about data distribution
    """)

# ══════════════════════════════════════════════
# TAB 2 — PREDICCIÓN
# ══════════════════════════════════════════════
with tab2:
    st.markdown("<div class='section-header'><h3 style='color:white;margin:0'>🔍 Predict a New Flower</h3></div>", unsafe_allow_html=True)
    st.write("Enter the flower measurements below:")

    col1, col2 = st.columns(2)
    with col1:
        sepal_length = st.slider("🌿 Sepal Length (cm)", 4.0, 8.0, 5.8, 0.1)
        sepal_width  = st.slider("🌿 Sepal Width (cm)",  2.0, 4.5, 3.0, 0.1)
    with col2:
        petal_length = st.slider("🌺 Petal Length (cm)", 1.0, 7.0, 4.0, 0.1)
        petal_width  = st.slider("🌺 Petal Width (cm)",  0.1, 2.5, 1.2, 0.1)

    if st.button("🔮 Predict Species", type="primary", use_container_width=True):
        input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
        input_scaled = scaler.transform(input_data)
        prediction = model.predict(input_scaled)[0]
        probabilities = model.predict_proba(input_scaled)[0]
        species_name = target_names[prediction]

        emoji_map = {"setosa": "🌸", "versicolor": "💜", "virginica": "🌺"}
        emoji = emoji_map.get(species_name, "🌼")

        st.markdown(f"<div class='prediction-box'>{emoji} Predicted Species: <span style='color:#1b4332'>Iris {species_name.capitalize()}</span></div>",
                    unsafe_allow_html=True)

        # Probabilidades
        st.subheader("Prediction Confidence")
        prob_df = pd.DataFrame({'Species': target_names, 'Probability': probabilities})
        fig_prob = px.bar(prob_df, x='Species', y='Probability',
                          color='Species',
                          color_discrete_sequence=['#95d5b2','#52b788','#2d6a4f'],
                          title="Probability per Species")
        fig_prob.update_yaxes(range=[0,1])
        st.plotly_chart(fig_prob, use_container_width=True)

        # Gráfico 3D
        st.subheader("3D Position of Your Sample vs Dataset")
        fig_3d = px.scatter_3d(
            df,
            x='petal length (cm)', y='petal width (cm)', z='sepal length (cm)',
            color='species',
            color_discrete_sequence=['#95d5b2','#52b788','#1b4332'],
            opacity=0.6,
            title="Your sample (⭐) vs the full dataset"
        )
        fig_3d.add_trace(go.Scatter3d(
            x=[petal_length], y=[petal_width], z=[sepal_length],
            mode='markers',
            marker=dict(size=12, color='red', symbol='diamond'),
            name="⭐ Your Sample"
        ))
        fig_3d.update_layout(height=500)
        st.plotly_chart(fig_3d, use_container_width=True)

# ══════════════════════════════════════════════
# TAB 3 — EXPLORACIÓN
# ══════════════════════════════════════════════
with tab3:
    st.markdown("<div class='section-header'><h3 style='color:white;margin:0'>📈 Data Exploration</h3></div>", unsafe_allow_html=True)

    st.subheader("Dataset Overview")
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Samples", "150")
    col2.metric("Features", "4")
    col3.metric("Species", "3")

    st.dataframe(df.drop('target', axis=1).head(10), use_container_width=True)

    st.markdown("---")

    # Histogramas
    st.subheader("Feature Distributions by Species")
    feature_choice = st.selectbox("Select a feature:", feature_names)
    fig_hist = px.histogram(df, x=feature_choice, color='species',
                             barmode='overlay', nbins=20,
                             color_discrete_sequence=['#95d5b2','#52b788','#1b4332'],
                             title=f"Distribution of {feature_choice}")
    st.plotly_chart(fig_hist, use_container_width=True)

    st.markdown("---")

    # Scatter Matrix
    st.subheader("Scatter Matrix (All Features)")
    fig_scatter = px.scatter_matrix(
        df,
        dimensions=list(feature_names),
        color='species',
        color_discrete_sequence=['#95d5b2','#52b788','#1b4332'],
        title="Relationships between all features"
    )
    fig_scatter.update_traces(diagonal_visible=False)
    fig_scatter.update_layout(height=600)
    st.plotly_chart(fig_scatter, use_container_width=True)

    # Boxplot
    st.subheader("Boxplot by Species")
    feat_box = st.selectbox("Select feature for boxplot:", feature_names, key='box')
    fig_box = px.box(df, x='species', y=feat_box, color='species',
                     color_discrete_sequence=['#95d5b2','#52b788','#1b4332'],
                     title=f"{feat_box} by Species")
    st.plotly_chart(fig_box, use_container_width=True)

# ─────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────
st.markdown("---")
st.markdown("<p style='text-align:center; color:#aaa'>🌸 Iris Classification Project — Data Mining — Universidad de la Costa</p>",
            unsafe_allow_html=True)
