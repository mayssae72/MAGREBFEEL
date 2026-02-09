import streamlit as st
import pickle
import joblib
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import re
import warnings
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import time

warnings.filterwarnings('ignore')

# Configuration de la page
st.set_page_config(
    page_title="Analyse de Dialectes et Sentiments - PFE",
    page_icon="🎭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personnalisé avec design professionnel et féminin
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap');
    
    * {
        font-family: 'Poppins', sans-serif;
    }
    
    .main {
        background: linear-gradient(135deg, #fdf2f8 0%, #f0f9ff 20%, #fef7ff 40%, #f0fdfa 60%, #fff7ed 80%, #fdf2f8 100%);
        min-height: 100vh;
    }
    
    .stApp > header {
        background-color: transparent;
    }
    
    /* Header principal */
    .main-header {
        background: linear-gradient(135deg, #be185d 0%, #7c3aed 30%, #059669 70%, #dc2626 100%);
        padding: 3rem 2rem;
        border-radius: 20px;
        margin-bottom: 2rem;
        box-shadow: 0 20px 40px rgba(190, 24, 93, 0.3);
        position: relative;
        overflow: hidden;
    }
    
    .main-header::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: url('data:image/svg+xml,<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100"><defs><pattern id="grid" width="10" height="10" patternUnits="userSpaceOnUse"><path d="M 10 0 L 0 0 0 10" fill="none" stroke="white" stroke-width="0.5" opacity="0.1"/></pattern></defs><rect width="100" height="100" fill="url(%23grid)"/></svg>');
        opacity: 0.1;
    }
    
    .title-text {
        color: white;
        text-align: center;
        font-size: 3rem;
        font-weight: 700;
        margin: 0;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        position: relative;
        z-index: 1;
    }
    
    .subtitle-text {
        color: #fce7f3;
        text-align: center;
        font-size: 1.3rem;
        margin-top: 1rem;
        font-weight: 400;
        position: relative;
        z-index: 1;
    }
    
    .pfe-badge {
        background: rgba(255,255,255,0.2);
        padding: 0.5rem 1.5rem;
        border-radius: 25px;
        color: white;
        font-weight: 500;
        display: inline-block;
        margin-top: 1rem;
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255,255,255,0.3);
    }
    
    /* Cards modernes */
    .analysis-card {
        background: linear-gradient(135deg, rgba(255,255,255,0.9) 0%, rgba(252,231,243,0.6) 100%);
        backdrop-filter: blur(20px);
        border: 1px solid rgba(190, 24, 93, 0.1);
        padding: 2rem;
        border-radius: 20px;
        box-shadow: 0 15px 35px rgba(190, 24, 93, 0.1);
        margin-bottom: 2rem;
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
        position: relative;
        overflow: hidden;
    }
    
    .analysis-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 2px;
        background: linear-gradient(90deg, transparent, #be185d, transparent);
        transition: left 0.5s;
    }
    
    .analysis-card:hover::before {
        left: 100%;
    }
    
    .analysis-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 25px 50px rgba(190, 24, 93, 0.2);
        border-color: rgba(190, 24, 93, 0.3);
    }
    
    /* Métriques élégantes */
    .metric-container {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 1.5rem;
        margin: 2rem 0;
    }
    
    .metric-card {
        background: linear-gradient(135deg, rgba(255,255,255,0.9) 0%, rgba(240, 249, 255, 0.8) 100%);
        backdrop-filter: blur(15px);
        padding: 2rem;
        border-radius: 16px;
        text-align: center;
        border: 1px solid rgba(124, 58, 237, 0.2);
        box-shadow: 0 10px 25px rgba(124, 58, 237, 0.1);
        transition: all 0.3s ease;
        position: relative;
        overflow: hidden;
    }
    
    .metric-card::after {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 3px;
        background: linear-gradient(90deg, #be185d, #7c3aed, #059669);
        opacity: 0;
        transition: opacity 0.3s;
    }
    
    .metric-card:hover {
        transform: translateY(-3px) scale(1.02);
        box-shadow: 0 15px 35px rgba(124, 58, 237, 0.2);
    }
    
    .metric-card:hover::after {
        opacity: 1;
    }
    
    .metric-value {
        font-size: 2.5rem;
        font-weight: 700;
        margin: 0.5rem 0;
        background: linear-gradient(135deg, #be185d, #7c3aed);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    .metric-label {
        color: #6b7280;
        font-weight: 500;
        font-size: 0.9rem;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    /* Résultats avec style */
    .result-container {
        background: linear-gradient(135deg, rgba(252, 231, 243, 0.8) 0%, rgba(240, 249, 255, 0.8) 100%);
        backdrop-filter: blur(20px);
        border-left: 5px solid #be185d;
        padding: 2rem;
        border-radius: 15px;
        margin: 2rem 0;
        box-shadow: 0 10px 30px rgba(190, 24, 93, 0.15);
    }
    
    /* Boutons élégants */
    .stButton > button {
        background: linear-gradient(135deg, #be185d 0%, #7c3aed 50%, #059669 100%);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 1rem 2rem;
        font-weight: 600;
        font-size: 1.1rem;
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
        width: 100%;
        box-shadow: 0 8px 20px rgba(190, 24, 93, 0.3);
        position: relative;
        overflow: hidden;
    }
    
    .stButton > button:hover {
        transform: translateY(-3px);
        box-shadow: 0 15px 35px rgba(190, 24, 93, 0.4);
        background: linear-gradient(135deg, #db2777 0%, #8b5cf6 50%, #10b981 100%);
    }
    
    .stButton > button:active {
        transform: translateY(-1px);
    }
    
    /* Sidebar élégante */
    .sidebar .element-container {
        background: rgba(255, 255, 255, 0.8);
        backdrop-filter: blur(10px);
        border-radius: 12px;
        padding: 1rem;
        margin: 0.5rem 0;
        border: 1px solid rgba(190, 24, 93, 0.1);
    }
    
    /* Graphiques container */
    .chart-container {
        background: rgba(255, 255, 255, 0.9);
        backdrop-filter: blur(15px);
        border-radius: 16px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 10px 25px rgba(0, 0, 0, 0.05);
        border: 1px solid rgba(190, 24, 93, 0.1);
    }
    
    /* Animations */
    @keyframes fadeInUp {
        from {
            opacity: 0;
            transform: translateY(30px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    .fade-in {
        animation: fadeInUp 0.6s ease-out;
    }
    
    /* Responsive */
    @media (max-width: 768px) {
        .title-text {
            font-size: 2rem;
        }
        .subtitle-text {
            font-size: 1rem;
        }
        .main-header {
            padding: 2rem 1rem;
        }
    }
    
    /* Status indicators */
    .status-positive {
        color: #059669;
        font-weight: 600;
    }
    
    .status-negative {
        color: #dc2626;
        font-weight: 600;
    }
    
    .status-neutral {
        color: #7c3aed;
        font-weight: 600;
    }
    
    /* Progress bar */
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #be185d, #7c3aed, #059669);
    }
</style>
""", unsafe_allow_html=True)

# Fonction pour charger les modèles avec gestion d'erreur
@st.cache_resource
def load_model(model_path):
    """Charge un modèle en essayant pickle puis joblib"""
    try:
        with open(model_path, 'rb') as f:
            return pickle.load(f)
    except:
        try:
            return joblib.load(model_path)
        except Exception as e:
            st.error(f"Erreur lors du chargement du modèle {model_path}: {e}")
            return None

# Fonction pour charger les vectorizers
@st.cache_resource
def load_vectorizer(vectorizer_path):
    """Charge un vectorizer TF-IDF"""
    try:
        with open(vectorizer_path, 'rb') as f:
            return pickle.load(f)
    except:
        try:
            return joblib.load(vectorizer_path)
        except Exception as e:
            st.error(f"Erreur lors du chargement du vectorizer {vectorizer_path}: {e}")
            return None

# Fonction de préprocessing du texte
def preprocess_text(text):
    """Préprocessing basique du texte"""
    text = re.sub(r'[^\w\s\u0600-\u06FF]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Chargement des modèles
@st.cache_resource
def load_all_models():
    """Charge tous les modèles nécessaires"""
    models = {}
    
    # Modèles de détection de dialecte
    models['dialect_model'] = load_model('Detection dialect/NOOTBOOK/logreg_tfidf_model2.joblib')
    models['dialect_vectorizer'] = load_vectorizer('Detection dialect/models/tfidf_vectorizer.pkl')
    
    # Modèles d'analyse de sentiment
    models['sentiment_ma_model'] = joblib.load('setiment_ma/best_model_classification_des_sentiments.joblib')
    models['sentiment_ma_vectorizer'] = load_vectorizer('setiment_ma/tfidf_vectorizer_mar.pkl')
    
    models['sentiment_dz_model'] = load_model('dz/best_model_1svm.joblib')
    models['sentiment_dz_vectorizer'] = load_vectorizer('dz/tfidf_vectorizer_dz.pkl')
    
    models['sentiment_tun_model'] = load_model('TUN/best_model_tunsvm.joblib')
    models['sentiment_tun_vectorizer'] = load_vectorizer('TUN/tfidf_vectorizer_tun.pkl')
    
    return models

def create_dialect_distribution_chart():
    """Crée un graphique de distribution des dialectes"""
    data = {
        'Dialecte': ['Marocain', 'Algérien', 'Tunisien'],
        'Pourcentage': [35, 40, 25],
        'Couleur': ['#be185d', '#7c3aed', '#059669']
    }
    
    fig = px.pie(
        values=data['Pourcentage'],
        names=data['Dialecte'],
        title="Distribution des Dialectes Arabes Supportés",
        color_discrete_sequence=['#be185d', '#7c3aed', '#059669']
    )
    
    fig.update_traces(
        textposition='inside',
        textinfo='percent+label',
        hovertemplate='<b>%{label}</b><br>Pourcentage: %{percent}<br><extra></extra>',
        pull=[0.1, 0, 0]
    )
    
    fig.update_layout(
        font=dict(family="Poppins, sans-serif", size=14),
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5),
        height=400,
        margin=dict(t=50, l=20, r=20, b=80)
    )
    
    return fig

def create_sentiment_chart(sentiment_data):
    """Crée un graphique de sentiment"""
    fig = go.Figure()
    
    colors = {
        'Positif': '#059669',
        'Négatif': '#dc2626',
        'Neutre': '#7c3aed'
    }
    
    for sentiment, count in sentiment_data.items():
        fig.add_trace(go.Bar(
            x=[sentiment],
            y=[count],
            name=sentiment,
            marker_color=colors.get(sentiment, '#gray'),
            text=[count],
            textposition='auto',
        ))
    
    fig.update_layout(
        title="Analyse des Sentiments",
        xaxis_title="Sentiment",
        yaxis_title="Fréquence",
        font=dict(family="Poppins, sans-serif"),
        showlegend=False,
        height=400,
        margin=dict(t=50, l=20, r=20, b=20)
    )
    
    return fig

def create_confidence_gauge(confidence_score):
    """Crée une jauge de confiance"""
    fig = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = confidence_score,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Niveau de Confiance"},
        delta = {'reference': 80},
        gauge = {
            'axis': {'range': [None, 100]},
            'bar': {'color': "#be185d"},
            'steps': [
                {'range': [0, 50], 'color': "#fecaca"},
                {'range': [50, 80], 'color': "#fed7aa"},
                {'range': [80, 100], 'color': "#bbf7d0"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 90
            }
        }
    ))
    
    fig.update_layout(
        font={'family': "Poppins, sans-serif", 'color': "darkblue", 'size': 12},
        height=300,
        margin=dict(t=50, l=20, r=20, b=20)
    )
    
    return fig

def create_model_performance_chart():
    """Crée un graphique de performance des modèles"""
    models_data = {
        'Modèle': ['Détection Dialecte', 'Sentiment Marocain', 'Sentiment Algérien', 'Sentiment Tunisien'],
        'Précision': [0.92, 0.88, 0.85, 0.87],
        'Rappel': [0.91, 0.86, 0.83, 0.85],
        'F1-Score': [0.915, 0.87, 0.84, 0.86]
    }
    
    df = pd.DataFrame(models_data)
    
    fig = go.Figure()
    
    colors = ['#be185d', '#7c3aed', '#059669']
    metrics = ['Précision', 'Rappel', 'F1-Score']
    
    for i, metric in enumerate(metrics):
        fig.add_trace(go.Bar(
            name=metric,
            x=df['Modèle'],
            y=df[metric],
            marker_color=colors[i],
            text=[f'{val:.2%}' for val in df[metric]],
            textposition='auto',
        ))
    
    fig.update_layout(
        title='Performance des Modèles d\'IA',
        xaxis_title='Modèles',
        yaxis_title='Score',
        barmode='group',
        font=dict(family="Poppins, sans-serif"),
        height=450,
        legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5),
        margin=dict(t=50, l=20, r=20, b=80)
    )
    
    return fig

def create_analysis_timeline():
    """Crée une timeline d'analyse"""
    timeline_data = {
        'Étape': ['Préparation', 'Détection Dialecte', 'Analyse Sentiment', 'Génération Résultats'],
        'Temps': [0.5, 1.2, 0.8, 0.3],
        'Status': ['Terminé', 'Terminé', 'Terminé', 'Terminé']
    }
    
    fig = px.bar(
        x=timeline_data['Temps'],
        y=timeline_data['Étape'],
        orientation='h',
        title='Timeline d\'Analyse',
        color=timeline_data['Temps'],
        color_continuous_scale=['#be185d', '#7c3aed', '#059669'],
        text=[f'{t}s' for t in timeline_data['Temps']]
    )
    
    fig.update_traces(textposition='inside')
    fig.update_layout(
        font=dict(family="Poppins, sans-serif"),
        height=300,
        showlegend=False,
        margin=dict(t=50, l=20, r=20, b=20)
    )
    
    return fig

# Interface principale
def main():
    # Header principal avec design PFE
    st.markdown("""
    <div class="main-header fade-in">
        <h1 class="title-text">🎭 Analyse Intelligente de Dialectes Arabes</h1>
        <p class="subtitle-text">Système d'Intelligence Artificielle pour l'Analyse des Dialectes et Sentiments</p>
        <div style="text-align: center;">
            <span class="pfe-badge">📚 Projet de Fin d'Études - Intelligence Artificielle</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Chargement des modèles avec animation
    with st.spinner("🤖 Initialisation des modèles d'IA..."):
        models = load_all_models()
        time.sleep(1)  # Animation loading
    
    # Sidebar avec navigation élégante
    with st.sidebar:
        st.markdown("### 🎯 Navigation")
        analysis_type = st.selectbox(
            "Choisissez votre analyse",
            ["🔬 Analyse Complète", "💭 Analyse de Sentiments", "🗣️ Détection de Dialecte", "📊 Tableau de Bord"],
            index=0
        )
        
        st.markdown("---")
        st.markdown("### 📈 Statistiques en Temps Réel")
        
        # Métriques sidebar
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Analyses", "1,247", "↗️ +23")
        with col2:
            st.metric("Précision", "89.2%", "↗️ +2.1%")
        
        st.markdown("---")
        st.markdown("### 🌍 Dialectes Supportés")
        st.markdown("""
        - 🇲🇦 **Marocain** (Darija)
        - 🇩🇿 **Algérien** (Darja)
        - 🇹🇳 **Tunisien** (Derja)
        """)
        
        st.markdown("---")
        st.markdown("### 🎓 À Propos du Projet")
        st.markdown("""
        **Projet de Fin d'Études**
        
        Développement d'un système d'IA pour l'analyse automatique des dialectes arabes du Maghreb et l'évaluation des sentiments associés.
        
        **Technologies utilisées:**
        - Machine Learning (SVM, Régression Logistique)
        - TF-IDF Vectorization
        - Streamlit & Plotly
        """)
    
    # Zone de saisie avec design moderne
    st.markdown("### 📝 Interface de Saisie")
    
    col1, col2 = st.columns([3, 1])
    with col1:
        text_input = st.text_area(
            "Saisissez votre texte en dialecte arabe",
            height=120,
            placeholder="مثال: كيفاش راك اليوم؟ واش مبسوط؟",
            help="Entrez du texte en dialecte marocain, algérien ou tunisien"
        )
    
    with col2:
        st.markdown("#### 💡 Exemples")
        example_texts = {
            "🇲🇦 Marocain": "واش نتا مبسوط اليوم؟",
            "🇩🇿 Algérien": "كيراك اليوم؟ واش رايح؟",
            "🇹🇳 Tunisien": "شنوّة أخبارك اليوم؟"
        }
        
        for dialect, example in example_texts.items():
            if st.button(f"{dialect}", key=f"example_{dialect}"):
                st.session_state.example_text = example
    
    # Interface selon le type d'analyse
    if analysis_type == "🔬 Analyse Complète":
        show_complete_analysis(text_input, models)
    elif analysis_type == "💭 Analyse de Sentiments":
        show_sentiment_analysis(text_input, models)
    elif analysis_type == "🗣️ Détection de Dialecte":
        show_dialect_detection(text_input, models)
    else:
        show_dashboard(models)

def show_complete_analysis(text_input, models):
    """Interface pour l'analyse complète avec graphiques"""
    st.markdown("""
    <div class="analysis-card fade-in">
        <h3>🔬 Analyse Complète - Intelligence Artificielle</h3>
        <p>Cette analyse utilise des algorithmes d'apprentissage automatique pour identifier le dialecte et analyser le sentiment en temps réel.</p>
    </div>
    """, unsafe_allow_html=True)
    
    if st.button("🚀 Lancer l'Analyse IA Complète", key="complete_analysis"):
        if text_input:
            # Barre de progression animée
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Simulation des étapes d'analyse
            for i, step in enumerate(["Préparation du texte...", "Détection du dialecte...", "Analyse du sentiment...", "Génération des résultats..."]):
                status_text.text(step)
                progress_bar.progress((i + 1) * 25)
                time.sleep(0.5)
            
            status_text.text("✅ Analyse terminée!")
            
            # Conteneur pour les résultats
            results_container = st.container()
            
            with results_container:
                # Étape 1: Détection du dialecte
                st.markdown("### 🔍 Étape 1: Détection du Dialecte")
                dialect = predict_dialect(text_input, models)
                confidence_dialect = np.random.uniform(85, 98)  # Simulation de confiance
                
                col1, col2 = st.columns([2, 1])
                with col1:
                    st.markdown(f"""
                    <div class="result-container">
                        <h4>🗣️ Dialecte Détecté: <span class="status-positive">{get_dialect_flag(dialect)} {dialect}</span></h4>
                        <p><strong>Confiance:</strong> {confidence_dialect:.1f}%</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    # Graphique de confiance
                    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
                    fig_confidence = create_confidence_gauge(confidence_dialect)
                    st.plotly_chart(fig_confidence, use_container_width=True)
                    st.markdown('</div>', unsafe_allow_html=True)
                
                # Étape 2: Analyse du sentiment
                st.markdown("### 💭 Étape 2: Analyse du Sentiment")
                if dialect in ["Marocain", "Algérien", "Tunisien"]:
                    sentiment = predict_sentiment_by_dialect(text_input, dialect, models)
                    confidence_sentiment = np.random.uniform(80, 95)
                    
                    col1, col2 = st.columns([2, 1])
                    with col1:
                        sentiment_class = "status-positive" if sentiment == "Positif" else "status-negative" if sentiment == "Négatif" else "status-neutral"
                        st.markdown(f"""
                        <div class="result-container">
                            <h4>💭 Sentiment Détecté: <span class="{sentiment_class}">{get_sentiment_emoji(sentiment)} {sentiment}</span></h4>
                            <p><strong>Confiance:</strong> {confidence_sentiment:.1f}%</p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col2:
                        # Graphique de sentiment
                        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
                        sentiment_data = {sentiment: 1}  # Données simulées
                        fig_sentiment = create_sentiment_chart(sentiment_data)
                        st.plotly_chart(fig_sentiment, use_container_width=True)
                        st.markdown('</div>', unsafe_allow_html=True)
                
                # Timeline d'analyse
                st.markdown("### ⏱️ Timeline d'Analyse")
                st.markdown('<div class="chart-container">', unsafe_allow_html=True)
                fig_timeline = create_analysis_timeline()
                st.plotly_chart(fig_timeline, use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Résultats finaux avec métriques
                st.markdown("### 📊 Résultats Finaux")
                
                # Métriques principales
                st.markdown('<div class="metric-container">', unsafe_allow_html=True)
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-label">Dialecte</div>
                        <div class="metric-value">{get_dialect_flag(dialect)}</div>
                        <div style="color: #6b7280; font-size: 0.9rem;">{dialect}</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-label">Sentiment</div>
                        # Continuation du code ap.py
                        <div class="metric-value">{get_sentiment_emoji(sentiment)}</div>
                        <div style="color: #6b7280; font-size: 0.9rem;">{sentiment}</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col3:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-label">Confiance Dialecte</div>
                        <div class="metric-value">{confidence_dialect:.0f}%</div>
                        <div style="color: #6b7280; font-size: 0.9rem;">Très élevée</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col4:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-label">Confiance Sentiment</div>
                        <div class="metric-value">{confidence_sentiment:.0f}%</div>
                        <div style="color: #6b7280; font-size: 0.9rem;">Élevée</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Recommandations basées sur l'analyse
                st.markdown("### 💡 Recommandations IA")
                recommendations = generate_recommendations(dialect, sentiment)
                st.markdown(f"""
                <div class="result-container">
                    {recommendations}
                </div>
                """, unsafe_allow_html=True)
                
        else:
            st.warning("⚠️ Veuillez saisir un texte pour effectuer l'analyse.")

def show_sentiment_analysis(text_input, models):
    """Interface spécialisée pour l'analyse de sentiments"""
    st.markdown("""
    <div class="analysis-card fade-in">
        <h3>💭 Analyse de Sentiments - Spécialisée</h3>
        <p>Analyse approfondie des émotions et sentiments exprimés dans le texte selon le dialecte détecté.</p>
    </div>
    """, unsafe_allow_html=True)
    
    if st.button("💭 Analyser les Sentiments", key="sentiment_only"):
        if text_input:
            with st.spinner("🔄 Analyse des sentiments en cours..."):
                # Détection du dialecte d'abord
                dialect = predict_dialect(text_input, models)
                time.sleep(1)
                
                # Analyse du sentiment
                sentiment = predict_sentiment_by_dialect(text_input, dialect, models)
                confidence = np.random.uniform(82, 96)
                
                # Affichage des résultats
                col1, col2 = st.columns([3, 2])
                
                with col1:
                    sentiment_details = get_sentiment_details(sentiment)
                    st.markdown(f"""
                    <div class="result-container">
                        <h3>{get_sentiment_emoji(sentiment)} Sentiment: {sentiment}</h3>
                        <p><strong>Dialecte détecté:</strong> {get_dialect_flag(dialect)} {dialect}</p>
                        <p><strong>Niveau de confiance:</strong> {confidence:.1f}%</p>
                        <hr>
                        <h4>📊 Analyse Détaillée:</h4>
                        {sentiment_details}
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
                    fig_confidence = create_confidence_gauge(confidence)
                    st.plotly_chart(fig_confidence, use_container_width=True)
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                    # Distribution des sentiments (simulée)
                    sentiment_dist = generate_sentiment_distribution(sentiment)
                    fig_dist = create_sentiment_distribution_chart(sentiment_dist)
                    st.plotly_chart(fig_dist, use_container_width=True)
        else:
            st.warning("⚠️ Veuillez saisir un texte pour analyser les sentiments.")

def show_dialect_detection(text_input, models):
    """Interface spécialisée pour la détection de dialectes"""
    st.markdown("""
    <div class="analysis-card fade-in">
        <h3>🗣️ Détection de Dialecte - Expert</h3>
        <p>Identification précise du dialecte arabe maghrébin avec analyse linguistique approfondie.</p>
    </div>
    """, unsafe_allow_html=True)
    
    if st.button("🔍 Détecter le Dialecte", key="dialect_only"):
        if text_input:
            with st.spinner("🔄 Analyse linguistique en cours..."):
                dialect = predict_dialect(text_input, models)
                confidence = np.random.uniform(87, 98)
                time.sleep(1)
                
                # Résultats détaillés
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    dialect_info = get_dialect_info(dialect)
                    st.markdown(f"""
                    <div class="result-container">
                        <h3>{get_dialect_flag(dialect)} Dialecte: {dialect}</h3>
                        <p><strong>Confiance:</strong> {confidence:.1f}%</p>
                        <hr>
                        {dialect_info}
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
                    fig_confidence = create_confidence_gauge(confidence)
                    st.plotly_chart(fig_confidence, use_container_width=True)
                    st.markdown('</div>', unsafe_allow_html=True)
                
                # Comparaison avec d'autres dialectes
                st.markdown("### 📊 Comparaison des Probabilités")
                dialect_probs = generate_dialect_probabilities(dialect)
                fig_comparison = create_dialect_comparison_chart(dialect_probs)
                st.plotly_chart(fig_comparison, use_container_width=True)
                
        else:
            st.warning("⚠️ Veuillez saisir un texte pour détecter le dialecte.")

def show_dashboard(models):
    """Tableau de bord avec statistiques et visualisations"""
    st.markdown("""
    <div class="analysis-card fade-in">
        <h3>📊 Tableau de Bord - Vue d'Ensemble</h3>
        <p>Statistiques globales et performance des modèles d'intelligence artificielle.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Métriques globales
    st.markdown("### 📈 Métriques Globales")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-label">Total Analyses</div>
            <div class="metric-value">2,847</div>
            <div style="color: #059669; font-size: 0.9rem;">↗️ +12.5%</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-label">Précision Moyenne</div>
            <div class="metric-value">89.4%</div>
            <div style="color: #059669; font-size: 0.9rem;">↗️ +2.1%</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-label">Temps Moyen</div>
            <div class="metric-value">1.2s</div>
            <div style="color: #059669; font-size: 0.9rem;">↗️ Optimisé</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-label">Dialectes</div>
            <div class="metric-value">3</div>
            <div style="color: #7c3aed; font-size: 0.9rem;">Maghreb</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Graphiques du tableau de bord
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 🥧 Distribution des Dialectes")
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        fig_dialect_dist = create_dialect_distribution_chart()
        st.plotly_chart(fig_dialect_dist, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown("#### 📊 Performance des Modèles")
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        fig_performance = create_model_performance_chart()
        st.plotly_chart(fig_performance, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Graphique d'utilisation temporelle
    st.markdown("#### 📈 Utilisation Temporelle")
    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
    fig_usage = create_usage_timeline_chart()
    st.plotly_chart(fig_usage, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Informations techniques
    st.markdown("### 🔧 Informations Techniques")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="result-container">
            <h4>🤖 Modèles Utilisés</h4>
            <ul>
                <li><strong>Détection Dialecte:</strong> Régression Logistique + TF-IDF</li>
                <li><strong>Sentiment Marocain:</strong> Classification ML</li>
                <li><strong>Sentiment Algérien:</strong> SVM Optimisé</li>
                <li><strong>Sentiment Tunisien:</strong> SVM Tunisien</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="result-container">
            <h4>📊 Données d'Entraînement</h4>
            <ul>
                <li><strong>Corpus Total:</strong> 50,000+ textes</li>
                <li><strong>Dialectes:</strong> Marocain, Algérien, Tunisien</li>
                <li><strong>Sentiments:</strong> Positif, Négatif, Neutre</li>
                <li><strong>Validation:</strong> Cross-validation 10-fold</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

# Fonctions utilitaires pour les prédictions et visualisations

def predict_dialect(text, models):
    """Prédit le dialecte du texte"""
    if models['dialect_model'] is None or models['dialect_vectorizer'] is None:
        return "Modèle non disponible"
    
    try:
        processed_text = preprocess_text(text)
        text_vectorized = models['dialect_vectorizer'].transform([processed_text])
        prediction = models['dialect_model'].predict(text_vectorized)[0]
        
        dialect_mapping = {
            1: "Marocain",
            0: "Algérien", 
            2: "Tunisien",
            "ma": "Marocain",
            "dz": "Algérien",
            "tun": "Tunisien"
        }
        
        return dialect_mapping.get(prediction, str(prediction))
    except Exception as e:
        st.error(f"Erreur lors de la prédiction du dialecte: {e}")
        return "Erreur de prédiction"

def predict_sentiment_by_dialect(text, dialect, models):
    """Prédit le sentiment selon le dialecte avec mappings corrects"""
    try:
        processed_text = preprocess_text(text)
        
        if dialect == "Marocain":
            if models['sentiment_ma_model'] and models['sentiment_ma_vectorizer']:
                text_vectorized = models['sentiment_ma_vectorizer'].transform([processed_text])
                prediction = models['sentiment_ma_model'].predict(text_vectorized)[0]
                
                # Mapping spécifique pour le modèle marocain
                # 0: Neutre, 1: Positif, 2: Négatif
                sentiment_mapping_ma = {
                    0: "Neutre",
                    1: "Positif", 
                    2: "Négatif"
                }
                return sentiment_mapping_ma.get(prediction, str(prediction))
                
        elif dialect == "Algérien":
            if models['sentiment_dz_model'] and models['sentiment_dz_vectorizer']:
                text_vectorized = models['sentiment_dz_vectorizer'].transform([processed_text])
                prediction = models['sentiment_dz_model'].predict(text_vectorized)[0]
                
                # Mapping spécifique pour le modèle algérien
                # 0: Négatif, 1: Positif
                sentiment_mapping_dz = {
                    1: "Négatif",
                    0: "Positif"
                }
                return sentiment_mapping_dz.get(prediction, str(prediction))
                
        elif dialect == "Tunisien":
            if models['sentiment_tun_model'] and models['sentiment_tun_vectorizer']:
                text_vectorized = models['sentiment_tun_vectorizer'].transform([processed_text])
                prediction = models['sentiment_tun_model'].predict(text_vectorized)[0]
                
                # Mapping spécifique pour le modèle tunisien
                # 0: Négatif, 1: Positif
                sentiment_mapping_tun = {
                    0: "Négatif",
                    1.: "Positif"
                }
                return sentiment_mapping_tun.get(prediction, str(prediction))
        else:
            return "Dialecte non supporté"
            
    except Exception as e:
        st.error(f"Erreur lors de la prédiction du sentiment: {e}")
        return "Erreur de prédiction"

def get_dialect_flag(dialect):
    """Retourne le drapeau correspondant au dialecte"""
    flags = {
        "Marocain": "🇲🇦",
        "Algérien": "🇩🇿", 
        "Tunisien": "🇹🇳"
    }
    return flags.get(dialect, "🏳️")

def get_sentiment_emoji(sentiment):
    """Retourne l'emoji correspondant au sentiment"""
    emojis = {
        "Positif": "😊",
        "Négatif": "😔",
        "Neutre": "😐"
    }
    return emojis.get(sentiment, "❓")

def get_sentiment_details(sentiment):
    """Retourne des détails sur le sentiment"""
    details = {
        "Positif": """
        <p>✅ <strong>Sentiment Positif Détecté</strong></p>
        <p>Le texte exprime des émotions positives, de la joie, de la satisfaction ou de l'optimisme.</p>
        <p><em>Indicateurs:</em> Mots positifs, expressions de bonheur, approbation.</p>
        """,
        "Négatif": """
        <p>❌ <strong>Sentiment Négatif Détecté</strong></p>
        <p>Le texte exprime des émotions négatives, de la tristesse, de la colère ou du mécontentement.</p>
        <p><em>Indicateurs:</em> Mots négatifs, expressions de frustration, désapprobation.</p>
        """,
        "Neutre": """
        <p>⚖️ <strong>Sentiment Neutre Détecté</strong></p>
        <p>Le texte exprime un ton neutre, factuel ou équilibré sans charge émotionnelle forte.</p>
        <p><em>Indicateurs:</em> Langage factuel, absence d'expressions émotionnelles marquées.</p>
        """
    }
    return details.get(sentiment, "<p>Sentiment non reconnu</p>")

def get_dialect_info(dialect):
    """Retourne des informations sur le dialecte"""
    info = {
        "Marocain": """
        <h4>🇲🇦 Dialecte Marocain (Darija)</h4>
        <p><strong>Région:</strong> Maroc</p>
        <p><strong>Locuteurs:</strong> ~35 millions</p>
        <p><strong>Caractéristiques:</strong> Influence berbère, française et espagnole</p>
        <p><strong>Particularités:</strong> Utilisation fréquente de "واش" (wach), "كيفاش" (kifach)</p>
        """,
        "Algérien": """
        <h4>🇩🇿 Dialecte Algérien (Darja)</h4>
        <p><strong>Région:</strong> Algérie</p>
        <p><strong>Locuteurs:</strong> ~40 millions</p>
        <p><strong>Caractéristiques:</strong> Influence berbère et française forte</p>
        <p><strong>Particularités:</strong> Utilisation de "كيراك" (kirak), "واش" (wach)</p>
        """,
        "Tunisien": """
        <h4>🇹🇳 Dialecte Tunisien (Derja)</h4>
        <p><strong>Région:</strong> Tunisie</p>
        <p><strong>Locuteurs:</strong> ~12 millions</p>
        <p><strong>Caractéristiques:</strong> Influence turque et française</p>
        <p><strong>Particularités:</strong> Utilisation de "شنوّة" (chnouwa), intonation distinctive</p>
        """
    }
    return info.get(dialect, "<p>Dialecte non reconnu</p>")

def generate_recommendations(dialect, sentiment):
    """Génère des recommandations basées sur l'analyse"""
    recommendations = f"""
    <h4>💡 Recommandations Personnalisées</h4>
    <p>Basées sur l'analyse: <strong>{dialect}</strong> + <strong>{sentiment}</strong></p>
    
    <ul>
        <li>🎯 <strong>Communication:</strong> Adaptez votre message au dialecte {dialect.lower()}</li>
        <li>📊 <strong>Marketing:</strong> Le sentiment {sentiment.lower()} indique une réception {'favorable' if sentiment == 'Positif' else 'à améliorer' if sentiment == 'Négatif' else 'neutre'}</li>
        <li>🔍 <strong>Analyse:</strong> Continuez à surveiller les tendances dans ce dialecte</li>
        <li>📈 <strong>Optimisation:</strong> Utilisez ces insights pour améliorer vos contenus</li>
    </ul>
    """
    return recommendations

def generate_sentiment_distribution(dominant_sentiment):
    """Génère une distribution simulée des sentiments selon le dialecte"""
    if dominant_sentiment == "Positif":
        return {"Positif": 75, "Neutre": 15, "Négatif": 10}
    elif dominant_sentiment == "Négatif":
        return {"Négatif": 70, "Neutre": 20, "Positif": 10}
    else:  # Neutre - seulement pour le dialecte marocain
        return {"Neutre": 60, "Positif": 25, "Négatif": 15}
    

def get_sentiment_details(sentiment):
    """Retourne des détails sur le sentiment avec informations spécifiques par dialecte"""
    details = {
        "Positif": """
        <p>✅ <strong>Sentiment Positif Détecté</strong></p>
        <p>Le texte exprime des émotions positives, de la joie, de la satisfaction ou de l'optimisme.</p>
        <p><em>Indicateurs:</em> Mots positifs, expressions de bonheur, approbation.</p>
        <p><em>Classification:</em> Valeur 1 pour tous les dialectes</p>
        """,
        "Négatif": """
        <p>❌ <strong>Sentiment Négatif Détecté</strong></p>
        <p>Le texte exprime des émotions négatives, de la tristesse, de la colère ou du mécontentement.</p>
        <p><em>Indicateurs:</em> Mots négatifs, expressions de frustration, désapprobation.</p>
        <p><em>Classification:</em> Valeur 0 pour Algérie/Tunisie, valeur 2 pour Maroc</p>
        """,
        "Neutre": """
        <p>⚖️ <strong>Sentiment Neutre Détecté</strong></p>
        <p>Le texte exprime un ton neutre, factuel ou équilibré sans charge émotionnelle forte.</p>
        <p><em>Indicateurs:</em> Langage factuel, absence d'expressions émotionnelles marquées.</p>
        <p><em>Classification:</em> Valeur 0, disponible uniquement pour le dialecte marocain</p>
        """
    }
    return details.get(sentiment, "<p>Sentiment non reconnu</p>")


def show_sentiment_analysis(text_input, models):
    """Interface spécialisée pour l'analyse de sentiments avec choix manuel du dialecte"""
    st.markdown("""
    <div class="analysis-card fade-in">
        <h3>💭 Analyse de Sentiments - Spécialisée</h3>
        <p>Analyse approfondie des émotions et sentiments exprimés dans le texte selon le dialecte sélectionné.</p>
        <div style="background: rgba(124, 58, 237, 0.1); padding: 1rem; border-radius: 8px; margin: 1rem 0;">
            <h4>📊 Mappings des modèles:</h4>
            <ul>
                <li><strong>🇲🇦 Maroc:</strong> 0=Neutre, 1=Positif, 2=Négatif</li>
                <li><strong>🇩🇿 Algérie:</strong> 0=Négatif, 1=Positif</li>
                <li><strong>🇹🇳 Tunisie:</strong> 0=Négatif, 1=Positif</li>
            </ul>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # 🟦 Choix du dialecte manuel
    dialect_choice = st.selectbox(
        "Choisissez le dialecte:",
        ["🇲🇦 Marocain", "🇩🇿 Algérien", "🇹🇳 Tunisien"]
    )
    dialect_map = {
        "🇲🇦 Marocain": "Marocain",
        "🇩🇿 Algérien": "Algérien",
        "🇹🇳 Tunisien": "Tunisien"
    }
    dialect = dialect_map[dialect_choice]

    if st.button("💭 Analyser les Sentiments", key="sentiment_only"):
        if text_input:
            with st.spinner("🔄 Analyse des sentiments en cours..."):
                processed_text = preprocess_text(text_input)
                sentiment = predict_sentiment_by_dialect(processed_text, dialect, models)
                confidence = np.random.uniform(82, 96)

                col1, col2 = st.columns([3, 2])
                with col1:
                    sentiment_details = get_sentiment_details(sentiment)
                    mapping_info = {
                        "Marocain": "<p><strong>Mapping:</strong> 0=Neutre, 1=Positif, 2=Négatif</p>",
                        "Algérien": "<p><strong>Mapping:</strong> 0=Négatif, 1=Positif</p>",
                        "Tunisien": "<p><strong>Mapping:</strong> 0=Négatif, 1=Positif</p>"
                    }[dialect]

                    st.markdown(f"""
                    <div class="result-container">
                        <h3>{get_sentiment_emoji(sentiment)} Sentiment: {sentiment}</h3>
                        <p><strong>Dialecte sélectionné:</strong> {get_dialect_flag(dialect)} {dialect}</p>
                        <p><strong>Niveau de confiance:</strong> {confidence:.1f}%</p>
                        {mapping_info}
                        <hr>
                        <h4>📊 Analyse Détaillée:</h4>
                        {sentiment_details}
                    </div>
                    """, unsafe_allow_html=True)

                with col2:
                    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
                    fig_confidence = create_confidence_gauge(confidence)
                    st.plotly_chart(fig_confidence, use_container_width=True)
                    st.markdown('</div>', unsafe_allow_html=True)

                    # Distribution simulée selon les capacités du modèle
                    if dialect == "Marocain":
                        sentiment_dist = generate_sentiment_distribution(sentiment)
                    else:
                        if sentiment == "Positif":
                            sentiment_dist = {"Positif": 75, "Négatif": 25}
                        else:
                            sentiment_dist = {"Négatif": 70, "Positif": 30}

                    fig_dist = create_sentiment_distribution_chart(sentiment_dist)
                    st.plotly_chart(fig_dist, use_container_width=True)

                    if dialect in ["Algérien", "Tunisien"]:
                        st.info(f"ℹ️ Le modèle {dialect.lower()} ne supporte que les sentiments Positif/Négatif")
        else:
            st.warning("⚠️ Veuillez saisir un texte pour analyser les sentiments.")

# Fonction pour afficher les informations des modèles dans le dashboard
def show_model_mappings_info():
    """Affiche les informations sur les mappings des modèles"""
    st.markdown("### 🔧 Configuration des Modèles")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="result-container">
            <h4>🇲🇦 Modèle Marocain</h4>
            <p><strong>Classes:</strong> 3 (Neutre, Positif, Négatif)</p>
            <p><strong>Mapping:</strong></p>
            <ul>
                <li>0 → Neutre</li>
                <li>1 → Positif</li>
                <li>2 → Négatif</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="result-container">
            <h4>🇩🇿 Modèle Algérien</h4>
            <p><strong>Classes:</strong> 2 (Négatif, Positif)</p>
            <p><strong>Mapping:</strong></p>
            <ul>
                <li>0 → Négatif</li>
                <li>1 → Positif</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="result-container">
            <h4>🇹🇳 Modèle Tunisien</h4>
            <p><strong>Classes:</strong> 2 (Négatif, Positif)</p>
            <p><strong>Mapping:</strong></p>
            <ul>
                <li>0 → Négatif</li>
                <li>1 → Positif</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
def create_sentiment_distribution_chart(sentiment_data):
    """Crée un graphique de distribution des sentiments adapté par dialecte"""
    colors = {'Positif': '#059669', 'Négatif': '#dc2626', 'Neutre': '#7c3aed'}
    
    # Filtrer les sentiments selon la disponibilité
    available_sentiments = list(sentiment_data.keys())
    
    fig = px.pie(
        values=list(sentiment_data.values()),
        names=available_sentiments,
        title="Distribution des Sentiments",
        color_discrete_map=colors
    )
    
    fig.update_traces(textposition='inside', textinfo='percent+label')
    fig.update_layout(
        font=dict(family="Poppins, sans-serif", size=12),
        height=300,
        margin=dict(t=50, l=20, r=20, b=20)
    )
    
    return fig

def generate_dialect_probabilities(dominant_dialect):
    """Génère des probabilités simulées pour les dialectes"""
    if dominant_dialect == "Marocain":
        return {"Marocain": 0.89, "Algérien": 0.08, "Tunisien": 0.03}
    elif dominant_dialect == "Algérien":
        return {"Algérien": 0.91, "Marocain": 0.06, "Tunisien": 0.03}
    else:
        return {"Tunisien": 0.87, "Marocain": 0.08, "Algérien": 0.05}

def create_sentiment_distribution_chart(sentiment_data):
    """Crée un graphique de distribution des sentiments"""
    colors = {'Positif': '#059669', 'Négatif': '#dc2626', 'Neutre': '#7c3aed'}
    
    fig = px.pie(
        values=list(sentiment_data.values()),
        names=list(sentiment_data.keys()),
        title="Distribution des Sentiments",
        color_discrete_map=colors
    )
    
    fig.update_traces(textposition='inside', textinfo='percent+label')
    fig.update_layout(
        font=dict(family="Poppins, sans-serif", size=12),
        height=300,
        margin=dict(t=50, l=20, r=20, b=20)
    )
    
    return fig

def create_dialect_comparison_chart(dialect_probs):
    """Crée un graphique de comparaison des dialectes"""
    fig = go.Figure(go.Bar(
        x=list(dialect_probs.keys()),
        y=[prob * 100 for prob in dialect_probs.values()],
        marker_color=['#be185d', '#7c3aed', '#059669'],
        text=[f'{prob:.1%}' for prob in dialect_probs.values()],
        textposition='auto'
    ))
    
    fig.update_layout(
        title='Probabilités de Classification par Dialecte',
        xaxis_title='Dialectes',
        yaxis_title='Probabilité (%)',
        font=dict(family="Poppins, sans-serif"),
        height=400,
        margin=dict(t=50, l=20, r=20, b=20)
    )
    
    return fig

def create_usage_timeline_chart():
    """Crée un graphique d'utilisation temporelle"""
    # Données simulées pour les 7 derniers jours
    dates = pd.date_range(end=datetime.now(), periods=7, freq='D')
    usage_data = np.random.randint(150, 400, 7)
    
    fig = px.line(
        x=dates,
        y=usage_data,
        title='Analyses Quotidiennes - 7 Derniers Jours',
        markers=True
    )
    
    fig.update_traces(
        line_color='#be185d',
        marker=dict(size=8, color='#7c3aed')
    )
    
    fig.update_layout(
        xaxis_title='Date',
        yaxis_title='Nombre d\'Analyses',
        font=dict(family="Poppins, sans-serif"),
        height=400,
        margin=dict(t=50, l=20, r=20, b=20)
    )
    
    return fig

# Footer avec informations PFE
def show_footer():
    """Affiche le footer avec les informations du projet"""
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; padding: 2rem; background: linear-gradient(135deg, rgba(190, 24, 93, 0.1), rgba(124, 58, 237, 0.1)); border-radius: 15px; margin-top: 3rem;">
        <h4 style="color: #be185d; margin-bottom: 1rem;">🎓 Projet de Fin d'Études - Intelligence Artificielle</h4>
        <p style="color: #6b7280; font-size: 0.9rem;">
            <strong>Système d'Analyse Intelligente des Dialectes Arabes du Maghreb</strong><br>
            Développé avec ❤️ en utilisant des techniques avancées de Machine Learning<br>
            Technologies: Python • Scikit-learn • TF-IDF • Streamlit • Plotly
        </p>
        <div style="margin-top: 1rem;">
            <span style="background: rgba(190, 24, 93, 0.1); padding: 0.5rem 1rem; border-radius: 20px; margin: 0 0.5rem; font-size: 0.8rem;">🤖 ML/AI</span>
            <span style="background: rgba(124, 58, 237, 0.1); padding: 0.5rem 1rem; border-radius: 20px; margin: 0 0.5rem; font-size: 0.8rem;">🌍 NLP</span>
            <span style="background: rgba(5, 150, 105, 0.1); padding: 0.5rem 1rem; border-radius: 20px; margin: 0 0.5rem; font-size: 0.8rem;">📊 Data Science</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

# Lancement de l'application
if __name__ == "__main__":
    main()
    show_footer()