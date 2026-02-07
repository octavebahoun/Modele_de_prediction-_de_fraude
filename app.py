import streamlit as st
import pandas as pd
import joblib
import os
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier

# --- CONFIGURATION ---
st.set_page_config(
    page_title="Détection de Fraude Avancée",
    page_icon="🛡️",
    layout="wide"
)

BASE_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(BASE_DIR, "models/fraud_model_v2.joblib")
os.makedirs(os.path.join(BASE_DIR, "models"), exist_ok=True)

# Initialisation de l'historique dans la session
if 'history' not in st.session_state:
    st.session_state['history'] = []

# --- FONCTIONS UTILITAIRES ---

@st.cache_resource
def load_or_train_model():
    """
    Entraîne un modèle plus réaliste pour la simulation.
    Crée des corrélations : Montant élevé + Changement de lieu = Risque fort.
    """
    if os.path.exists(MODEL_PATH):
        return joblib.load(MODEL_PATH)
    else:
        # Génération de données synthétiques "intelligentes"
        np.random.seed(42)
        n_samples = 1000
        
        # Features: [amount, hour, transactions_today, loc_change, dev_change]
        X = np.random.rand(n_samples, 5)
        
        # Ajustement des échelles
        X[:, 0] = X[:, 0] * 5000  # Montant jusqu'à 5000
        X[:, 1] = X[:, 1] * 24    # Heures
        X[:, 2] = X[:, 2] * 20    # Transactions
        X[:, 3] = np.random.choice([0, 1], n_samples) # Loc change
        X[:, 4] = np.random.choice([0, 1], n_samples) # Dev change

        # Logique de fraude pour l'entraînement (Ground Truth)
        # Fraude si : (Montant > 3000) OU (Loc Change ET Dev Change)
        y = []
        for row in X:
            score = (row[0]/5000) * 0.5 + row[3] * 0.3 + row[4] * 0.2
            y.append(1 if score > 0.6 else 0)
            
        model = RandomForestClassifier(n_estimators=50, random_state=42)
        model.fit(X, y)
        joblib.dump(model, MODEL_PATH)
        return model

def explain_prediction(features, proba):
    """Génère un graphique expliquant quels facteurs ont pesé lourd."""
    # Simulation simple d'importance des features (SHAP-like) pour la démo
    importance = {
        'Montant': features['amount'].values[0] / 5000,
        'Heure': 0.1, # Impact faible simulé
        'Fréquence': features['transactions_today'].values[0] / 20,
        'Chg Lieu': features['location_change'].values[0] * 0.8,
        'Chg Appareil': features['device_change'].values[0] * 0.5
    }
    
    df_imp = pd.DataFrame(list(importance.items()), columns=['Facteur', 'Impact'])
    fig = px.bar(df_imp, x='Impact', y='Facteur', orientation='h', 
                 title="Contribution aux facteurs de risque",
                 color='Impact', color_continuous_scale='Reds')
    return fig

# --- SIDEBAR & CONFIGURATION ---
def sidebar_settings():
    st.sidebar.header("⚙️ Paramètres du Modèle")
    
    st.sidebar.info(
        "**Simulation de Faux Positifs :**\n"
        "Abaissez le seuil pour rendre le modèle 'paranoïaque'. "
        "Il signalera des transactions normales comme fraudes."
    )
    
    threshold = st.sidebar.slider(
        "Seuil de Sensibilité (Threshold)", 
        min_value=0.0, max_value=1.0, value=0.5, step=0.05,
        help="Si la probabilité dépasse ce seuil, c'est considéré comme une fraude."
    )
    
    return threshold

# --- PAGES ---

def prediction_page(model, threshold):
    st.header("🔍 Analyse Transactionnelle en Temps Réel")

    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col1:
        st.subheader("Données Transaction")
        amount = st.number_input("💰 Montant (€)", 0.0, 20000.0, 150.0, step=10.0)
        hour = st.slider("🕐 Heure de la journée", 0, 23, 14)
        transactions_today = st.number_input("📊 Tx ce jour", 0, 100, 3)
    
    with col2:
        st.subheader("Contexte Sécurité")
        loc_change = st.radio("📍 Changement de pays ?", [0, 1], format_func=lambda x: "Oui (Risque)" if x==1 else "Non")
        dev_change = st.radio("📱 Nouvel appareil ?", [0, 1], format_func=lambda x: "Oui (Risque)" if x==1 else "Non")

    features = pd.DataFrame([[amount, hour, transactions_today, loc_change, dev_change]], 
                           columns=['amount', 'hour', 'transactions_today', 'location_change', 'device_change'])
    
    # Prédiction
    prob = model.predict_proba(features)[0][1]
    is_fraud = prob > threshold

    with col3:
        st.subheader("Résultat du Modèle")
        
        # Jauge de risque
        fig_gauge = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = prob * 100,
            title = {'text': "Probabilité de Fraude (%)"},
            gauge = {
                'axis': {'range': [0, 100]},
                'bar': {'color': "darkred" if prob > threshold else "green"},
                'steps': [
                    {'range': [0, threshold*100], 'color': "lightgreen"},
                    {'range': [threshold*100, 100], 'color': "salmon"}],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': threshold * 100}
            }
        ))
        fig_gauge.update_layout(height=250, margin=dict(l=20,r=20,t=50,b=20))
        st.plotly_chart(fig_gauge, use_container_width=True)

    # Analyse et Verdict
    st.divider()
    
    # Logique de message avancée pour expliquer les Faux Positifs
    if is_fraud:
        msg_type = "error"
        status = "🚨 SUSPICION DE FRAUDE"
        
        if 0.5 > prob > threshold:
            explanation = "⚠️ **Note :** Ceci pourrait être un **Faux Positif**. Le risque est modéré, mais votre seuil de sensibilité est très bas."
        else:
            explanation = "Le modèle a identifié des indicateurs forts de compromission."
    else:
        msg_type = "success"
        status = "✅ TRANSACTION APPROUVÉE"
        
        if threshold > prob > 0.5:
             explanation = "⚠️ **Note :** Ceci pourrait être un **Faux Négatif**. Le risque est élevé (>50%), mais votre seuil est trop permissif."
        else:
             explanation = "Aucune anomalie majeure détectée."

    # Affichage du verdict
    if msg_type == "error":
        st.error(f"### {status}")
    else:
        st.success(f"### {status}")
        
    st.write(explanation)
    
    # Visualisation des facteurs
    with st.expander("📊 Voir pourquoi (Facteurs d'influence)"):
        st.plotly_chart(explain_prediction(features, prob), use_container_width=True)

    # Bouton pour sauvegarder dans l'historique
    if st.button("💾 Enregistrer dans l'historique de session"):
        st.session_state['history'].append({
            "Montant": amount,
            "Risque": prob,
            "Seuil": threshold,
            "Verdict": "Fraude" if is_fraud else "Légitime",
            "Type": "Simulation"
        })
        st.toast("Transaction enregistrée !", icon="💾")

def history_page():
    st.header("📜 Historique de la Session")
    
    if not st.session_state['history']:
        st.info("Aucune transaction testée pour le moment.")
        return

    df_hist = pd.DataFrame(st.session_state['history'])
    st.dataframe(df_hist, use_container_width=True)
    
    # Petit graph de répartition
    fig = px.pie(df_hist, names='Verdict', title='Répartition des décisions', color='Verdict',
                 color_discrete_map={'Fraude':'red', 'Légitime':'green'})
    st.plotly_chart(fig)

def dashboard_page(model):
    st.header("📂 Analyse en Lot (Batch Processing)")
    
    uploaded_file = st.file_uploader("Charger un CSV de transactions", type=['csv'])
    
    # Générateur de CSV intégré
    if st.checkbox("Générer des données de test aléatoires"):
        n_test = st.slider("Nombre de lignes", 10, 500, 50)
        df = pd.DataFrame({
            'amount': np.random.exponential(100, n_test) + np.random.choice([0, 2000], n_test, p=[0.9, 0.1]),
            'hour': np.random.randint(0, 24, n_test),
            'transactions_today': np.random.randint(1, 15, n_test),
            'location_change': np.random.choice([0, 1], n_test, p=[0.8, 0.2]),
            'device_change': np.random.choice([0, 1], n_test, p=[0.9, 0.1])
        })
    elif uploaded_file:
        df = pd.read_csv(uploaded_file)
    else:
        st.info("En attente de données...")
        return

    # Prédictions
    probs = model.predict_proba(df)[:, 1]
    df['Probabilité_Fraude'] = probs
    
    # Utilisation du slider de la sidebar pour filtrer ici aussi
    threshold = st.session_state.get('threshold_val', 0.5) 
    # (Note: on récupérerait la valeur si on la stockait, ici on utilise le standard ou on le passe en arg)
    
    st.write("---")
    col_kpi1, col_kpi2 = st.columns(2)
    
    # Interactif : Ajuster le seuil pour voir l'impact sur le lot
    local_threshold = st.slider("Ajuster le seuil pour ce lot", 0.0, 1.0, 0.5)
    
    df['Décision'] = df['Probabilité_Fraude'].apply(lambda x: "🚨 Fraude" if x > local_threshold else "✅ Valide")
    
    frauds = df[df['Décision'] == "🚨 Fraude"]
    
    col_kpi1.metric("Transactions Analysées", len(df))
    col_kpi2.metric("Fraudes Détectées", len(frauds), delta_color="inverse")
    
    # Scatter plot interactif
    fig = px.scatter(df, x="amount", y="Probabilité_Fraude", color="Décision",
                     size="transactions_today", hover_data=['location_change'],
                     color_discrete_map={"🚨 Fraude": "red", "✅ Valide": "green"},
                     title="Distribution des Risques vs Montant")
    
    # Ligne de seuil
    fig.add_hline(y=local_threshold, line_dash="dash", line_color="black", annotation_text="Seuil actuel")
    
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(df)

# --- MAIN ---
def main():
    model = load_or_train_model()
    
    # Sidebar Navigation
    st.sidebar.title("Navigation")
    menu = ["🔍 Analyse Unique", "📂 Analyse de Lot", "📜 Historique"]
    choice = st.sidebar.radio("Aller vers", menu)
    
    st.sidebar.divider()
    threshold = sidebar_settings()
    
    if choice == "🔍 Analyse Unique":
        prediction_page(model, threshold)
    elif choice == "📂 Analyse de Lot":
        dashboard_page(model)
    else:
        history_page()

if __name__ == "__main__":
    main()