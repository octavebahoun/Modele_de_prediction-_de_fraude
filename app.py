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

# Liste des features attendues par le modèle (Ordre Important)
FEATURES = ['amount', 'hour', 'transactions_today', 'location_change', 'device_change']

# Initialisation de l'historique dans la session
if 'history' not in st.session_state:
    st.session_state['history'] = []

# --- FONCTIONS UTILITAIRES ---

@st.cache_resource
def load_or_train_model():
    """
    Entraîne un modèle et le sauvegarde.
    """
    if os.path.exists(MODEL_PATH):
        try:
            return joblib.load(MODEL_PATH)
        except:
            pass # Si le fichier est corrompu, on re-entraine

    # Génération de données synthétiques "intelligentes"
    np.random.seed(42)
    n_samples = 1000
    
    # Création des données brutes
    data = np.random.rand(n_samples, 5)
    
    # Ajustement des échelles
    data[:, 0] = data[:, 0] * 5000  # Montant
    data[:, 1] = data[:, 1] * 24    # Heures
    data[:, 2] = data[:, 2] * 20    # Transactions
    data[:, 3] = np.random.choice([0, 1], n_samples) # Loc change
    data[:, 4] = np.random.choice([0, 1], n_samples) # Dev change

    # *** CORRECTION 1 : Créer un DataFrame avec les noms de colonnes ***
    X = pd.DataFrame(data, columns=FEATURES)

    # Logique de fraude (Ground Truth)
    y = []
    for index, row in X.iterrows():
        score = (row['amount']/5000) * 0.5 + row['location_change'] * 0.3 + row['device_change'] * 0.2
        y.append(1 if score > 0.6 else 0)
        
    model = RandomForestClassifier(n_estimators=50, random_state=42)
    model.fit(X, y)
    joblib.dump(model, MODEL_PATH)
    return model

def explain_prediction(features_df, proba):
    """Génère un graphique expliquant quels facteurs ont pesé lourd."""
    # Simulation simple d'importance pour la démo
    importance = {
        'Montant': features_df['amount'].values[0] / 5000,
        'Heure': 0.1, 
        'Fréquence': features_df['transactions_today'].values[0] / 20,
        'Chg Lieu': features_df['location_change'].values[0] * 0.8,
        'Chg Appareil': features_df['device_change'].values[0] * 0.5
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
        "Abaissez le seuil pour rendre le modèle 'paranoïaque'."
    )
    
    threshold = st.sidebar.slider(
        "Seuil de Sensibilité (Threshold)", 
        min_value=0.0, max_value=1.0, value=0.5, step=0.05
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

    # Création du DF pour la prédiction
    features = pd.DataFrame([[amount, hour, transactions_today, loc_change, dev_change]], 
                           columns=FEATURES)
    
    # Prédiction
    # On s'assure de ne passer que les bonnes colonnes (sécurité)
    prob = model.predict_proba(features[FEATURES])[0][1]
    is_fraud = prob > threshold

    with col3:
        st.subheader("Résultat du Modèle")
        
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
    
    if is_fraud:
        st.error("### 🚨 SUSPICION DE FRAUDE")
        if 0.5 > prob > threshold:
            st.write("⚠️ **Note :** Risque modéré, détecté car votre seuil est bas.")
        else:
            st.write("Indicateurs forts de compromission.")
    else:
        st.success("### ✅ TRANSACTION APPROUVÉE")
        if threshold > prob > 0.5:
             st.write("⚠️ **Note :** Risque élevé (>50%), mais accepté par votre seuil permissif.")

    with st.expander("📊 Voir pourquoi (Facteurs d'influence)"):
        st.plotly_chart(explain_prediction(features, prob), use_container_width=True)

    if st.button("💾 Enregistrer dans l'historique"):
        st.session_state['history'].append({
            "Montant": amount,
            "Risque": prob,
            "Verdict": "Fraude" if is_fraud else "Légitime"
        })
        st.toast("Enregistré !", icon="💾")

def dashboard_page(model, threshold): # Ajout de threshold en argument
    st.header("📂 Analyse en Lot (Batch Processing)")
    
    uploaded_file = st.file_uploader("Charger un CSV", type=['csv'])
    
    df = None # Initialisation explicite

    if st.checkbox("Générer des données de test aléatoires"):
        n_test = st.slider("Nombre de lignes", 10, 500, 50)
        # On s'assure que les clés du dictionnaire correspondent exactement à FEATURES
        df = pd.DataFrame({
            'amount': np.random.exponential(100, n_test) + np.random.choice([0, 2000], n_test, p=[0.9, 0.1]),
            'hour': np.random.randint(0, 24, n_test),
            'transactions_today': np.random.randint(1, 15, n_test),
            'location_change': np.random.choice([0, 1], n_test, p=[0.8, 0.2]),
            'device_change': np.random.choice([0, 1], n_test, p=[0.9, 0.1])
        })
    elif uploaded_file:
        df = pd.read_csv(uploaded_file)
    
    if df is not None:
        st.write("Aperçu des données :")
        st.dataframe(df.head())

        # *** CORRECTION 2 : Le cœur du problème ***
        # On vérifie que toutes les colonnes nécessaires sont là
        missing_cols = [col for col in FEATURES if col not in df.columns]
        
        if missing_cols:
            st.error(f"Il manque les colonnes suivantes dans le CSV : {missing_cols}")
        else:
            # On extrait UNIQUEMENT les colonnes utiles pour la prédiction
            # Cela rejette les colonnes en trop et évite l'erreur "6 features vs 5"
            X_pred = df[FEATURES]
            
            try:
                probs = model.predict_proba(X_pred)[:, 1]
                
                # On ajoute les résultats au DF original (pour l'affichage)
                df['Probabilité_Fraude'] = probs
                
                local_threshold = st.slider("Ajuster le seuil pour ce lot", 0.0, 1.0, threshold)
                df['Décision'] = df['Probabilité_Fraude'].apply(lambda x: "🚨 Fraude" if x > local_threshold else "✅ Valide")
                
                # Affichage des stats
                col1, col2 = st.columns(2)
                col1.metric("Transactions", len(df))
                col2.metric("Fraudes détectées", len(df[df['Décision'] == "🚨 Fraude"]), delta_color="inverse")
                
                # Graphique
                fig = px.scatter(df, x="amount", y="Probabilité_Fraude", color="Décision",
                                color_discrete_map={"🚨 Fraude": "red", "✅ Valide": "green"})
                fig.add_hline(y=local_threshold, line_dash="dash", annotation_text="Seuil")
                st.plotly_chart(fig, use_container_width=True)
                
            except Exception as e:
                st.error(f"Erreur lors de la prédiction : {e}")

def history_page():
    st.header("📜 Historique")
    if st.session_state['history']:
        df_hist = pd.DataFrame(st.session_state['history'])
        st.dataframe(df_hist, use_container_width=True)
    else:
        st.info("Historique vide.")

# --- MAIN ---
def main():
    model = load_or_train_model()
    
    st.sidebar.title("Navigation")
    choice = st.sidebar.radio("Aller vers", ["🔍 Analyse Unique", "📂 Analyse de Lot", "📜 Historique"])
    st.sidebar.divider()
    
    threshold = sidebar_settings()
    
    if choice == "🔍 Analyse Unique":
        prediction_page(model, threshold)
    elif choice == "📂 Analyse de Lot":
        # On passe threshold ici
        dashboard_page(model, threshold)
    else:
        history_page()

if __name__ == "__main__":
    main()