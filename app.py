import streamlit as st
import joblib
import re
import nltk
from nltk.corpus import stopwords
import pandas as pd # Import pandas
import matplotlib.pyplot as plt # Import matplotlib for plotting
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay # Import for confusion matrix

# --- PAGE CONFIG ---
st.set_page_config(page_title="CineSense AI", page_icon="🎬", layout="wide")

# --- CUSTOM CSS FOR STYLING ---
st.markdown("""
    <style>
    .main {
        background-color: #f5f7f9;
    }
    .stButton>button {
        width: 100%;
        border-radius: 5px;
        height: 3em;
        background-color: #FF4B4B;
        color: white;
        font-weight: bold;
    }
    .sentiment-box {
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        margin-bottom: 10px;
        color: white;
    }
    .pos-box { background-color: #28a745; }
    .neg-box { background-color: #dc3545; }
    </style>
    """, unsafe_allow_html=True)

# --- LOAD MODELS ---
@st.cache_resource
def load_assets():
    try:
        tfidf = joblib.load("/content/tfidf.pkl")
        model = joblib.load("/content/best_model_F1.pkl")
        kmeans = joblib.load("/content/kmeans.pkl")
        pca = joblib.load("/content/pca.pkl")
        X_train_pca = joblib.load("/content/X_train_pca.pkl") # Load X_train_pca
        clusters = joblib.load("/content/clusters.pkl") # Load clusters
        y_test = joblib.load("/content/y_test.pkl") # Load y_test
        data = joblib.load("/content/data.pkl") # Load the full data

        try:
            all_model_results = joblib.load("/content/model_results.pkl")
            model_accuracies = {k.replace('PCA_', ''): v for k, v in all_model_results.items()}
        except:
            model_accuracies = {"LR": 0.63, "KNN": 0.50, "LGBM": 0.61, "RF": 0.57}

        return tfidf, model, kmeans, pca, X_train_pca, clusters, y_test, data, model_accuracies
    except Exception as e:
        st.error(f"Error loading assets: {e}")
        return None, None, None, None, None, None, None, None, {}

tfidf, model, kmeans, pca, X_train_pca, clusters, y_test, data, model_accuracies = load_assets()

# --- HELPER FUNCTIONS ---
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    return text

# --- SIDEBAR ---
with st.sidebar:
    st.image("https://img.icons8.com/fluency/144/movie-projector.png", width=100)
    st.title("Movie Review Analyzer")
    st.info("This AI analyzes movie reviews using NLP, PCA dimensionality reduction, and K-Means clustering.")
    st.markdown("---")
    st.subheader("Model Metrics")
    # Convert to DataFrame before displaying to handle column names
    accuracies_df = pd.DataFrame(list(model_accuracies.items()), columns=["Model", "Accuracy"])
    st.dataframe(accuracies_df, use_container_width=True)

    st.markdown("---")
    st.subheader("Overall Sentiment Distribution")
    if data is not None:
        sentiment_counts = data['tag'].value_counts().rename(index={1: 'Positive', 0: 'Negative'})
        fig_sentiment, ax_sentiment = plt.subplots()
        ax_sentiment.bar(sentiment_counts.index, sentiment_counts.values, color=['green', 'red'])
        ax_sentiment.set_title('Distribution of Sentiments in Dataset')
        ax_sentiment.set_xlabel('Sentiment')
        ax_sentiment.set_ylabel('Number of Reviews')
        st.pyplot(fig_sentiment)
        plt.close(fig_sentiment)
    else:
        st.warning("Data for sentiment distribution not loaded.")

# --- MAIN UI ---
st.title("🎬 Movie Review Analyzer")
st.markdown("Experience high-fidelity sentiment analysis and behavioral clustering.")

col1, col2 = st.columns([2, 1])

with col1:
    user_input = st.text_area("✍️ Paste your review here:", height=200, placeholder="The cinematography was breathtaking, but the plot was thin...")
    analyze_btn = st.button("Run Intelligence Engine")

with col2:
    st.subheader("Analysis Results")
    if analyze_btn and user_input.strip() != "":
        # Process
        clean = clean_text(user_input)
        vector = tfidf.transform([clean])
        vector_pca = pca.transform(vector)

        prediction = model.predict(vector_pca)[0]
        cluster = kmeans.predict(vector_pca)[0]

        if hasattr(model, 'predict_proba'):
            proba = model.predict_proba(vector_pca)[0]

        # Display Sentiment
        if prediction == 1:
            st.markdown('<div class="sentiment-box pos-box"><h3>POSITIVE SENTIMENT</h3></div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="sentiment-box neg-box"><h3>NEGATIVE SENTIMENT</h3></div>', unsafe_allow_html=True)

        # Display Cluster Metrics
        cluster_labels = {0: "Harsh Critics", 1: "Emotional Fans", 2: "Neutral Reviewers"}
        c_name = cluster_labels.get(cluster, "Unknown")
        st.metric(label="Reviewer Persona", value=c_name)

        # Confidence Scores
        if hasattr(model, 'predict_proba'):
            proba = model.predict_proba(vector_pca)[0]
            st.write("**Confidence Level:**")
            st.progress(float(proba[1]))
            st.caption(f"Confidence: {max(proba)*100:.1f}%")

        st.subheader("Cluster Visualization")
        if X_train_pca is not None and clusters is not None:
            fig, ax = plt.subplots(figsize=(8, 6))
            scatter = ax.scatter(X_train_pca[:, 0], X_train_pca[:, 1], c=clusters, cmap='viridis', alpha=0.6, s=5)
            ax.scatter(vector_pca[:, 0], vector_pca[:, 1], color='red', marker='X', s=200, label='Current Review', edgecolor='black')
            ax.set_title("K-Means Clusters with Current Review Highlighted")
            ax.set_xlabel("Principal Component 1")
            ax.set_ylabel("Principal Component 2")
            ax.legend()
            st.pyplot(fig)
            plt.close(fig) # Close the figure to prevent it from displaying again
        else:
            st.warning("Training data for cluster visualization not loaded.")

    else:
        st.write("Results will appear here after analysis.")

# --- FOOTER VISUALS ---
st.markdown("---")
st.subheader("Model Performance Benchmarks")
st.bar_chart(model_accuracies)

st.markdown("---")
st.subheader("Confusion Matrix for Best Model")
if model is not None and pca is not None and y_test is not None:
    # Predict on the test set to get predictions for the confusion matrix
    X_test_pca = pca.transform(tfidf.transform(data['cleaned_text'].loc[y_test.index]).toarray()) # Ensure X_test_pca is correctly derived from y_test indices
    y_pred = model.predict(X_test_pca)
    cm = confusion_matrix(y_test, y_pred, labels=model.classes_)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Negative', 'Positive'])
    fig_cm, ax_cm = plt.subplots()
    disp.plot(cmap='Blues', ax=ax_cm)
    ax_cm.set_title('Confusion Matrix')
    st.pyplot(fig_cm)
    plt.close(fig_cm)
else:
    st.warning("Model, PCA, or y_test not loaded for confusion matrix.")
