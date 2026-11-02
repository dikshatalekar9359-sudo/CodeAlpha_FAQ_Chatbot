import json
import re
from pathlib import Path

import nltk
import streamlit as st
from nltk.tokenize import wordpunct_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="College FAQ AI Assistant",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)


# -----------------------------
# Custom Styling
# -----------------------------
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #f5f7ff 0%, #eef2ff 100%);
}
.main-title {
    font-size: 36px;
    font-weight: 800;
    color: #253b80;
    text-align: center;
    margin-bottom: 4px;
}
.subtitle {
    text-align: center;
    color: #5b6480;
    font-size: 16px;
    margin-bottom: 24px;
}
.info-card {
    background: white;
    padding: 18px;
    border-radius: 14px;
    border: 1px solid #e0e7ff;
    margin-bottom: 12px;
}
.small-note {
    color: #64748b;
    font-size: 13px;
}
</style>
""", unsafe_allow_html=True)


# -----------------------------
# Text Preprocessing
# -----------------------------
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    tokens = wordpunct_tokenize(text)
    tokens = [token for token in tokens if token.strip()]
    return " ".join(tokens)


# -----------------------------
# Load FAQ Dataset
# -----------------------------
@st.cache_data
def load_faqs():
    file_path = Path(__file__).parent / "faqs.json"

    with open(file_path, "r", encoding="utf-8") as file:
        data = json.load(file)

    if not isinstance(data, list) or not data:
        raise ValueError("FAQ dataset must be a non-empty JSON list.")

    for item in data:
        if not item.get("question") or not item.get("answer"):
            raise ValueError("Every FAQ needs a question and an answer.")

    return data


# -----------------------------
# Build TF-IDF Model
# -----------------------------
@st.cache_resource
def build_model(questions):
    processed_questions = [
        preprocess_text(question) for question in questions
    ]

    vectorizer = TfidfVectorizer(ngram_range=(1, 2))

    question_vectors = vectorizer.fit_transform(processed_questions)

    return vectorizer, question_vectors


faqs = load_faqs()

questions = [faq["question"] for faq in faqs]
answers = [faq["answer"] for faq in faqs]

vectorizer, question_vectors = build_model(questions)


# -----------------------------
# Find Best Matching FAQ
# -----------------------------
def get_response(user_question):
    cleaned_question = preprocess_text(user_question)

    if not cleaned_question.strip():
        return (
            "Please type a question so I can help you.",
            0.0,
            None
        )

    user_vector = vectorizer.transform([cleaned_question])

    scores = cosine_similarity(
        user_vector,
        question_vectors
    )[0]

    best_index = scores.argmax()
    best_score = float(scores[best_index])

    # This threshold is a starting point, not a universal rule.
    # Adjust it after testing real questions.
    threshold = 0.18

    if best_score < threshold:
        return (
            "Sorry, I couldn't find a reliable answer in my FAQ "
            "database. Please try another question or contact your "
            "college office.",
            best_score,
            None
        )

    return answers[best_index], best_score, questions[best_index]


# -----------------------------
# Sidebar
# -----------------------------
with st.sidebar:
    st.title("🎓 FAQ Assistant")

    st.write(
        "Your AI-powered assistant for common college questions."
    )

    st.divider()

    st.subheader("✨ Example Questions")

    examples = [
        "What courses are available?",
        "How can I apply for admission?",
        "Are scholarships available?",
        "How can I check my results?",
        "How can I get my timetable?"
    ]

    for example in examples:
        if st.button(example, use_container_width=True):
            st.session_state.pending_question = example

    st.divider()

    st.metric("FAQs Available", len(faqs))

    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.session_state.pending_question = None
        st.rerun()

    st.caption("Built using Python, NLP and Streamlit.")


# -----------------------------
# Main Header
# -----------------------------
st.markdown(
    '<div class="main-title">🎓 College FAQ AI Assistant</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Ask your college-related questions and get instant answers.'
    '</div>',
    unsafe_allow_html=True
)

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown(
        '<div class="info-card"><h4>🧠 NLP Powered</h4>'
        '<p>Text preprocessing and TF-IDF matching.</p></div>',
        unsafe_allow_html=True
    )

with col2:
    st.markdown(
        '<div class="info-card"><h4>⚡ Instant Answers</h4>'
        '<p>Find the closest matching FAQ quickly.</p></div>',
        unsafe_allow_html=True
    )

with col3:
    st.markdown(
        '<div class="info-card"><h4>💬 Easy to Use</h4>'
        '<p>A simple conversational interface.</p></div>',
        unsafe_allow_html=True
    )


# -----------------------------
# Initialize Chat History
# -----------------------------
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": (
                "Hello! 👋 Welcome to the College FAQ Assistant. "
                "Ask me about admissions, courses, examinations, "
                "scholarships or other college-related topics."
            )
        }
    ]

if "pending_question" not in st.session_state:
    st.session_state.pending_question = None


# -----------------------------
# Display Previous Messages
# -----------------------------
st.subheader("💬 Chat with the Assistant")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        if message.get("matched_question"):
            st.caption(
                "Matched FAQ: " + message["matched_question"]
            )

        if message.get("score") is not None:
            st.caption(
                f"Similarity score: {message['score']:.2%}"
            )


# -----------------------------
# User Input
# -----------------------------
user_input = st.chat_input("Type your question here...")

if st.session_state.pending_question:
    user_input = st.session_state.pending_question
    st.session_state.pending_question = None


# -----------------------------
# Generate Response
# -----------------------------
if user_input:
    st.session_state.messages.append(
        {
            "role": "user",
            "content": user_input
        }
    )

    answer, score, matched_question = get_response(user_input)

    assistant_message = {
        "role": "assistant",
        "content": answer,
        "score": score,
        "matched_question": matched_question
    }

    st.session_state.messages.append(assistant_message)

    st.rerun()


# -----------------------------
# Footer
# -----------------------------
st.divider()

st.markdown(
    '<p class="small-note" style="text-align:center;">'
    'AI-Based FAQ Chatbot | Python • NLP • Scikit-learn • Streamlit'
    '</p>',
    unsafe_allow_html=True
)