import streamlit as st
import pandas as pd
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="AI Study Buddy", layout="centered")
st.title("📘 AI Study Buddy for Students")
st.write("An interactive learning assistant using AI")

# ---------------- LOAD DATA ----------------
@st.cache_data
def load_data():
    return pd.read_csv("data/qa_dataset.csv")

df = load_data()

# ---------------- NLP MODEL ----------------
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df["question"])

def get_answer(question):
    q_vec = vectorizer.transform([question])
    similarity = cosine_similarity(q_vec, X)
    index = similarity.argmax()
    return df.iloc[index]["answer"]

# ---------------- SESSION STATE ----------------
if "chat" not in st.session_state:
    st.session_state.chat = []

if "score" not in st.session_state:
    st.session_state.score = 0

if "flash_index" not in st.session_state:
    st.session_state.flash_index = 0

# ---------------- SIDEBAR ----------------
feature = st.sidebar.selectbox(
    "Choose Feature",
    ["Explain Topic", "Chat with AI", "Summarize Notes", "Quiz (MCQ)", "Flashcards"]
)

difficulty = st.sidebar.radio(
    "Difficulty Level",
    ["Beginner", "Intermediate", "Advanced"]
)

st.sidebar.markdown("---")
st.sidebar.write("🏆 Quiz Score:", st.session_state.score)

# ---------------- EXPLAIN TOPIC ----------------
if feature == "Explain Topic":
    st.subheader("📖 Explain a Topic")
    q = st.text_input("Enter your question")

    if st.button("Explain"):
        if q:
            ans = get_answer(q)
            if difficulty == "Beginner":
                st.success(ans)
            elif difficulty == "Intermediate":
                st.info(ans + " This concept is commonly used in real-world applications.")
            else:
                st.warning(ans + " Example: This is used in advanced AI systems.")
        else:
            st.warning("Please enter a question")

# ---------------- CHAT MODE ----------------
elif feature == "Chat with AI":
    st.subheader("💬 Chat with Study Buddy")

    user_q = st.text_input("Ask anything")

    if st.button("Send"):
        if user_q:
            ans = get_answer(user_q)
            st.session_state.chat.append(("You", user_q))
            st.session_state.chat.append(("AI", ans))

    for role, msg in st.session_state.chat:
        if role == "You":
            st.write("🧑 **You:**", msg)
        else:
            st.write("🤖 **AI:**", msg)

# ---------------- SUMMARIZE NOTES ----------------
elif feature == "Summarize Notes":
    st.subheader("📝 Summarize Notes")
    notes = st.text_area("Paste your notes here")

    if st.button("Summarize"):
        if notes:
            sentences = notes.split(".")
            summary = ". ".join(sentences[:2])
            st.success(summary)
        else:
            st.warning("Please enter notes")

# ---------------- QUIZ (MCQ) ----------------
elif feature == "Quiz (MCQ)":
    st.subheader("❓ Quiz Time")

    q = df.sample(1).iloc[0]
    correct = q["answer"]

    options = random.sample(list(df["answer"]), 3)
    options.append(correct)
    random.shuffle(options)

    st.write(q["question"])
    choice = st.radio("Choose the correct answer:", options)

    if st.button("Submit Answer"):
        if choice == correct:
            st.success("Correct 🎉")
            st.session_state.score += 1
        else:
            st.error("Wrong ❌")
            st.info("Correct answer: " + correct)

# ---------------- FLASHCARDS ----------------
elif feature == "Flashcards":
    st.subheader("🧠 Flashcards")

    card = df.iloc[st.session_state.flash_index % len(df)]

    st.info(card["question"])

    if st.button("Show Answer"):
        st.success(card["answer"])

    if st.button("Next Card"):
        st.session_state.flash_index += 1
