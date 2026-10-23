"""
Interactive Streamlit Web Application: Amharic Predictive Text System
=====================================================================
Addis Ababa University | College of Technology & Built Environment
Course: Natural Language Processing (NLP) - Assignment 1 (Task #4)

Run locally using:
  streamlit run "NLP assignment/app_streamlit.py"
"""

import os
import sys
import streamlit as st

# Ensure local imports work
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from q4_predictive_system import AmharicPredictiveSystem
from amharic_preprocessor import tokenize_amharic_sentence

st.set_page_config(
    page_title="Amharic Predictive Text & Next-Word Suggestion",
    page_icon="🇪🇹",
    layout="wide"
)

# Custom styling
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 800;
        color: #1e3a8a;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #4b5563;
        margin-bottom: 1.5rem;
    }
    .stButton>button {
        border-radius: 8px;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource(show_spinner="Loading GPAC N-gram Language Models...")
def load_predictive_system():
    system = AmharicPredictiveSystem()
    system.train_or_load(max_sentences=100000)
    return system


predictor = load_predictive_system()

# ---------------------------------------------------------------------------
# Session State & Callback Handlers (Runs BEFORE widget instantiation)
# ---------------------------------------------------------------------------
if "composed_sentence" not in st.session_state:
    st.session_state["composed_sentence"] = "ኢትዮጵያ ውስጥ የቡና"


def append_word_and_refresh(word):
    """Callback: Appends chosen candidate word BEFORE widgets instantiate."""
    current = st.session_state.get("composed_sentence", "")
    st.session_state["composed_sentence"] = f"{current} {word}".strip()


def reset_sentence():
    """Callback: Clears the sentence buffer."""
    st.session_state["composed_sentence"] = ""


def load_example(example_text: str):
    """Callback: Sets the sentence buffer to a preset example."""
    st.session_state["composed_sentence"] = example_text


# Sidebar controls
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/7/71/Flag_of_Ethiopia.svg", width=80)
    st.header("⚙️ Model Configuration")
    st.info("Course: NLP Assignment 1\nInstructor: Dr. Fantahun B.\nCorpus: GPAC (General-Purpose Amharic Corpus)")
    
    top_k = st.slider("Suggestions Count (Top-k)", min_value=3, max_value=8, value=5)
    smoothing_alpha = st.slider("Smoothing Alpha (k)", min_value=0.01, max_value=1.0, value=0.1, step=0.01)
    
    st.divider()
    st.subheader("💡 Example Sentences")
    preset_examples = [
        "ኢትዮጵያ ውስጥ የቡና",
        "አዲስ አበባ የኢትዮጵያ",
        "የትምህርት ጥራት ማሻሻል",
        "ጠቅላይ ሚኒስትር ዶክተር",
        "የህዝብ ተወካዮች ምክር"
    ]
    for ex in preset_examples:
        st.button(
            f"📝 {ex}",
            key=f"ex_{ex}",
            on_click=load_example,
            args=(ex,),
            use_container_width=True
        )

st.markdown('<div class="main-title">🇪🇹 Amharic Predictive Text & Next-Word Suggestion System</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Addis Ababa University | College of Technology & Built Environment | NLP Task #4</div>', unsafe_allow_html=True)

# Main layout
col_left, col_right = st.columns([3, 2])

with col_left:
    st.subheader("1. Amharic Sentence Input")
    
    # Bound directly to key="composed_sentence"
    user_text = st.text_area(
        "Type or click suggestions below to compose your sentence:",
        key="composed_sentence",
        height=110,
        help="Type Amharic text or select suggestions below. The model extracts the last 1-2 words and predicts completions."
    )

    col_btn1, col_btn2 = st.columns([1, 1])
    with col_btn1:
        # Primary Action Button named "🔮 Predict"
        st.button("🔮 Predict", type="primary", use_container_width=True)
    with col_btn2:
        # Reset button using callback
        st.button("🔄 Reset Sentence", on_click=reset_sentence, use_container_width=True)

    # Retrieve candidate next words based on current sentence
    current_text = st.session_state.get("composed_sentence", "")
    suggestions = predictor.predict_next_words(current_text, top_k=top_k, smoothing_alpha=smoothing_alpha)

    st.subheader("2. Candidate Next-Word Suggestions")
    st.write("👉 **Click any suggestion below to append it and continue composing the sentence:**")

    if suggestions:
        cols = st.columns(len(suggestions))
        for idx, (word, prob) in enumerate(suggestions):
            with cols[idx]:
                pct = prob * 100.0
                st.button(
                    f"#{idx+1} {word}\n({pct:.1f}%)",
                    key=f"btn_sugg_{idx}_{word}",
                    on_click=append_word_and_refresh,
                    args=(word,),
                    use_container_width=True
                )
    else:
        st.info("Enter at least one Amharic word above and click 'Predict' to receive next-word suggestions.")

with col_right:
    st.subheader("📊 Candidate Probabilities & Confidence")
    if suggestions:
        st.write("Ranked by probability estimated from the GPAC Language Model:")
        for idx, (word, prob) in enumerate(suggestions, 1):
            pct = prob * 100.0
            st.markdown(f"**#{idx} {word}** — `{pct:.1f}% confidence` (P = {prob:.4f})")
            # Streamlit native progress bar: renders full UTF-8 Unicode Amharic in browser
            st.progress(min(prob, 1.0))
    else:
        st.info("Awaiting input to generate candidate probability distribution.")

    st.divider()
    st.subheader("🔍 Context Analysis")
    tokens = tokenize_amharic_sentence(current_text)
    st.write(f"• **Tokens in Current Sentence:** `{len(tokens)}`")
    if tokens:
        st.write(f"• **Extracted Context (Last 1-2 words):** `{' '.join(tokens[-2:])}`")
        if 2 in predictor.models and 3 in predictor.models:
            st.success("Language Model Engine: Trigram with Bigram/Unigram Katz Backoff")
