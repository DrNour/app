# app_glossary.py

import io
from collections import Counter
from itertools import tee

import pandas as pd
import streamlit as st
import spacy
from spacy.cli import download as spacy_download

# ========= NLP BACKEND =========

@st.cache_resource
def get_nlp():
    """
    Load the spaCy model.
    If 'en_core_web_sm' is not installed, download it, then load.
    Cached so it only happens once per session on Streamlit Cloud.
    """
    model_name = "en_core_web_sm"
    try:
        nlp_model = spacy.load(model_name)
    except OSError:
        # Model not installed in the environment: download then load
        spacy_download(model_name)
        nlp_model = spacy.load(model_name)
    return nlp_model

nlp = get_nlp()

# ========= HELPER FUNCTIONS =========

def extract_noun_chunks(texts, min_len=1, max_len=6):
    """
    Extract noun-phrase candidates using spaCy noun_chunks.
    Returns a Counter of {phrase: freq}.
    """
    counter = Counter()
    if nlp is None:
        return counter

    for doc in nlp.pipe(texts, disable=["ner"]):
        for chunk in doc.noun_chunks:
            phrase = chunk.text.strip().lower()
            token_len = len([t for t in chunk if not t.is_space])
            if min_len <= token_len <= max_len:
                counter[phrase] += 1
    return counter


def ngrams(tokens, n):
    iters = tee(tokens, n)
    for i, it in enumerate(iters):
        for _ in range(i):
            next(it, None)
    return zip(*iters)


def extract_ngrams(texts, n=2, min_freq=2):
    """
    Extract n-gram candidates (bigrams, trigrams, etc.).
    Simple frequency-based filtering.
    """
    counter = Counter()
    if nlp is None:
        return counter

    for doc in nlp.pipe(texts, disable=["ner"]):
        tokens = [
            t.text.lower()
            for t in doc
            if not t.is_punct and not t.is_space
        ]
        for ng in ngrams(tokens, n):
            phrase = " ".join(ng)
            counter[phrase] += 1

    filtered = Counter({k: v for k, v in counter.items() if v >= min_freq})
    return filtered


def kwic(texts, phrase, window=60, max_examples=10):
    """
    Very simple KWIC: returns list of text snippets around the phrase.
    """
    phrase_lower = phrase.lower()
    examples = []
    for text in texts:
        lower = text.lower()
        start_idx = 0
        while True:
            idx = lower.find(phrase_lower, start_idx)
            if idx == -1:
                break
            start = max(0, idx - window)
            end = min(len(text), idx + len(phrase) + window)
            examples.append(text[start:end])
            if len(examples) >= max_examples:
                return examples
            start_idx = idx + len(phrase)
    return examples


# ========= STREAMLIT APP =========

st.title("Corpus-Based Glossary Builder")

st.markdown(
    """
This tool builds a **semi-automatic glossary** from domain texts (e.g. tickets, emails, reports).

**Workflow:**
1. Upload a CSV with at least one text column.
2. Extract noun-phrases, bigrams, and trigrams.
3. Review and annotate:
   - include / exclude,
   - add definitions,
   - add translations,
   - add notes.
4. Export the selected entries as a CSV glossary.
"""
)

if nlp is None:
    st.error("spaCy model could not be loaded.")
    st.stop()

# ---- Upload section ----
uploaded_file = st.file_uploader(
    "Upload a CSV file (must contain a text column)",
    type=["csv"]
)

text_col_name = None
texts = None

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    st.write("Preview of uploaded data:")
    st.dataframe(df.head(), use_container_width=True)

    cols = df.columns.tolist()
    text_col_name = st.selectbox(
        "Select the column that contains the ticket/text data:",
        options=cols
    )

    if text_col_name:
        texts = df[text_col_name].dropna().astype(str).tolist()

# ---- Parameter controls ----
st.sidebar.header("Extraction Settings")

min_np_len = st.sidebar.slider("Min tokens in noun phrase", 1, 5, 1)
max_np_len = st.sidebar.slider("Max tokens in noun phrase", 2, 8, 5)

min_bigram_freq = st.sidebar.slider("Min frequency for bigrams", 1, 20, 3)
min_trigram_freq = st.sidebar.slider("Min frequency for trigrams", 1, 20, 3)

st.sidebar.markdown("---")
st.sidebar.write("Adjust these if you get too many / too few candidates.")

# ---- Extraction ----
if texts is not None and st.button("Extract glossary candidates"):
    with st.spinner("Extracting candidates..."):
        noun_chunks = extract_noun_chunks(
            texts,
            min_len=min_np_len,
            max_len=max_np_len
        )
        bigrams = extract_ngrams(
            texts,
            n=2,
            min_freq=min_bigram_freq
        )
        trigrams = extract_ngrams(
            texts,
            n=3,
            min_freq=min_trigram_freq
        )

        rows = []
        for phrase, freq in noun_chunks.most_common():
            rows.append(
                {"candidate": phrase, "freq": freq, "type": "noun_phrase"}
            )

        for phrase, freq in bigrams.most_common():
            rows.append(
                {"candidate": phrase, "freq": freq, "type": "bigram"}
            )

        for phrase, freq in trigrams.most_common():
            rows.append(
                {"candidate": phrase, "freq": freq, "type": "trigram"}
            )

        cand_df = pd.DataFrame(rows)

        if cand_df.empty:
            st.warning("No candidates found with the current settings.")
        else:
            cand_df = cand_df.assign(
                include=True,
                definition="",
                translation="",
                note=""
            )

            st.session_state["cand_df"] = cand_df
            st.session_state["texts"] = texts
            st.success(f"Found {len(cand_df)} candidates.")

# ---- Editor + KWIC + Export ----
if "cand_df" in st.session_state:
    st.subheader("Glossary candidates (edit and annotate)")

    edited_df = st.data_editor(
        st.session_state["cand_df"],
        num_rows="dynamic",
        use_container_width=True,
        key="cand_editor",
        column_config={
            "include": st.column_config.CheckboxColumn("Include"),
            "candidate": st.column_config.TextColumn("Candidate"),
            "freq": st.column_config.NumberColumn("Freq", disabled=True),
            "type": st.column_config.TextColumn("Type", disabled=True),
            "definition": st.column_config.TextColumn("Definition"),
            "translation": st.column_config.TextColumn("Preferred translation"),
            "note": st.column_config.TextColumn("Note / domain / comment")
        }
    )

    st.session_state["cand_df"] = edited_df

    st.subheader("Examples in context (KWIC)")

    col1, col2 = st.columns([1, 2])

    with col1:
        unique_candidates = edited_df["candidate"].unique().tolist()
        selected_phrase = st.selectbox(
            "Choose a candidate to inspect:",
            options=unique_candidates if unique_candidates else []
        )

    with col2:
        if selected_phrase:
            examples = kwic(
                st.session_state["texts"],
                selected_phrase,
                window=60,
                max_examples=10
            )
            if not examples:
                st.info("No examples found for this candidate.")
            else:
                st.write(f"Examples for **{selected_phrase}**:")
                for ex in examples:
                    st.markdown(
                        "- " + ex.replace(
                            selected_phrase,
                            f"**{selected_phrase}**"
                        )
                    )

    st.subheader("Export glossary")

    export_df = edited_df[edited_df["include"] == True].copy()

    csv_buffer = io.StringIO()
    export_df.to_csv(csv_buffer, index=False)
    csv_data = csv_buffer.getvalue()

    st.download_button(
        label="Download selected entries as CSV",
        data=csv_data,
        file_name="glossary.csv",
        mime="text/csv"
    )
