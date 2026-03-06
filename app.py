import streamlit as st
import openai
import json
import re
import time
import datetime
import numpy as np
import pandas as pd
from itertools import islice
from collections import Counter
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk

nltk.download("stopwords", quiet=True)
nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)

# Language code to NLTK stopword language name
LANG_MAP = {
    "ar": "arabic", "az": "azerbaijani", "eu": "basque", "bn": "bengali",
    "ca": "catalan", "zh": "chinese", "da": "danish", "nl": "dutch",
    "en": "english", "fi": "finnish", "fr": "french", "de": "german",
    "el": "greek", "he": "hebrew", "hu": "hungarian", "id": "indonesian",
    "it": "italian", "kk": "kazakh", "ne": "nepali", "no": "norwegian",
    "pt": "portuguese", "ro": "romanian", "ru": "russian", "sl": "slovene",
    "es": "spanish", "sv": "swedish", "tg": "tajik", "tr": "turkish"
}

def detect_languages(texts, sample_size=1000):
    from langdetect import detect, LangDetectException
    import random
    sample   = random.sample(texts, min(sample_size, len(texts)))
    detected = set()
    for text in sample:
        try:
            lang = detect(str(text))
            if lang in LANG_MAP:
                detected.add(LANG_MAP[lang])
        except LangDetectException:
            pass
    detected.add("english")
    return detected

def get_multilingual_stopwords(languages):
    combined = set()
    for lang in languages:
        try:
            combined.update(stopwords.words(lang))
        except OSError:
            pass
    return combined

def translate_phrases(client, phrases, target_languages, source_language="english"):
    """Translate phrases into all target languages silently. Returns flat list of all translations."""
    non_english = [l for l in target_languages if l != source_language]
    if not non_english:
        return phrases
    all_phrases = list(phrases)
    lang_list   = ", ".join(non_english)
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": (
            f"Translate each of the following phrases into these languages: {lang_list}.\n"
            "Return a flat JSON array containing ALL translations (not the originals). "
            "One entry per phrase per language. No explanations, no markdown, just the JSON array.\n\n"
            f"Phrases: {json.dumps(phrases)}"
        )}],
        max_tokens=2000
    )
    try:
        translations = json.loads(re.sub(r"```json|```", "", resp.choices[0].message.content.strip()).strip())
        all_phrases.extend(translations)
    except Exception:
        pass  # if translation fails, just use original phrases
    return list(dict.fromkeys(all_phrases))

st.set_page_config(page_title="Query Builder", page_icon="🔍", layout="wide")
st.title("🔍 Query Builder Tool")
st.caption("AI-powered keyword discovery for social media & Brandwatch queries")

# ── Session state ─────────────────────────────────────────────
for key in ["concept_phrases", "phrases_from_objective", "phrases_ai_expansions",
            "approved_keywords", "candidate_keywords",
            "all_similarity_scores", "brandwatch_query", "pipeline_done",
            "df_seed", "collection_name", "confirm_delete"]:
    if key not in st.session_state:
        st.session_state[key] = None

# ── Helpers ───────────────────────────────────────────────────
def batch_list(lst, size):
    it = iter(lst)
    return list(iter(lambda: list(islice(it, size)), []))

def embed_texts(client, texts, batch_size=500):
    vectors = []
    for batch in batch_list(texts, batch_size):
        resp = client.embeddings.create(model="text-embedding-3-small", input=batch)
        vectors.extend([r.embedding for r in resp.data])
    return vectors

def extract_ngrams(texts, max_n=4, freq_pct=0.2, stop_words=None):
    if stop_words is None:
        stop_words = set(stopwords.words("english"))
    ngram_counts = Counter()
    for text in texts:
        tokens = word_tokenize(str(text).lower())
        tokens = [re.sub(r"[^a-z0-9\-]", "", t) for t in tokens]
        tokens = [t for t in tokens if t and t not in stop_words]
        row_ngrams = set()
        for n in range(1, max_n + 1):
            for i in range(len(tokens) - n + 1):
                gram = " ".join(tokens[i:i+n])
                if gram.strip():
                    row_ngrams.add(gram)
        ngram_counts.update(row_ngrams)
    min_count = max(1, int(len(texts) * freq_pct / 100))
    kept = [g for g, c in ngram_counts.items() if c >= min_count]
    return kept, len(ngram_counts), min_count

# ── SIDEBAR ───────────────────────────────────────────────────
with st.sidebar:
    st.header("🔑 Credentials")
    st.caption("Stored in session only — never saved.")
    openai_key = st.text_input("OpenAI API Key", type="password", placeholder="sk-...")
    qdrant_url = st.text_input("Qdrant URL", placeholder="https://xyz.qdrant.io")
    qdrant_key = st.text_input("Qdrant API Key", type="password", placeholder="your-qdrant-key")

    st.divider()
    st.header("⚙️ Settings")
    similarity_threshold = st.slider("Similarity Threshold", 0.20, 0.80, 0.20, 0.05,
        help="Loose 0.20 — Balanced 0.45 — Strict 0.80")
    freq_pct = st.number_input("N-gram Frequency %", min_value=0.01, max_value=5.0, value=0.05, step=0.01,
        help="N-grams must appear in at least this % of rows")

    st.divider()
    st.header("🗑️ Clean Database")
    if st.button("Delete All Collections", use_container_width=True):
        st.session_state["confirm_delete"] = True
    if st.session_state.get("confirm_delete"):
        st.warning("Deletes ALL Qdrant collections. Cannot be undone.")
        c1, c2 = st.columns(2)
        with c1:
            if st.button("Yes, Delete", type="primary", use_container_width=True):
                try:
                    client = QdrantClient(url=qdrant_url, api_key=qdrant_key)
                    cols = client.get_collections().collections
                    for c in cols:
                        client.delete_collection(c.name)
                    st.success(f"Deleted {len(cols)} collection(s).")
                    st.session_state["confirm_delete"] = False
                except Exception as e:
                    st.error(str(e))
        with c2:
            if st.button("Cancel", use_container_width=True):
                st.session_state["confirm_delete"] = False

# ── STEP 1 — Objective ────────────────────────────────────────
st.header("Step 1 — Research Objective")
objective = st.text_area(
    "Describe what you want to find. Include what is relevant and what should be excluded.",
    placeholder="Example: Capture conversations about textile — materials and finished products. Excludes leather.",
    height=100
)

# ── STEP 2 — Upload File ──────────────────────────────────────
st.header("Step 2 — Upload Seed Data")
uploaded_file   = st.file_uploader("Upload Excel (.xlsx) or CSV", type=["xlsx", "csv"])
selected_column = None

if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith(".csv") else pd.read_excel(uploaded_file)
        st.session_state["df_seed"] = df
        min_count = max(1, int(len(df) * freq_pct / 100))
        st.success(f"✅ **{uploaded_file.name}** — {len(df):,} rows, {len(df.columns)} columns")
        selected_column = st.selectbox("Select the column containing your text data", df.columns.tolist())
        st.caption(f"At {freq_pct}% → n-grams must appear in at least {min_count:,} rows")
    except Exception as e:
        st.error(f"Could not read file: {e}")

# ── STEP 3a — Extract Concept Phrases ────────────────────────
st.header("Step 3 — Run Pipeline")

if st.button("🔍 Extract Concept Phrases", type="primary",
             disabled=not (objective and selected_column and openai_key)):
    try:
        client_openai = openai.OpenAI(api_key=openai_key)
        with st.spinner("Extracting concept phrases..."):

            # From objective
            resp_obj = client_openai.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": (
                    "You are helping build a keyword search query. "
                    "Extract as many short concept phrases (2-4 words each) as are genuinely present in this objective. "
                    "Only extract topics and themes the user wants to FIND and INCLUDE. "
                    "Do NOT extract exclusion criteria or anything the user wants to avoid. "
                    "Return only a JSON array of strings, nothing else.\n\n"
                    f"Objective: {objective}"
                )}],
                max_tokens=1000
            )
            from_objective = json.loads(re.sub(r"```json|```", "", resp_obj.choices[0].message.content.strip()).strip())

            # AI expansions
            resp_exp = client_openai.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": (
                    "You are helping expand a keyword search query. "
                    "Based on the research objective below, suggest as many related concept phrases "
                    "(2-4 words each) as are relevant — covering adjacent topics, synonyms, subcategories, "
                    "and related themes that the user did NOT explicitly mention but would likely want to capture. "
                    "Do NOT repeat phrases already in the objective. "
                    "Do NOT include exclusion criteria. "
                    "Return only a JSON array of strings, nothing else.\n\n"
                    f"Objective: {objective}"
                )}],
                max_tokens=1000
            )
            ai_expansions = json.loads(re.sub(r"```json|```", "", resp_exp.choices[0].message.content.strip()).strip())

            st.session_state["phrases_from_objective"] = from_objective
            st.session_state["phrases_ai_expansions"]  = ai_expansions
            st.session_state["pipeline_done"]          = False

    except openai.AuthenticationError:
        st.error("Invalid OpenAI API key.")
    except Exception as e:
        st.error(f"Error: {e}")

# ── STEP 3b — Review & Edit Concept Phrases ──────────────────
if st.session_state.get("phrases_from_objective") is not None:
    st.subheader("Review Concept Phrases")
    st.caption("Edit freely — one phrase per line. Delete a line to remove, type a new line to add. Both lists will be merged into your anchor vector.")

    col_obj, col_ai = st.columns(2)

    with col_obj:
        st.markdown("**📌 From your objective**")
        obj_default = "\n".join(st.session_state["phrases_from_objective"])
        obj_text    = st.text_area("From objective", value=obj_default, height=300,
                                    key="ta_objective", label_visibility="collapsed")

    with col_ai:
        st.markdown("**🤖 AI suggested expansions**")
        ai_default = "\n".join(st.session_state["phrases_ai_expansions"])
        ai_text    = st.text_area("AI expansions", value=ai_default, height=300,
                                   key="ta_expansions", label_visibility="collapsed")

    # Merge both into flat deduplicated list
    obj_lines = [l.strip() for l in obj_text.splitlines() if l.strip()]
    ai_lines  = [l.strip() for l in ai_text.splitlines() if l.strip()]
    updated_phrases = list(dict.fromkeys(obj_lines + ai_lines))

    st.caption(f"{len(obj_lines)} from objective + {len(ai_lines)} AI suggestions = **{len(updated_phrases)} total phrases** going into anchor vector")

    # ── STEP 3c — Run Full Pipeline ───────────────────────────
    if st.button("▶ Run Pipeline with These Phrases", type="primary",
                 disabled=len(updated_phrases) == 0):
        df    = st.session_state["df_seed"]
        texts = df[selected_column].dropna().astype(str).str.strip().replace("", pd.NA).dropna().tolist()

        pipeline_error = None
        try:
            client_openai = openai.OpenAI(api_key=openai_key)
            client_qdrant = QdrantClient(url=qdrant_url, api_key=qdrant_key)

            with st.status("Running pipeline...", expanded=True) as status:
                st.write("🧲 Embedding concept phrases...")
                phrase_vectors = embed_texts(client_openai, updated_phrases)
                anchor_vector  = list(np.mean(phrase_vectors, axis=0))
                st.write("✅ Anchor vector ready.")
                detected_langs_list = []  # will be set after language detection

                st.write(f"📄 Extracting n-grams from {len(texts):,} rows...")
                st.write("🌍 Detecting languages in dataset...")
                detected_langs = detect_languages(texts)
                stop_words     = get_multilingual_stopwords(detected_langs)
                st.write(f"✅ Languages detected: {', '.join(sorted(detected_langs))} — {len(stop_words):,} stopwords loaded.")
                detected_langs_list = sorted(detected_langs)

                st.write("🌐 Translating anchor phrases into detected languages...")
                updated_phrases = translate_phrases(client_openai, updated_phrases, detected_langs)
                st.write(f"✅ Anchor vector will use {len(updated_phrases)} phrases across all languages.")

                ngrams, total, min_ct = extract_ngrams(texts, max_n=4, freq_pct=freq_pct, stop_words=stop_words)
                st.write(f"✅ {total:,} unique n-grams → **{len(ngrams):,} kept** (min {min_ct:,} rows)")

                st.write("☁️ Embedding & uploading to Qdrant...")
                ts    = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                cname = f"run_{ts}"
                existing = [c.name for c in client_qdrant.get_collections().collections]
                if cname in existing:
                    cname += f"_{int(time.time()) % 10000}"
                client_qdrant.create_collection(cname, vectors_config=VectorParams(size=1536, distance=Distance.COSINE))
                st.write(f"Collection: `{cname}`")
                st.session_state["collection_name"] = cname

                ngram_vectors  = embed_texts(client_openai, ngrams)
                prog           = st.progress(0)
                upload_batches = batch_list(list(zip(ngrams, ngram_vectors)), 100)
                for i, batch in enumerate(upload_batches):
                    client_qdrant.upsert(cname, points=[
                        PointStruct(id=abs(hash(g)) % (2**63), vector=v,
                                    payload={"ngram": g, "ngram_length": len(g.split())})
                        for g, v in batch
                    ])
                    prog.progress((i + 1) / len(upload_batches))
                st.write("✅ Uploaded.")

                st.write("🔎 Running similarity search...")
                result     = client_qdrant.query_points(cname, query=anchor_vector,
                                                        limit=min(len(ngrams), 10000), with_payload=True)
                all_scores = {r.payload["ngram"]: round(r.score, 4) for r in result.points}
                above      = {k: v for k, v in all_scores.items() if v >= similarity_threshold}
                st.session_state["all_similarity_scores"] = all_scores
                st.write(f"✅ {len(above):,} candidates above threshold {similarity_threshold:.2f}")

                st.write("🤖 GPT keyword approval...")
                candidates  = list(above.keys())
                MAX_TOKENS  = 25000
                gpt_batches, current, tokens = [], [], 0
                for term in candidates:
                    t = len(term) // 4
                    if tokens + t > MAX_TOKENS and current:
                        gpt_batches.append(current); current = [term]; tokens = t
                    else:
                        current.append(term); tokens += t
                if current:
                    gpt_batches.append(current)

                approved = []
                gpt_prog = st.progress(0)
                for i, batch in enumerate(gpt_batches):
                    st.write(f"Batch {i+1} of {len(gpt_batches)}...")
                    for attempt in range(2):
                        try:
                            r = client_openai.chat.completions.create(
                                model="gpt-4o-mini",
                                messages=[{"role": "user", "content": (
                                    f"Given this research objective: {objective}\n\n"
                                    f"The dataset contains text in the following languages: {', '.join(detected_langs_list)}. "
                                    "A term in ANY of these languages that is relevant to the objective should be kept, regardless of which language it is in. "
                                    "Do not filter out non-English terms just because they are not in English. "
                                    "From the following list of terms, return ONLY those genuinely relevant to the objective. "
                                    "Remove duplicates, near-duplicates, and overly generic terms. "
                                    "Return ONLY a flat JSON array of strings, no other text, no markdown.\n\n"
                                    f"Terms: {json.dumps(batch)}"
                                )}],
                                max_tokens=4000
                            )
                            approved.extend(json.loads(re.sub(r"```json|```", "",
                                r.choices[0].message.content.strip()).strip()))
                            break
                        except Exception:
                            if attempt == 1:
                                st.warning(f"Batch {i+1} skipped after retry.")
                    gpt_prog.progress((i + 1) / len(gpt_batches))

                st.session_state["candidate_keywords"] = list(dict.fromkeys(approved))
                st.session_state["pipeline_done"]      = True
                status.update(
                    label=f"✅ Done — {len(st.session_state['candidate_keywords'])} keywords found",
                    state="complete"
                )

        except openai.AuthenticationError:
            st.error("Invalid OpenAI API key.")
        except Exception as e:
            st.error(f"Error: {e}")

# ── STEP 4 — Review Keywords ──────────────────────────────────
if st.session_state.get("pipeline_done") and st.session_state["candidate_keywords"]:
    st.header("Step 4 — Review & Approve Keywords")

    all_scores = st.session_state["all_similarity_scores"] or {}
    candidates = st.session_state["candidate_keywords"]

    review_threshold = st.slider("Adjust threshold", 0.20, 0.80, similarity_threshold, 0.05,
                                  key="review_threshold")
    filtered = sorted(
        [k for k in candidates if all_scores.get(k, 0) >= review_threshold],
        key=lambda k: all_scores.get(k, 0), reverse=True
    )

    MIN_KEYWORDS = 20
    st.caption(f"**{len(filtered)} keywords** at threshold {review_threshold:.2f} "
               f"{'✅' if len(filtered) >= MIN_KEYWORDS else f'— need at least {MIN_KEYWORDS}'}")

    to_remove      = st.multiselect("Select keywords to remove", options=filtered, default=[])
    final_keywords = [k for k in filtered if k not in to_remove]
    st.caption(f"{len(final_keywords)} keywords will be approved")

    if st.button("✅ Approve Keywords", type="primary", disabled=len(final_keywords) < MIN_KEYWORDS):
        st.session_state["approved_keywords"] = final_keywords
        st.success(f"✅ {len(final_keywords)} keywords approved. Proceed to Step 5.")

# ── STEP 5 — Outputs ──────────────────────────────────────────
if st.session_state.get("approved_keywords"):
    st.header("Step 5 — Outputs")
    approved = st.session_state["approved_keywords"]

    if st.button("⚡ Generate Brandwatch Query", type="primary"):
        if not openai_key:
            st.error("OpenAI API key required.")
        else:
            with st.spinner("Generating Brandwatch query..."):
                try:
                    client_openai = openai.OpenAI(api_key=openai_key)
                    resp = client_openai.chat.completions.create(
                        model="gpt-4o-mini",
                        messages=[{"role": "user", "content": (
                            f"Given this research objective: {objective}\n\n"
                            f"Using only these approved keywords: {', '.join(approved)}\n\n"
                            "Construct a Brandwatch boolean query using OR, AND, and NOT operators. "
                            "Group related terms with OR inside parentheses. "
                            "Use NOT to exclude terms that contradict the objective. "
                            "Return only the query string, nothing else."
                        )}],
                        max_tokens=1000
                    )
                    st.session_state["brandwatch_query"] = resp.choices[0].message.content.strip()
                except Exception as e:
                    st.error(f"Error: {e}")

    if st.session_state.get("brandwatch_query"):
        bw_query = st.session_state["brandwatch_query"]
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Social Media Keywords")
            hashtag_mode = st.toggle("Hashtag mode (#CamelCase)")
            social_list  = "\n".join(
                ["#" + "".join(w.capitalize() for w in kw.split()) for kw in approved]
                if hashtag_mode else approved
            )
            st.text_area("Copy your keywords", value=social_list, height=300, key="social_out")

        with col2:
            st.subheader("Brandwatch Query")
            st.text_area("Copy your query", value=bw_query, height=300, key="bw_out")

        download_txt = f"=== SOCIAL MEDIA KEYWORDS ===\n{chr(10).join(approved)}\n\n=== BRANDWATCH QUERY ===\n{bw_query}\n"
        st.download_button("💾 Download as .txt", data=download_txt,
                           file_name="query_builder_output.txt", mime="text/plain")
