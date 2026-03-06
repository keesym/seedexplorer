import streamlit as st
import openai
import json
import re
import time
import datetime
import numpy as np
import pandas as pd
import asyncio
import concurrent.futures
from itertools import islice
from collections import Counter
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk
import random

nltk.download("stopwords", quiet=True)
nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)

st.set_page_config(page_title="Query Builder", page_icon="🔍", layout="wide")
st.title("🔍 Query Builder Tool")
st.caption("AI-powered multilingual keyword discovery for Brandwatch queries")

# ── Session state ─────────────────────────────────────────────
for key in ["phrases_from_objective", "phrases_ai_expansions", "labeled_df",
            "detected_langs", "selected_langs", "pipeline_done",
            "keywords_by_lang", "approved_by_lang", "brandwatch_query",
            "df_seed", "confirm_delete"]:
    if key not in st.session_state:
        st.session_state[key] = None

# ── Language map ──────────────────────────────────────────────
LANG_MAP = {
    "ar": "arabic", "az": "azerbaijani", "eu": "basque", "bn": "bengali",
    "ca": "catalan", "zh": "chinese", "da": "danish", "nl": "dutch",
    "en": "english", "fi": "finnish", "fr": "french", "de": "german",
    "el": "greek", "he": "hebrew", "hu": "hungarian", "id": "indonesian",
    "it": "italian", "kk": "kazakh", "ne": "nepali", "no": "norwegian",
    "pt": "portuguese", "ro": "romanian", "ru": "russian", "sl": "slovene",
    "es": "spanish", "sv": "swedish", "tg": "tajik", "tr": "turkish"
}
LANG_MAP_INV = {v: k for k, v in LANG_MAP.items()}

# ── Helpers ───────────────────────────────────────────────────
def is_valid_text(text, min_words=5):
    """Return True if text has 5+ real words (not just emojis/links)."""
    text = str(text).strip()
    # Remove URLs
    text = re.sub(r"http\S+|www\.\S+", "", text)
    # Remove emojis
    text = re.sub(r"[^\w\s]", "", text, flags=re.UNICODE)
    words = [w for w in text.split() if len(w) > 1]
    return len(words) >= min_words

def detect_lang_single(text):
    """Detect language of a single text. Returns NLTK lang name or None."""
    from langdetect import detect, LangDetectException
    try:
        code = detect(str(text))
        return LANG_MAP.get(code)
    except LangDetectException:
        return None

def label_dataframe(df, text_col, sample_size=1000):
    """
    Step 1: detect languages from a random sample of valid rows.
    Step 2: label all valid rows async using ThreadPoolExecutor.
    Returns df with new '_lang' column.
    """
    valid_mask = df[text_col].apply(is_valid_text)
    valid_df   = df[valid_mask].copy()

    # Sample for detection
    sample_idx = random.sample(list(valid_df.index), min(sample_size, len(valid_df)))
    sample_texts = valid_df.loc[sample_idx, text_col].tolist()

    # Detect languages present in sample
    detected = set()
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as ex:
        results = list(ex.map(detect_lang_single, sample_texts))
    for r in results:
        if r:
            detected.add(r)
    detected.add("english")

    # Label ALL valid rows async
    all_texts = valid_df[text_col].tolist()
    with concurrent.futures.ThreadPoolExecutor(max_workers=20) as ex:
        labels = list(ex.map(detect_lang_single, all_texts))

    valid_df["_lang"] = labels
    # Rows that couldn't be detected → mark as None (excluded)
    df = df.copy()
    df["_lang"] = None
    df.loc[valid_df.index, "_lang"] = valid_df["_lang"].values

    return df, detected

def get_stopwords(languages):
    combined = set()
    for lang in languages:
        try:
            combined.update(stopwords.words(lang))
        except OSError:
            pass
    return combined

def batch_list(lst, size):
    it = iter(lst)
    return list(iter(lambda: list(islice(it, size)), []))

def embed_texts(client, texts, batch_size=500):
    vectors = []
    for batch in batch_list(texts, batch_size):
        resp = client.embeddings.create(model="text-embedding-3-small", input=batch)
        vectors.extend([r.embedding for r in resp.data])
    return vectors

def extract_ngrams(texts, stop_words, max_n=4, freq_pct=0.05):
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

def translate_phrases_to_language(client, phrases, target_language):
    """Translate a list of phrases into a single target language."""
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": (
            f"Translate each of the following phrases into {target_language}. "
            "Return a flat JSON array of translated strings only, same order, no explanations, no markdown.\n\n"
            f"Phrases: {json.dumps(phrases)}"
        )}],
        max_tokens=2000
    )
    try:
        return json.loads(re.sub(r"```json|```", "", resp.choices[0].message.content.strip()).strip())
    except Exception:
        return []

def gpt_filter_keywords(client, candidates, objective, detected_langs_list):
    """GPT filters a list of candidates for relevance, multilingual aware."""
    MAX_TOKENS  = 25000
    batches, current, tokens = [], [], 0
    for term in candidates:
        t = len(term) // 4
        if tokens + t > MAX_TOKENS and current:
            batches.append(current); current = [term]; tokens = t
        else:
            current.append(term); tokens += t
    if current:
        batches.append(current)

    approved = []
    for batch in batches:
        for attempt in range(2):
            try:
                r = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": (
                        f"Given this research objective: {objective}\n\n"
                        f"The dataset contains text in: {', '.join(detected_langs_list)}. "
                        "Keep terms in ANY of these languages if they are relevant to the objective. "
                        "Do not filter out non-English terms just because they are not in English. "
                        "Return ONLY genuinely relevant terms. Remove duplicates and overly generic terms. "
                        "Return ONLY a flat JSON array of strings, no markdown.\n\n"
                        f"Terms: {json.dumps(batch)}"
                    )}],
                    max_tokens=4000
                )
                approved.extend(json.loads(re.sub(r"```json|```", "", r.choices[0].message.content.strip()).strip()))
                break
            except Exception:
                if attempt == 1:
                    pass
    return list(dict.fromkeys(approved))

def cross_translate_keywords(client, keywords_by_lang, all_languages):
    """
    For each language, find keywords that exist in other languages but not this one,
    translate them in, and merge.
    """
    # Collect all unique keywords across all languages
    all_keywords = set()
    for kws in keywords_by_lang.values():
        all_keywords.update(kws)

    enriched = {}
    for lang in all_languages:
        existing   = set(keywords_by_lang.get(lang, []))
        missing    = all_keywords - existing
        if missing and lang != "english":
            translations = translate_phrases_to_language(client, list(missing), lang)
            existing.update(translations)
        enriched[lang] = list(existing)
    return enriched

# ── SIDEBAR ───────────────────────────────────────────────────
with st.sidebar:
    st.header("🔑 Credentials")
    st.caption("Session only — never saved.")
    openai_key = st.text_input("OpenAI API Key", type="password", placeholder="sk-...")
    qdrant_url = st.text_input("Qdrant URL", placeholder="https://xyz.qdrant.io")
    qdrant_key = st.text_input("Qdrant API Key", type="password", placeholder="your-qdrant-key")

    st.divider()
    st.header("⚙️ Settings")
    similarity_threshold = st.slider("Similarity Threshold", 0.20, 0.80, 0.20, 0.05,
        help="Loose 0.20 — Balanced 0.45 — Strict 0.80")
    freq_pct = st.number_input("N-gram Frequency %", min_value=0.01, max_value=5.0, value=0.05, step=0.01,
        help="Per language slice — n-grams must appear in at least this % of that language's rows")

    st.divider()
    st.header("🗑️ Clean Database")
    if st.button("Delete All Collections", use_container_width=True):
        st.session_state["confirm_delete"] = True
    if st.session_state.get("confirm_delete"):
        st.warning("Deletes ALL Qdrant collections.")
        c1, c2 = st.columns(2)
        with c1:
            if st.button("Yes, Delete", type="primary", use_container_width=True):
                try:
                    client = QdrantClient(url=qdrant_url, api_key=qdrant_key)
                    cols   = client.get_collections().collections
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
        st.success(f"✅ **{uploaded_file.name}** — {len(df):,} rows, {len(df.columns)} columns")
        selected_column = st.selectbox("Select the column containing your text data", df.columns.tolist())
    except Exception as e:
        st.error(f"Could not read file: {e}")

# ── STEP 3 — Extract Concept Phrases ─────────────────────────
st.header("Step 3 — Extract Concept Phrases")

if st.button("🔍 Extract Concept Phrases", type="primary",
             disabled=not (objective and selected_column and openai_key)):
    try:
        client_openai = openai.OpenAI(api_key=openai_key)
        with st.spinner("Extracting concept phrases..."):
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
            from_obj = json.loads(re.sub(r"```json|```", "", resp_obj.choices[0].message.content.strip()).strip())

            resp_exp = client_openai.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": (
                    "You are helping expand a keyword search query. "
                    "Based on the research objective below, suggest as many related concept phrases "
                    "(2-4 words each) as are relevant — covering adjacent topics, synonyms, subcategories, "
                    "and related themes the user did NOT explicitly mention but would likely want to capture. "
                    "Do NOT repeat phrases from the objective. Do NOT include exclusion criteria. "
                    "Return only a JSON array of strings, nothing else.\n\n"
                    f"Objective: {objective}"
                )}],
                max_tokens=1000
            )
            ai_exp = json.loads(re.sub(r"```json|```", "", resp_exp.choices[0].message.content.strip()).strip())

            st.session_state["phrases_from_objective"] = from_obj
            st.session_state["phrases_ai_expansions"]  = ai_exp
            st.session_state["pipeline_done"]          = False

    except openai.AuthenticationError:
        st.error("Invalid OpenAI API key.")
    except Exception as e:
        st.error(f"Error: {e}")

# ── STEP 3b — Edit Concept Phrases ───────────────────────────
if st.session_state.get("phrases_from_objective") is not None:
    st.subheader("Review Concept Phrases")
    st.caption("Edit freely — one phrase per line. Both lists merge into your anchor vector.")

    col_obj, col_ai = st.columns(2)
    with col_obj:
        st.markdown("**📌 From your objective**")
        obj_text = st.text_area("From objective",
            value="\n".join(st.session_state["phrases_from_objective"]),
            height=250, key="ta_obj", label_visibility="collapsed")
    with col_ai:
        st.markdown("**🤖 AI suggested expansions**")
        ai_text = st.text_area("AI expansions",
            value="\n".join(st.session_state["phrases_ai_expansions"]),
            height=250, key="ta_ai", label_visibility="collapsed")

    obj_lines = [l.strip() for l in obj_text.splitlines() if l.strip()]
    ai_lines  = [l.strip() for l in ai_text.splitlines() if l.strip()]
    anchor_phrases = list(dict.fromkeys(obj_lines + ai_lines))
    st.caption(f"{len(obj_lines)} from objective + {len(ai_lines)} AI suggestions = **{len(anchor_phrases)} total anchor phrases**")

    # ── STEP 4 — Detect Languages & Label Data ────────────────
    st.header("Step 4 — Detect Languages")

    if st.button("🌍 Detect Languages & Label Data", type="primary",
                 disabled=len(anchor_phrases) == 0):
        df = st.session_state["df_seed"]
        with st.spinner(f"Labeling {len(df):,} rows — this takes ~2 minutes for large datasets..."):
            try:
                labeled_df, detected = label_dataframe(df, selected_column)
                st.session_state["labeled_df"]     = labeled_df
                st.session_state["detected_langs"] = detected
                st.session_state["pipeline_done"]  = False
            except Exception as e:
                st.error(f"Error during language detection: {e}")

    if st.session_state.get("detected_langs") is not None:
        labeled_df  = st.session_state["labeled_df"]
        detected    = st.session_state["detected_langs"]

        st.subheader("Select Languages to Process")
        st.caption("Only rows from selected languages will be used. Frequency filter applies per language slice.")

        lang_counts = labeled_df["_lang"].value_counts().to_dict()
        selected    = {}
        cols        = st.columns(min(len(detected), 4))
        for i, lang in enumerate(sorted(detected)):
            count = lang_counts.get(lang, 0)
            with cols[i % len(cols)]:
                selected[lang] = st.checkbox(
                    f"{lang.capitalize()} ({count:,} rows)",
                    value=True, key=f"lang_{lang}"
                )

        selected_langs = [l for l, v in selected.items() if v]
        st.session_state["selected_langs"] = selected_langs

        if len(selected_langs) == 0:
            st.warning("Select at least one language to continue.")
        else:
            st.caption(f"**{len(selected_langs)} language(s) selected** — pipeline will run once per language")

            # ── STEP 5 — Run Pipeline ─────────────────────────
            st.header("Step 5 — Run Pipeline")

            if st.button("▶ Run Pipeline", type="primary"):
                df          = st.session_state["labeled_df"]
                all_langs   = selected_langs
                ts          = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

                try:
                    client_openai = openai.OpenAI(api_key=openai_key)
                    client_qdrant = QdrantClient(url=qdrant_url, api_key=qdrant_key)
                    keywords_by_lang = {}

                    for lang in all_langs:
                        st.markdown(f"---\n### 🌐 Processing: **{lang.capitalize()}**")

                        # Slice this language's rows
                        lang_df    = df[df["_lang"] == lang]
                        lang_texts = lang_df[selected_column].dropna().astype(str).tolist()
                        st.write(f"Rows: {len(lang_texts):,}")

                        if len(lang_texts) < 10:
                            st.warning(f"Too few rows for {lang}, skipping.")
                            continue

                        # Translate anchor phrases into this language
                        st.write(f"🌐 Translating anchor phrases into {lang}...")
                        if lang == "english":
                            lang_phrases = anchor_phrases
                        else:
                            translated   = translate_phrases_to_language(client_openai, anchor_phrases, lang)
                            lang_phrases = list(dict.fromkeys(anchor_phrases + translated))
                        st.write(f"✅ {len(lang_phrases)} anchor phrases ready.")

                        # Embed anchor → anchor vector
                        st.write("🧲 Embedding anchor phrases...")
                        phrase_vectors = embed_texts(client_openai, lang_phrases)
                        anchor_vector  = list(np.mean(phrase_vectors, axis=0))
                        st.write("✅ Anchor vector ready.")

                        # N-gram extraction with language-specific stopwords
                        st.write(f"📄 Extracting n-grams...")
                        lang_stopwords = get_stopwords([lang, "english"])
                        ngrams, total, min_ct = extract_ngrams(lang_texts, lang_stopwords, freq_pct=freq_pct)
                        st.write(f"✅ {total:,} n-grams → **{len(ngrams):,} kept** (min {min_ct:,} rows)")

                        if len(ngrams) == 0:
                            st.warning(f"No n-grams survived frequency filter for {lang}. Try lowering the %.")
                            continue

                        # Embed & upload to Qdrant
                        st.write("☁️ Embedding & uploading to Qdrant...")
                        cname    = f"run_{ts}_{lang}"
                        existing = [c.name for c in client_qdrant.get_collections().collections]
                        if cname in existing:
                            cname += f"_{int(time.time()) % 10000}"
                        client_qdrant.create_collection(cname, vectors_config=VectorParams(size=1536, distance=Distance.COSINE))

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

                        # Similarity search
                        st.write("🔎 Similarity search...")
                        result     = client_qdrant.query_points(cname, query=anchor_vector,
                                                                limit=min(len(ngrams), 10000), with_payload=True)
                        all_scores = {r.payload["ngram"]: round(r.score, 4) for r in result.points}
                        above      = [k for k, v in all_scores.items() if v >= similarity_threshold]
                        st.write(f"✅ {len(above):,} candidates above threshold {similarity_threshold:.2f}")

                        # GPT filtering
                        st.write("🤖 GPT filtering...")
                        approved = gpt_filter_keywords(client_openai, above, objective, sorted(detected))
                        st.write(f"✅ **{len(approved)} keywords** approved for {lang.capitalize()}")
                        keywords_by_lang[lang] = approved

                    # Cross-translation
                    st.markdown("---")
                    st.write("🔄 Cross-translating keywords across languages...")
                    enriched = cross_translate_keywords(client_openai, keywords_by_lang, all_langs)
                    st.session_state["keywords_by_lang"] = enriched
                    st.session_state["pipeline_done"]    = True

                    total_kws = sum(len(v) for v in enriched.values())
                    st.success(f"✅ Pipeline complete — {total_kws} keywords across {len(all_langs)} language(s)")

                except openai.AuthenticationError:
                    st.error("Invalid OpenAI API key.")
                except Exception as e:
                    st.error(f"Error: {e}")

# ── STEP 6 — Review Keywords ──────────────────────────────────
if st.session_state.get("pipeline_done") and st.session_state.get("keywords_by_lang"):
    st.header("Step 6 — Review & Approve Keywords")
    st.caption("Keywords grouped by language. Remove any you disagree with, then approve.")

    keywords_by_lang = st.session_state["keywords_by_lang"]
    approved_by_lang = {}
    MIN_TOTAL        = 20

    for lang, keywords in keywords_by_lang.items():
        st.subheader(f"{lang.capitalize()} — {len(keywords)} keywords")
        to_remove = st.multiselect(
            f"Remove from {lang}",
            options=keywords, default=[],
            key=f"remove_{lang}"
        )
        approved_by_lang[lang] = [k for k in keywords if k not in to_remove]
        st.caption(f"{len(approved_by_lang[lang])} keywords kept")

    total_approved = sum(len(v) for v in approved_by_lang.values())
    st.markdown(f"**Total: {total_approved} keywords across all languages**")

    if st.button("✅ Approve All", type="primary", disabled=total_approved < MIN_TOTAL):
        st.session_state["approved_by_lang"] = approved_by_lang
        st.success(f"✅ {total_approved} keywords approved across {len(approved_by_lang)} language(s). Proceed to Step 7.")

# ── STEP 7 — Outputs ──────────────────────────────────────────
if st.session_state.get("approved_by_lang"):
    st.header("Step 7 — Outputs")
    approved_by_lang = st.session_state["approved_by_lang"]
    all_approved     = [kw for kws in approved_by_lang.values() for kw in kws]

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
                            f"Using only these approved keywords (multilingual): {', '.join(all_approved)}\n\n"
                            "Construct a single Brandwatch boolean query using OR, AND, and NOT operators. "
                            "The query should work across all languages — group related terms with OR. "
                            "Use NOT to exclude terms that contradict the objective. "
                            "Return only the query string, nothing else."
                        )}],
                        max_tokens=2000
                    )
                    st.session_state["brandwatch_query"] = resp.choices[0].message.content.strip()
                except Exception as e:
                    st.error(f"Error: {e}")

    if st.session_state.get("brandwatch_query"):
        bw_query = st.session_state["brandwatch_query"]

        # Social media — grouped by language
        st.subheader("Social Media Keywords")
        hashtag_mode = st.toggle("Hashtag mode (#CamelCase)")
        lang_tabs    = st.tabs([l.capitalize() for l in approved_by_lang.keys()])
        for tab, (lang, kws) in zip(lang_tabs, approved_by_lang.items()):
            with tab:
                if hashtag_mode:
                    out = "\n".join(["#" + "".join(w.capitalize() for w in kw.split()) for kw in kws])
                else:
                    out = "\n".join(kws)
                st.text_area(f"{lang}", value=out, height=200, key=f"social_{lang}")

        st.subheader("Brandwatch Query")
        st.text_area("Copy your query", value=bw_query, height=150, key="bw_out")

        # Download
        lines = ["=== SOCIAL MEDIA KEYWORDS ==="]
        for lang, kws in approved_by_lang.items():
            lines.append(f"\n-- {lang.upper()} --")
            lines.extend(kws)
        lines.append("\n\n=== BRANDWATCH QUERY ===")
        lines.append(bw_query)
        st.download_button("💾 Download as .txt", data="\n".join(lines),
                           file_name="query_builder_output.txt", mime="text/plain")
