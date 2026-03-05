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
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct
import nltk

nltk.download("stopwords", quiet=True)
nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)

st.set_page_config(page_title="Query Builder", page_icon="🔍", layout="wide")

st.title("🔍 Query Builder Tool")
st.caption("AI-powered keyword discovery for social media & Brandwatch queries")

# ── Session state init ────────────────────────────────────────
for key in ["approved_keywords", "candidate_keywords", "all_similarity_scores",
            "brandwatch_query", "pipeline_done", "df_seed", "collection_name"]:
    if key not in st.session_state:
        st.session_state[key] = None

# ── SIDEBAR ───────────────────────────────────────────────────
with st.sidebar:
    st.header("🔑 Credentials")
    st.caption("Your keys are never stored — session only.")
    openai_key  = st.text_input("OpenAI API Key",  type="password", placeholder="sk-...")
    qdrant_url  = st.text_input("Qdrant URL",       placeholder="https://xyz.qdrant.io")
    qdrant_key  = st.text_input("Qdrant API Key",   type="password", placeholder="your-qdrant-key")

    st.divider()
    st.header("⚙️ Settings")
    similarity_threshold = st.slider(
        "Similarity Threshold",
        min_value=0.20, max_value=0.80, value=0.45, step=0.05,
        help="How closely must a keyword match your objective. Loose ← 0.45 → Strict"
    )
    freq_pct = st.number_input(
        "N-gram Frequency Filter %",
        min_value=0.01, max_value=5.0, value=0.2, step=0.01,
        help="Only keep n-grams appearing in at least this % of rows"
    )

    st.divider()
    st.header("🗑️ Database")
    if st.button("Clean Database", type="secondary", use_container_width=True):
        st.session_state["confirm_delete"] = True

    if st.session_state.get("confirm_delete"):
        st.warning("This will delete ALL collections from your Qdrant instance.")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Yes, Delete All", type="primary", use_container_width=True):
                try:
                    client = QdrantClient(url=qdrant_url, api_key=qdrant_key)
                    cols   = client.get_collections().collections
                    for c in cols:
                        client.delete_collection(c.name)
                    st.success(f"Deleted {len(cols)} collection(s).")
                    st.session_state["confirm_delete"] = False
                except Exception as e:
                    st.error(f"Error: {e}")
        with col2:
            if st.button("Cancel", use_container_width=True):
                st.session_state["confirm_delete"] = False

# ── STEP 1 — Objective ────────────────────────────────────────
st.header("Step 1 — Research Objective")
objective = st.text_area(
    "Describe what you want to find. Include what is relevant and what should be excluded.",
    placeholder="Example: We want to capture all conversations about textile — materials and finished products. Excludes leather.",
    height=100
)

# ── STEP 2 — Upload File ──────────────────────────────────────
st.header("Step 2 — Upload Seed Data")
uploaded_file = st.file_uploader("Upload your Excel (.xlsx) or CSV file", type=["xlsx", "csv"])

selected_column = None
if uploaded_file:
    try:
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
        st.session_state["df_seed"] = df
        st.success(f"✅ **{uploaded_file.name}** — {len(df):,} rows, {len(df.columns)} columns")
        selected_column = st.selectbox("Select the column containing your text data", df.columns.tolist())
        min_count = max(1, int(len(df) * freq_pct / 100))
        st.caption(f"At {freq_pct}% frequency filter → n-grams must appear in at least {min_count:,} rows")
    except Exception as e:
        st.error(f"Could not read file: {e}")

# ── STEP 3 — Run Pipeline ─────────────────────────────────────
st.header("Step 3 — Run Pipeline")

def batch_list(lst, size):
    it = iter(lst)
    return list(iter(lambda: list(islice(it, size)), []))

def embed_texts(client_openai, texts, batch_size=500):
    all_vectors = []
    for batch in batch_list(texts, batch_size):
        resp = client_openai.embeddings.create(model="text-embedding-3-small", input=batch)
        all_vectors.extend([r.embedding for r in resp.data])
    return all_vectors

def extract_ngrams(texts, max_n=4, freq_pct=0.2):
    stop_words   = set(stopwords.words("english"))
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
    return [g for g, c in ngram_counts.items() if c >= min_count], len(ngram_counts), min_count

if st.button("▶ Run Pipeline", type="primary", disabled=not (objective and selected_column and openai_key and qdrant_url and qdrant_key)):
    if not objective:
        st.error("Please enter a research objective.")
    elif not selected_column:
        st.error("Please upload a file and select a column.")
    else:
        df = st.session_state["df_seed"]
        texts = df[selected_column].dropna().astype(str).str.strip().replace("", pd.NA).dropna().tolist()

        try:
            client_openai = openai.OpenAI(api_key=openai_key)
            client_qdrant = QdrantClient(url=qdrant_url, api_key=qdrant_key)

            # Step 1 — Concept phrases
            with st.status("🔍 Extracting concept phrases...", expanded=True) as status:
                resp = client_openai.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": (
                        "You are helping build a keyword search query. "
                        "Extract 3 to 5 short concept phrases (2-4 words each) that represent ONLY the topics "
                        "and themes the user wants to FIND and INCLUDE. "
                        "Do NOT extract exclusion criteria or anything the user wants to avoid. "
                        "Return only a JSON array of strings, nothing else.\n\n"
                        f"Objective: {objective}"
                    )}],
                    max_tokens=200
                )
                concept_phrases = json.loads(re.sub(r"```json|```", "", resp.choices[0].message.content.strip()).strip())
                st.write(f"✅ Concept phrases: `{concept_phrases}`")

                # Step 2 — Anchor vector
                st.write("🧲 Embedding concept phrases...")
                phrase_vectors = embed_texts(client_openai, concept_phrases)
                anchor_vector  = list(np.mean(phrase_vectors, axis=0))
                st.write("✅ Anchor vector ready.")

                # Step 3 — N-grams
                st.write(f"📄 Extracting n-grams from {len(texts):,} rows...")
                ngrams, total_ngrams, min_count = extract_ngrams(texts, max_n=4, freq_pct=freq_pct)
                st.write(f"✅ {total_ngrams:,} unique n-grams → **{len(ngrams):,} kept** after frequency filter (min {min_count:,} rows)")

                # Step 4 — Embed & upload
                st.write("☁️ Embedding n-grams & uploading to Qdrant...")
                ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                collection_name = f"run_{ts}"
                existing = [c.name for c in client_qdrant.get_collections().collections]
                if collection_name in existing:
                    collection_name += f"_{int(time.time()) % 10000}"

                client_qdrant.create_collection(
                    collection_name=collection_name,
                    vectors_config=VectorParams(size=1536, distance=Distance.COSINE)
                )
                st.write(f"Collection: `{collection_name}`")

                ngram_vectors = embed_texts(client_openai, ngrams)
                progress = st.progress(0)
                batches  = batch_list(list(zip(ngrams, ngram_vectors)), 100)
                for i, batch in enumerate(batches):
                    client_qdrant.upsert(
                        collection_name=collection_name,
                        points=[PointStruct(id=abs(hash(g)) % (2**63), vector=v, payload={"ngram": g, "ngram_length": len(g.split())}) for g, v in batch]
                    )
                    progress.progress((i + 1) / len(batches))
                st.write("✅ Uploaded.")
                st.session_state["collection_name"] = collection_name

                # Step 5 — Similarity search
                st.write("🔎 Running similarity search...")
                search_result = client_qdrant.query_points(
                    collection_name=collection_name,
                    query=anchor_vector,
                    limit=min(len(ngrams), 10000),
                    with_payload=True
                )
                all_scores = {r.payload["ngram"]: round(r.score, 4) for r in search_result.points}
                above      = {k: v for k, v in all_scores.items() if v >= similarity_threshold}
                st.write(f"✅ {len(above):,} candidates above threshold {similarity_threshold:.2f}")
                st.session_state["all_similarity_scores"] = all_scores

                # Step 6 — GPT filtering
                st.write("🤖 GPT keyword approval...")
                candidates = list(above.keys())
                MAX_TOKENS = 25000
                batches_gpt, current, tokens = [], [], 0
                for term in candidates:
                    t = len(term) // 4
                    if tokens + t > MAX_TOKENS and current:
                        batches_gpt.append(current); current = [term]; tokens = t
                    else:
                        current.append(term); tokens += t
                if current:
                    batches_gpt.append(current)

                approved = []
                gpt_progress = st.progress(0)
                for i, batch in enumerate(batches_gpt):
                    st.write(f"Processing batch {i+1} of {len(batches_gpt)}...")
                    for attempt in range(2):
                        try:
                            r = client_openai.chat.completions.create(
                                model="gpt-4o-mini",
                                messages=[{"role": "user", "content": (
                                    f"Given this research objective: {objective}\n\n"
                                    "From the following list of terms, return ONLY those genuinely relevant. "
                                    "Remove duplicates, near-duplicates, and overly generic terms. "
                                    "Return ONLY a flat JSON array of strings, no other text, no markdown.\n\n"
                                    f"Terms: {json.dumps(batch)}"
                                )}],
                                max_tokens=4000
                            )
                            approved.extend(json.loads(re.sub(r"```json|```", "", r.choices[0].message.content.strip()).strip()))
                            break
                        except:
                            if attempt == 1:
                                st.warning(f"Batch {i+1} skipped after retry.")
                    gpt_progress.progress((i + 1) / len(batches_gpt))

                st.session_state["candidate_keywords"] = list(dict.fromkeys(approved))
                st.session_state["pipeline_done"]      = True
                status.update(label=f"✅ Pipeline complete — {len(st.session_state['candidate_keywords'])} keywords found", state="complete")

        except openai.AuthenticationError:
            st.error("Invalid OpenAI API key. Check your credentials in the sidebar.")
        except Exception as e:
            st.error(f"Error: {e}")

# ── STEP 4 — Review Keywords ──────────────────────────────────
if st.session_state.get("pipeline_done") and st.session_state["candidate_keywords"]:
    st.header("Step 4 — Review & Approve Keywords")

    all_scores = st.session_state["all_similarity_scores"] or {}
    candidates = st.session_state["candidate_keywords"]

    review_threshold = st.slider(
        "Adjust threshold to expand or narrow the list",
        min_value=0.20, max_value=0.80, value=similarity_threshold, step=0.05,
        key="review_threshold"
    )

    filtered = sorted(
        [k for k in candidates if all_scores.get(k, 0) >= review_threshold],
        key=lambda k: all_scores.get(k, 0), reverse=True
    )

    MIN_KEYWORDS = 20
    st.caption(f"**{len(filtered)} keywords** at threshold {review_threshold:.2f} {'✅' if len(filtered) >= MIN_KEYWORDS else f'— need at least {MIN_KEYWORDS}'}")

    to_remove = st.multiselect(
        "Select keywords to remove (then click Approve below)",
        options=filtered,
        default=[],
        help="Hold Ctrl/Cmd to select multiple"
    )

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
            if hashtag_mode:
                social_list = "\n".join(["#" + "".join(w.capitalize() for w in kw.split()) for kw in approved])
            else:
                social_list = "\n".join(approved)
            st.text_area("Copy your keywords", value=social_list, height=300, key="social_output")

        with col2:
            st.subheader("Brandwatch Query")
            st.text_area("Copy your query", value=bw_query, height=300, key="bw_output")

        # Download
        download_content = f"=== SOCIAL MEDIA KEYWORDS ===\n{chr(10).join(approved)}\n\n=== BRANDWATCH QUERY ===\n{bw_query}\n"
        st.download_button(
            label="💾 Download as .txt",
            data=download_content,
            file_name="query_builder_output.txt",
            mime="text/plain"
        )
