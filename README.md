# Query Builder Tool — Streamlit App

AI-powered keyword discovery for social media & Brandwatch queries.

## Deploy to Streamlit Community Cloud (Free)

### Step 1 — GitHub
1. Create a free account at https://github.com
2. Create a new repository (e.g. `query-builder`)
3. Upload both files: `app.py` and `requirements.txt`

### Step 2 — Streamlit Community Cloud
1. Go to https://share.streamlit.io
2. Sign in with your GitHub account
3. Click **New app**
4. Select your repository and set Main file path to `app.py`
5. Click **Deploy**

That's it. You'll get a URL like `https://yourname-query-builder.streamlit.app` to share with your team.

## How to Use

1. **Sidebar** — Enter your OpenAI API key, Qdrant URL, and Qdrant API key
2. **Step 1** — Enter your research objective
3. **Step 2** — Upload your Excel or CSV file, select the text column
4. **Step 3** — Click Run Pipeline
5. **Step 4** — Review and approve keywords
6. **Step 5** — Generate Brandwatch query and download outputs

## Credentials Needed

- **OpenAI API key** → https://platform.openai.com/api-keys
- **Qdrant URL + API key** → https://cloud.qdrant.io (free account)
