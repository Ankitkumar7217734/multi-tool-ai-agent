# Deploying to Streamlit Community Cloud

This app runs on the free [Streamlit Community Cloud](https://streamlit.io/cloud). It deploys straight from a GitHub repo.

## Files in this folder

| File | Purpose |
|------|---------|
| `app.py` | The Streamlit app (entry point) |
| `requirements.txt` | Pinned Python dependencies Streamlit installs on deploy |
| `.streamlit/config.toml` | Theme and server settings |
| `.streamlit/secrets.toml.example` | Template for the optional preset key |
| `.gitignore` | Keeps secrets, venvs, and backups out of the repo |

## Before you push: rename this folder

The folder is currently named `Langchain-tool-and-agents ` with a **trailing space**, which breaks git paths and URLs. Rename it to something clean, for example `multi-tool-ai-agent`, before creating the repo.

## Step 1: Put the code on GitHub

The simplest path is to make **this folder the repository root** so `requirements.txt` and `.streamlit/` sit at the top level.

```bash
cd multi-tool-ai-agent        # the renamed folder
git init
git add .
git commit -m "Multi-tool AI agent (Streamlit)"
git branch -M main
git remote add origin https://github.com/<you>/multi-tool-ai-agent.git
git push -u origin main
```

If instead you keep the app inside a larger repo, Streamlit Cloud looks for `requirements.txt` at the **repository root**. Either move it there or keep the app folder as its own repo.

> Double-check `git status` shows no `.env` and no `secrets.toml` before pushing.

## Step 2: Create the app on Streamlit Cloud

1. Go to https://share.streamlit.io and sign in with GitHub.
2. Click **Create app -> Deploy a public app from GitHub**.
3. Set:
   - **Repository:** `<you>/multi-tool-ai-agent`
   - **Branch:** `main`
   - **Main file path:** `app.py` (or the full path if nested)
4. (Optional) Under **Advanced settings**, pick **Python 3.11**.

## Step 3: Handle the Groq API key

The app reads the Groq key from the sidebar text box. You have two options:

- **Public demo (recommended):** leave secrets empty. Every visitor pastes their own Groq key, so you never pay for their usage.
- **Private / personal:** set the key once so it is pre-filled. In the app's **Settings -> Secrets**, paste:

  ```toml
  GROQ_API_KEY = "gsk_your_key_here"
  ```

  `app.py` reads this via `st.secrets` and pre-fills the sidebar. Anyone who can open the app then uses your quota, so only do this for a private deployment.

A Groq key is free at https://console.groq.com.

## Step 4: Deploy

Click **Deploy**. First build takes a few minutes while dependencies install. After that, pushes to `main` redeploy automatically.

## Notes

- `yfinance` is included so the stock-price tool works. If a deploy ever fails on it, it is optional: the app drops that one tool and still runs.
- `arxiv >= 2.1.0` is required (pinned to 4.0.0 here) to avoid HTTP 301 errors from the older API.
- No system packages are needed, so there is no `packages.txt`.
