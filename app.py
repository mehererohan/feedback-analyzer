import os
import json
from datetime import datetime, timezone
from typing import Any, Optional, Tuple

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from collections import Counter
from dotenv import load_dotenv
from groq import Groq
from supabase import Client, create_client

load_dotenv()
client = Groq(
    api_key=os.getenv("GROQ_API_KEY") or st.secrets.get("GROQ_API_KEY", "")
)


@st.cache_resource
def get_supabase() -> Optional[Client]:
    url = (os.getenv("SUPABASE_URL") or st.secrets.get("SUPABASE_URL", "")).strip()
    key = (os.getenv("SUPABASE_KEY") or st.secrets.get("SUPABASE_KEY", "")).strip()
    if not url or not key:
        return None
    return create_client(url, key)


def analysis_title(feedback: str) -> str:
    text = feedback.strip()
    return text[:50] if text else ""


def format_ts_short(value: Any) -> str:
    if value is None:
        return ""
    s = str(value)
    if len(s) >= 19:
        return s[:19].replace("T", " ")
    return s


def save_analysis(feedback: str, data: dict) -> Optional[int]:
    sb = get_supabase()
    if not sb:
        return None
    title = analysis_title(feedback)
    created_at = datetime.now(timezone.utc).isoformat()
    payload = json.dumps(data, ensure_ascii=False)
    row = {
        "created_at": created_at,
        "title": title,
        "feedback_text": feedback,
        "results_json": payload,
    }
    resp = sb.table("analyses").insert(row).execute()
    inserted = resp.data
    if not inserted:
        raise RuntimeError("Supabase insert returned no rows")
    return int(inserted[0]["id"])


def list_analyses(limit: int = 100):
    sb = get_supabase()
    if not sb:
        return []
    resp = (
        sb.table("analyses")
        .select("id, created_at, title, feedback_text, results_json")
        .order("id", desc=True)
        .limit(limit)
        .execute()
    )
    return list(resp.data or [])


def get_analysis(analysis_id: int):
    sb = get_supabase()
    if not sb:
        return None
    resp = (
        sb.table("analyses")
        .select("id, created_at, title, feedback_text, results_json")
        .eq("id", analysis_id)
        .limit(1)
        .execute()
    )
    rows = resp.data or []
    return rows[0] if rows else None


def list_analyses_for_trends(limit: int = 5000):
    sb = get_supabase()
    if not sb:
        return []
    resp = (
        sb.table("analyses")
        .select("id, created_at, results_json")
        .order("created_at", desc=False)
        .limit(limit)
        .execute()
    )
    return list(resp.data or [])


SENTIMENT_SCORE = {"positive": 3, "mixed": 2, "neutral": 1, "negative": 0}

SENTIMENT_COLORS = {
    "positive": "🟢",
    "negative": "🔴",
    "neutral": "🟡",
    "mixed": "🟠",
}


def render_results(data: dict):
    col1, col2, col3 = st.columns(3)
    col1.metric("Reviews analyzed", data["review_count"])
    col2.metric("Themes found", len(data["themes"]))
    col3.metric(
        "Overall sentiment",
        f"{SENTIMENT_COLORS.get(data['overall_sentiment'], '')} {data['overall_sentiment']}",
    )

    st.markdown("### Summary")
    st.write(data["summary"])

    st.markdown("### Top themes")
    for theme in data["themes"]:
        icon = SENTIMENT_COLORS.get(theme["sentiment"], "⚪")
        with st.expander(f"{icon} {theme['title']} — {theme['sentiment']}"):
            st.write(theme["description"])
            st.caption(f'"{theme["example_quote"]}"')


def _parse_analysis_row(results_json: str) -> Optional[dict]:
    if not results_json:
        return None
    try:
        return json.loads(results_json)
    except json.JSONDecodeError:
        return None


def build_trends_series(rows: list) -> Tuple[list, list, list]:
    """Returns (timestamps, scores, labels) for valid rows only."""
    times: list = []
    scores: list = []
    labels: list = []
    for row in rows:
        data = _parse_analysis_row(row.get("results_json") or "")
        if not data:
            continue
        raw = (data.get("overall_sentiment") or "").strip().lower()
        if raw not in SENTIMENT_SCORE:
            continue
        ts = row.get("created_at")
        if ts is None:
            continue
        times.append(ts)
        scores.append(SENTIMENT_SCORE[raw])
        labels.append(raw)
    return times, scores, labels


def collect_theme_counts(rows: list) -> Counter:
    counter: Counter = Counter()
    for row in rows:
        data = _parse_analysis_row(row.get("results_json") or "")
        if not data:
            continue
        for theme in data.get("themes") or []:
            title = (theme.get("title") or "").strip()
            if title:
                counter[title] += 1
    return counter


def render_trends_dashboard():
    sb = get_supabase()
    if sb is None:
        st.warning(
            "Set **SUPABASE_URL** and **SUPABASE_KEY** in your environment or `.env` file "
            "to load trends."
        )
        return
    try:
        rows = list_analyses_for_trends()
    except Exception as exc:
        st.error(f"Could not load analyses from Supabase: {exc}")
        return
    if not rows:
        st.info("No saved analyses yet. Run an analysis on the **Analyzer** tab first.")
        return

    times, scores, labels = build_trends_series(rows)
    theme_counter = collect_theme_counts(rows)

    valid_scores = scores
    n_total = len(rows)
    avg_sent = sum(valid_scores) / len(valid_scores) if valid_scores else 0.0

    m1, m2 = st.columns(2)
    m1.metric("Total analyses", f"{n_total:,}")
    m2.metric(
        "Average sentiment",
        f"{avg_sent:.2f}" if valid_scores else "—",
        help="Mean of numeric scores: negative=0, neutral=1, mixed=2, positive=3. "
        "Only runs with a mappable overall_sentiment are included.",
    )
    if not valid_scores and n_total:
        st.caption(
            "No mappable **overall_sentiment** in stored JSON; average shown as — until data includes "
            "positive, negative, neutral, or mixed."
        )

    st.markdown("### Sentiment over time")
    if not times:
        st.caption("No rows with a mappable overall_sentiment in stored JSON.")
    else:
        fig_line = go.Figure()
        fig_line.add_trace(
            go.Scatter(
                x=times,
                y=scores,
                mode="lines+markers",
                name="Sentiment score",
                text=labels,
                hovertemplate=(
                    "%{x}<br>"
                    "Score: %{y}<br>"
                    "Sentiment: %{text}<extra></extra>"
                ),
            )
        )
        fig_line.update_layout(
            yaxis=dict(
                title="Sentiment (numeric)",
                tickmode="array",
                tickvals=[0, 1, 2, 3],
                ticktext=["Negative (0)", "Neutral (1)", "Mixed (2)", "Positive (3)"],
                range=[-0.2, 3.2],
            ),
            xaxis_title="Created at (UTC)",
            hovermode="x unified",
            height=400,
            margin=dict(l=40, r=20, t=40, b=40),
        )
        st.plotly_chart(fig_line, use_container_width=True)

    st.markdown("### Most common themes")
    if not theme_counter:
        st.caption("No theme titles found in stored results.")
    else:
        top_n = theme_counter.most_common(25)
        themes_rev = [t for t, _ in reversed(top_n)]
        counts_rev = [c for _, c in reversed(top_n)]
        fig_bar = go.Figure(
            go.Bar(
                x=counts_rev,
                y=themes_rev,
                orientation="h",
                hovertemplate="%{y}<br>Count: %{x}<extra></extra>",
            )
        )
        fig_bar.update_layout(
            xaxis_title="Number of analyses mentioning theme",
            yaxis_title="",
            height=max(320, 28 * len(top_n)),
            margin=dict(l=20, r=20, t=40, b=40),
        )
        st.plotly_chart(fig_bar, use_container_width=True)


if "analysis_results" not in st.session_state:
    st.session_state.analysis_results = None

with st.sidebar:
    st.header("Past analyses")
    sb = get_supabase()
    if sb is None:
        st.warning("Set **SUPABASE_URL** and **SUPABASE_KEY** in your environment or `.env` file.")
        rows = []
    else:
        try:
            rows = list_analyses()
        except Exception as exc:
            st.error(f"Could not load history from Supabase: {exc}")
            rows = []
    if sb is not None and not rows:
        st.caption("Run an analysis to build history.")
    elif rows:
        for row in rows:
            label = format_ts_short(row.get("created_at"))
            if row.get("title"):
                label = f"{label} — {row['title']}"
            if st.button(label, key=f"load_hist_{row['id']}", use_container_width=True):
                loaded = get_analysis(row["id"])
                if loaded:
                    st.session_state.analysis_results = {
                        "data": json.loads(loaded["results_json"]),
                        "feedback": loaded["feedback_text"],
                        "loaded_id": loaded["id"],
                        "loaded_at": format_ts_short(loaded.get("created_at")),
                    }

st.title("Customer Feedback Analyzer")

tab_analyzer, tab_trends = st.tabs(["Analyzer", "Trends"])

with tab_analyzer:
    tab1, tab2 = st.tabs(["Paste text", "Upload CSV"])

    feedback = ""

    with tab1:
        feedback = st.text_area("Paste reviews here — one per line", height=200)

    with tab2:
        uploaded_file = st.file_uploader("Upload a CSV file", type="csv")
        if uploaded_file:
            df = pd.read_csv(uploaded_file)
            df.index = df.index + 1
            st.write("Preview:", df.head())
            if len(df.columns) == 1:
                column = df.columns[0]
            else:
                column = st.selectbox("Which column contains the reviews?", df.columns)
            feedback = "\n".join(df[column].dropna().astype(str).tolist())

    review_count = len([r for r in feedback.strip().splitlines() if r.strip()])

    if st.button("Analyze"):
        if not feedback.strip():
            st.warning("Please paste some feedback or upload a CSV first.")
        else:
            with st.spinner("Analyzing..."):
                response = client.chat.completions.create(
                    model="llama-3.3-70b-versatile",
                    messages=[
                        {
                            "role": "user",
                            "content": f"""Analyze this customer feedback. Reply ONLY with a JSON object, no explanation, no markdown.

{{
  "summary": "2-3 sentence overall summary",
  "overall_sentiment": "positive|negative|neutral|mixed",
  "review_count": {review_count},
  "themes": [
    {{
      "title": "Theme name",
      "sentiment": "positive|negative|neutral",
      "description": "1-2 sentence description",
      "example_quote": "short example from the feedback"
    }}
  ]
}}

3-5 themes. JSON only.

FEEDBACK:
{feedback[:8000]}""",
                        }
                    ],
                )

                raw = response.choices[0].message.content
                clean = raw.replace("```json", "").replace("```", "").strip()

                try:
                    data = json.loads(clean)
                    if get_supabase() is None:
                        st.error(
                            "Supabase is not configured; analysis was not saved. "
                            "Set SUPABASE_URL and SUPABASE_KEY."
                        )
                    else:
                        try:
                            save_analysis(feedback, data)
                        except Exception as exc:
                            st.error(f"Could not save analysis to Supabase: {exc}")
                    st.session_state.analysis_results = {
                        "data": data,
                        "feedback": feedback,
                        "loaded_id": None,
                        "loaded_at": None,
                    }
                except json.JSONDecodeError:
                    st.session_state.analysis_results = None
                    st.markdown(raw)

    if st.session_state.analysis_results:
        res = st.session_state.analysis_results
        if res.get("loaded_id"):
            st.caption(
                f"Showing saved analysis #{res['loaded_id']} — {res.get('loaded_at', '')} UTC"
            )
        st.markdown("---")
        st.markdown("## Results")
        render_results(res["data"])

with tab_trends:
    st.markdown("## Trends")
    render_trends_dashboard()
