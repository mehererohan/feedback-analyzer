import streamlit as st
import os
import json
import pandas as pd
from groq import Groq
from dotenv import load_dotenv

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

st.title("Customer Feedback Analyzer")

SENTIMENT_COLORS = {
    "positive": "🟢",
    "negative": "🔴",
    "neutral": "🟡",
    "mixed": "🟠"
}

tab1, tab2 = st.tabs(["Paste text", "Upload CSV"])

feedback = ""

with tab1:
    feedback = st.text_area("Paste reviews here — one per line", height=200)

with tab2:
    uploaded_file = st.file_uploader("Upload a CSV file", type="csv")
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.write("Preview:", df.head())
        column = st.selectbox("Which column contains the reviews?", df.columns)
        feedback = "\n".join(df[column].dropna().astype(str).tolist())

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
  "review_count": <integer>,
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
{feedback[:8000]}"""
                    }
                ]
            )

            raw = response.choices[0].message.content
            clean = raw.replace("```json", "").replace("```", "").strip()

            try:
                data = json.loads(clean)

                col1, col2, col3 = st.columns(3)
                col1.metric("Reviews analyzed", data["review_count"])
                col2.metric("Themes found", len(data["themes"]))
                col3.metric("Overall sentiment", f"{SENTIMENT_COLORS.get(data['overall_sentiment'], '')} {data['overall_sentiment']}")

                st.markdown("### Summary")
                st.write(data["summary"])

                st.markdown("### Top themes")
                for theme in data["themes"]:
                    icon = SENTIMENT_COLORS.get(theme["sentiment"], "⚪")
                    with st.expander(f"{icon} {theme['title']} — {theme['sentiment']}"):
                        st.write(theme["description"])
                        st.caption(f'"{theme["example_quote"]}"')

            except json.JSONDecodeError:
                st.markdown(raw)