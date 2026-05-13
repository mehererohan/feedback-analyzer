import streamlit as st
import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

st.title("Customer Feedback Analyzer")
st.write("Paste customer reviews below — one per line.")

feedback = st.text_area("Customer feedback", height=200)

if st.button("Analyze"):
    if not feedback.strip():
        st.warning("Please paste some feedback first.")
    else:
        with st.spinner("Analyzing..."):
            response = client.chat.completions.create(
               model="llama-3.3-70b-versatile",
                messages=[
                    {
                        "role": "user",
                        "content": f"""Analyze this customer feedback. Identify the top 3-5 themes, 
the sentiment for each (positive/negative/neutral), and an example quote. 
Also give an overall sentiment summary.

FEEDBACK:
{feedback}"""
                    }
                ]
            )
            result = response.choices[0].message.content
            st.markdown(result)
