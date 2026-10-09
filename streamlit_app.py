import streamlit as st
import pandas as pd

st.set_page_config(page_title="Daily Dashboard", page_icon="📊", layout="wide")

st.title("📊 My Daily Dashboard")
st.write("Track your day and see a quick summary in one place.")

with st.sidebar:
    st.header("Your details")
    name = st.text_input("Name", "Ava")
    mood = st.selectbox("How are you feeling today?", ["Happy", "Calm", "Focused", "Tired"])
    focus_hours = st.slider("Focus hours", 0, 12, 4)
    water_cups = st.slider("Water cups", 0, 15, 6)

st.subheader(f"Welcome, {name}!")

col1, col2, col3 = st.columns(3)
col1.metric("Mood", mood)
col2.metric("Focus hours", f"{focus_hours}h")
col3.metric("Water cups", f"{water_cups} cups")

activity_data = pd.DataFrame(
    {
        "Task": ["Deep Work", "Meetings", "Breaks", "Exercise"],
        "Hours": [focus_hours, 2, 1, 1],
    }
)

st.subheader("Today's activity")
st.bar_chart(activity_data.set_index("Task"))

if st.button("Show summary"):
    st.success(
        f"{name}, you have {focus_hours} hours of focus and {water_cups} cups of water today. "
        f"Your current mood is {mood.lower()}."
    )
