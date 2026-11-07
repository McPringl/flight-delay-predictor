"""
The chat page (Streamlit).

Run from the project folder:   python -m streamlit run app.py

Type about your flight in plain words. The chatbot (chatbot.py) works out the
airline, airports, date and time, asks for anything missing, and answers with
the model's estimate (predictor.py). You can keep chatting to change details.
"""

import streamlit as st

from chatbot import FlightChat
from predictor import AIRLINE_NAMES, load_model, predict_flight

st.set_page_config(page_title="Flight delay chat", page_icon="✈️")
booster, artifacts, metrics = load_model()          # loaded once and cached

st.title("✈️ Flight delay chat")
st.caption(
    f"Estimates from 2023–2024 US flight records. On flights from July–December 2024 "
    f"the model scored AUC {metrics['scores']['model']['auc']:.2f}. "
    "These are likelihoods, not guarantees."
)


def predict_for_chat(slots: dict) -> dict:
    """The chatbot calls this with the flight details it has collected."""
    return predict_flight(slots["carrier"], slots["origin"], slots["dest"],
                          slots["flight_num"], slots["date"], slots["time"])


def new_conversation() -> None:
    bot = FlightChat(
        carriers=artifacts["categories"]["carrier"],
        airports=artifacts["categories"]["origin"],
        predict=predict_for_chat,
        airline_names=AIRLINE_NAMES,
    )
    st.session_state.bot = bot
    st.session_state.messages = [{"role": "assistant", "content": bot.greeting()}]


if "bot" not in st.session_state:
    new_conversation()

with st.sidebar:
    st.write("**Try asking:**")
    st.markdown(
        "- *Delta from Atlanta to LAX tomorrow at 9:30 am*\n"
        "- *Is AA 1234 from Chicago to Miami on Friday at 6 pm late?*\n"
        "- *What if it leaves 2 hours later?*\n"
        "- *Why?*"
    )
    if st.button("New conversation"):
        new_conversation()
        st.rerun()

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Describe your flight…"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        try:
            with st.spinner("Checking…"):
                answer = st.session_state.bot.reply(prompt)
        except Exception as error:               # show the problem instead of crashing the page
            answer = f"Sorry, something went wrong while checking that: {error}"
        st.markdown(answer)
    st.session_state.messages.append({"role": "assistant", "content": answer})
