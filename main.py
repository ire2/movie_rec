from transformers import GPT2LMHeadModel, GPT2Tokenizer
from dotenv import load_dotenv
import os
import time
import streamlit as st
from openai import OpenAI


# Load environment variables from .env
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY)

# Set the cache directory
CACHE_DIR = os.getenv("CACHE_DIR", "./cache")
os.makedirs(CACHE_DIR, exist_ok=True)

# Streamlit caching


@st.cache_resource()
def load_local_model(model_dir="./gpt2-movie-finetuned"):
    try:
        tokenizer = GPT2Tokenizer.from_pretrained(model_dir)
        model = GPT2LMHeadModel.from_pretrained(model_dir)
        model.eval()
        return tokenizer, model
    except Exception as e:
        st.error(f"Error loading local model: {e}")
        return None, None


def generate_local_answer(prompt, tokenizer, model, max_length=150, temperature=0.7):
    inputs = tokenizer.encode(prompt, return_tensors="pt")
    outputs = model.generate(
        inputs,
        max_length=max_length,
        do_sample=True,
        temperature=temperature,
        top_k=50,
        top_p=0.95,
        num_return_sequences=1,
    )
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return answer


def generate_openai_answer(prompt, max_tokens=150, temperature=0.7):
    try:
        response = client.chat.completions.create(model="gpt-3.5-turbo",
                                                  messages=[
                                                      {"role": "user", "content": prompt}],
                                                  temperature=temperature,
                                                  max_tokens=max_tokens)
        answer = response.choices[0].message.content
        return answer
    except Exception as e:
        return f"Error using OpenAI: {e}"

# CSS Styling


def local_css(file_name: str):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Main function and UX


def main():
    st.set_page_config(page_title="Movie Hero Chatbot", page_icon="🎬")

    local_css(os.path.join(os.path.dirname(__file__), "../static/styles.css"))
    st.markdown(
        '<div class="header">Movie Hero Chatbot <i class="fa-solid fa-film"></i></div>', unsafe_allow_html=True)
    st.write(
        "Welcome to Movie Hero Chatbot! Choose your preferred backend from the sidebar.")

    backend = st.sidebar.radio(
        "Chatbot Backend", ("Local GPT-2 Model", "OpenAI API"))

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    prompt = st.chat_input("What movie question do you have?")
    if prompt:
        with st.chat_message("user"):
            st.markdown(prompt)
        st.session_state.chat_history.append(
            {"role": "user", "content": prompt})

        with st.spinner("Generating answer..."):
            if backend == "Local GPT-2 Model":
                tokenizer, model = load_local_model()
                if tokenizer is None or model is None:
                    answer = "Local GPT-2 model not available."
                else:
                    answer = generate_local_answer(prompt, tokenizer, model)
            else:
                answer = generate_openai_answer(prompt)

        message_placeholder = st.empty()
        full_response = ""
        for chunk in answer.split():
            full_response += chunk + " "
            time.sleep(0.02)
            message_placeholder.markdown(full_response + "▌")
        message_placeholder.markdown(full_response)

        with st.chat_message("assistant"):
            st.markdown(full_response)
        st.session_state.chat_history.append(
            {"role": "assistant", "content": full_response})

        # st.experimental_rerun()


if __name__ == "__main__":
    main()
