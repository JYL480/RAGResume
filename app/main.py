import google.generativeai as genai
import os
from dotenv import load_dotenv
from process import read_embeddings, retrieve_relevant_resources, prompt_formatter
import streamlit as st

# Load environment variables
load_dotenv()

# Initialize embeddings and resources
embeddings, pages_and_chunks = read_embeddings()

# Function to process the query and generate a response
def generate_response(query):
    # Retrieve relevant resources
    score, indices, time_taken = retrieve_relevant_resources(query, embeddings)
    context_items = [pages_and_chunks[i] for i in indices]
    prompt = prompt_formatter(query, context_items)
    
    # Configure API
    api_key = st.secrets["API_KEY"]
    genai.configure(api_key=api_key)
    
    # Safety settings
    safety_settings = [
        {"category": "HARM_CATEGORY_DANGEROUS", "threshold": "BLOCK_NONE"},
        {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
        {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
        {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
        {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
    ]
    
    # Initialize model
    model = genai.GenerativeModel(model_name='gemini-1.5-flash', safety_settings=safety_settings)
    
    # Generate content
    response = model.generate_content(prompt)
    return response.text

# Streamlit UI setup
st.set_page_config(page_title="Chatbot", layout="wide")
st.title("Lee Jun Yang's RAG Chatbot")
st.subheader("Powered by Gemini LLM")
st.write("Enter your query and get an answer based off Jun Yang's Resume")


# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User input and processing
user_input = st.chat_input("Ask me anything:",placeholder="List me projects he has done!")

if user_input:
    # Append user input to chat history
    st.session_state.messages.append({"role": "user", "content": user_input})
    
    # Generate response
    with st.spinner("Generating response..."):
        response = generate_response(user_input)
    
    # Append bot response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})
    
    # Display user input and bot response
    st.chat_message("user").markdown(user_input)
    st.chat_message("assistant").markdown(response)
