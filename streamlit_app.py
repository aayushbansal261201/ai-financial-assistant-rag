import streamlit as st
import tempfile

from groq import Groq
from dotenv import load_dotenv
import os

from utils.pdf_loader import load_pdf
from utils.text_splitter import split_documents
from utils.embeddings import get_embedding_model
from utils.retriever import create_vector_store

# Load environment variables
load_dotenv()

# Get API key
api_key = os.getenv("GROQ_API_KEY")

# Initialize Groq client
client = Groq(api_key=api_key)

# Streamlit page config
st.set_page_config(
    page_title="AI Financial Assistant",
    page_icon="🤖",
    layout="wide"
)

# App title
st.title("🤖 AI Financial Assistant")

st.write("Upload a PDF and ask questions from it.")

# Session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Upload PDF
uploaded_file = st.file_uploader(
    "Upload PDF",
    type="pdf"
)

# Process uploaded PDF
if uploaded_file is not None:

    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".pdf"
    ) as tmp_file:

        tmp_file.write(uploaded_file.read())

        temp_pdf_path = tmp_file.name

    st.success("PDF uploaded successfully!")

    # Load PDF
    documents = load_pdf(temp_pdf_path)

    # Split documents into chunks
    chunks = split_documents(documents)

    # Load embedding model
    embedding_model = get_embedding_model()

    # Create vector store
    vector_store = create_vector_store(
        chunks,
        embedding_model
    )

    st.success("Vector database created successfully!")

    # Display previous messages
    for message in st.session_state.messages:

        with st.chat_message(message["role"]):

            st.markdown(message["content"])

    # Chat input
    user_question = st.chat_input(
        "Ask a question from the PDF"
    )

    # If user enters question
    if user_question:

        # Store user message
        st.session_state.messages.append(
            {
                "role": "user",
                "content": user_question
            }
        )

        # Display user message
        with st.chat_message("user"):

            st.markdown(user_question)

        # Retrieve relevant chunks
        retrieved_docs = vector_store.similarity_search(
            user_question,
            k=3
        )

        # Combine retrieved context
        context = "\n\n".join(
            [doc.page_content for doc in retrieved_docs]
        )

        # Source references
        sources = []

        for doc in retrieved_docs:

            page_number = doc.metadata.get(
                "page",
                "Unknown"
            )

            if page_number != "Unknown":

                sources.append(
                    f"Page {page_number + 1}"
                )

        # Build conversation history
        conversation_history = ""

        for message in st.session_state.messages[-6:]:

            role = message["role"]

            content = message["content"]

            conversation_history += (
                f"{role}: {content}\n"
            )

        # Prompt
        prompt = f"""
        You are an AI Financial Assistant.

        Use the conversation history and provided context
        to answer the user's question.

        Answer ONLY from the provided context.

        Conversation History:
        {conversation_history}

        Context:
        {context}

        Current Question:
        {user_question}
        """

        # Generate AI response
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )

        # Extract answer
        answer = response.choices[0].message.content

        # Store assistant response
        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": answer
            }
        )

        # Display assistant response
        with st.chat_message("assistant"):

            st.markdown(answer)

            st.markdown("### 📚 Sources")

            for source in set(sources):

                st.write(source)