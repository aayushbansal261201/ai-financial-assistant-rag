import streamlit as st
import tempfile

from groq import Groq
from dotenv import load_dotenv
import os

from utils.pdf_loader import load_pdf
from utils.text_splitter import split_documents
from utils.embeddings import get_embedding_model
from utils.retriever import create_vector_store


load_dotenv()


api_key = os.getenv("GROQ_API_KEY")


client = Groq(api_key=api_key)


st.set_page_config(
    page_title="AI Financial Assistant",
    page_icon="🤖",
    layout="wide"
)


st.title("🤖 AI Financial Assistant")

st.write("Upload a PDF and ask questions from it.")


if "messages" not in st.session_state:
    st.session_state.messages = []


uploaded_file = st.file_uploader(
    "Upload PDF",
    type="pdf"
)


if uploaded_file is not None:

    
    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".pdf"
    ) as tmp_file:

        tmp_file.write(uploaded_file.read())

        temp_pdf_path = tmp_file.name

    st.success("PDF uploaded successfully!")

    
    documents = load_pdf(temp_pdf_path)

    
    chunks = split_documents(documents)

    
    embedding_model = get_embedding_model()

    
    vector_store = create_vector_store(
        chunks,
        embedding_model
    )

    st.success("Vector database created successfully!")

    
    for message in st.session_state.messages:

        with st.chat_message(message["role"]):

            st.markdown(message["content"])

    
    user_question = st.chat_input(
        "Ask a question from the PDF"
    )

    
    if user_question:

        
        st.session_state.messages.append(
            {
                "role": "user",
                "content": user_question
            }
        )

        
        with st.chat_message("user"):

            st.markdown(user_question)

        
        retrieved_docs = vector_store.similarity_search(
            user_question,
            k=3
        )

        
        context = "\n\n".join(
            [doc.page_content for doc in retrieved_docs]
        )

        
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

        
        conversation_history = ""

        for message in st.session_state.messages[-6:]:

            role = message["role"]

            content = message["content"]

            conversation_history += (
                f"{role}: {content}\n"
            )

        
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

        
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )

        
        answer = response.choices[0].message.content

        
        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": answer
            }
        )

        
        with st.chat_message("assistant"):

            st.markdown(answer)

            st.markdown("### 📚 Sources")

            for source in set(sources):

                st.write(source)