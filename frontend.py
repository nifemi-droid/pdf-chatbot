import streamlit as st
import requests

st.set_page_config(page_title="Chat with your PDF", page_icon="📄")
st.title("📄 Chat with your PDF")

# Initialize session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "pdf_uploaded" not in st.session_state:
    st.session_state.pdf_uploaded = False

# Sidebar for PDF Upload
with st.sidebar:
    st.header("Upload Document")
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
    
    if st.button("Ingest PDF"):
        if uploaded_file is not None:
            with st.spinner("Processing PDF..."):
                files = {"file": (uploaded_file.name, uploaded_file, "application/pdf")}
                response = requests.post("http://127.0.0.1:8000/upload", files=files)
                
                if response.status_code == 200:
                    st.success(response.json()["message"])
                    st.session_state.pdf_uploaded = True
                else:
                    st.error("Error processing PDF.")
        else:
            st.warning("Please upload a file first.")

# Main Chat Interface
if st.session_state.pdf_uploaded:
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Ask a question about your document..."):
        # Display user message
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Get AI response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = requests.get(f"http://127.0.0.1:8000/chat?query={prompt}")
                answer = response.json()["answer"]
                st.markdown(answer)
                
        st.session_state.chat_history.append({"role": "assistant", "content": answer})
else:
    st.info("👈 Please upload a PDF in the sidebar to start chatting.")