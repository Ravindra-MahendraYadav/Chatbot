import streamlit as st
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
import qrcode
from PIL import Image
import os

# Function to generate the QR code
def generate_qr_code(url):
    qr = qrcode.QRCode(
        version=1,
        error_correction=qrcode.constants.ERROR_CORRECT_L,
        box_size=10,
        border=4,
    )
    qr.add_data(url)
    qr.make(fit=True)
    img = qr.make_image(fill_color="black", back_color="white")
    return img

def load_document(file):
    name , extension = os.path.splitext(file)
    if extension == ".pdf":
        from langchain.document_loaders import PyPDFLoader
        loader = PyPDFLoader(file)
    elif extension == ".docx":
        from langchain.document_loaders import Docx2txtLoader
        loader = Docx2txtLoader(file)
    elif extension == ".txt":
        from langchain.document_loaders import TextLoader
        loader = TextLoader(file)
    else:
        st.write("Document Type Not supported")
        return None
    data = loader.load()
    return data

def chunk_data(data, chunk_size=256, chunk_overlap=20):
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = text_splitter.split_documents(data)
    return chunks

def create_embeddings(chunks):
    embeddings = OpenAIEmbeddings()
    vector_store = Chroma.from_documents(chunks, embeddings)
    return vector_store

def ask_and_get_answer(vector_store, q, k=3):
    from langchain.chains import RetrievalQA
    from langchain.chat_models import ChatOpenAI
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=1)
    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": k})
    chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)
    answer = chain.run(q)
    return answer

if __name__ == "__main__":
    import os
    from dotenv import load_dotenv
    load_dotenv("Ravindra.env")

    st.image("Meril.jpeg")
    st.subheader("Meril ChatBot")

    # Manually set the URL of your app
    app_url = "https://chatbot-346watdk4jejiujeys8pzk.streamlit.app"  # Change this to your actual app URL if deployed

    # Generate the QR code
    qr_image = generate_qr_code(app_url)

    # Save the QR code image
    qr_image_path = "qr_code.png"
    qr_image.save(qr_image_path)

    # Display the QR code in the left-hand column
    col1, col2 = st.columns([1, 4])
    with col1:
        st.image(qr_image_path, use_column_width=True)

    with col2:
        with st.sidebar:
            api_key = st.text_input("OPENAI API KEY: ", type="password")
            if api_key:
                os.environ["OPENAI_API_KEY"] = api_key
            uploaded_file = st.file_uploader("Upload a File:", type=["pdf", "docx", "txt"])
            chunk_size = st.number_input("Chunk size:", min_value=100, max_value=2048, value=512)
            k = st.number_input("k", min_value=1, max_value=20, value=3)

            add_data = st.button("Add Data")
            if uploaded_file and add_data:
                with st.spinner("Reading, Chunking, and Embedding file..."):
                    bytes_data = uploaded_file.read()
                    file_name = os.path.join("./", uploaded_file.name)
                    with open(file_name, "wb") as f:
                        f.write(bytes_data)

                    data = load_document(file_name)
                    chunks = chunk_data(data, chunk_size=chunk_size)
                    st.write(f'Chunk size: {chunk_size}, Chunks: {len(chunks)}')
                    vector_store = create_embeddings(chunks)

                    st.session_state.vs = vector_store
                    st.success("File Uploaded, Chunked, Embedded Successfully")

    q = st.text_input("Ask any question related to the file uploaded:")
    if q:
        if "vs" in st.session_state:
            vector_store = st.session_state.vs
            st.write(f'k: {k}')
            answer = ask_and_get_answer(vector_store, q, k)
            st.text_area("LLM Answer:", value=answer)
