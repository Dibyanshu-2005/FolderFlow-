import os
from typing import List, Dict, Any
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, UnstructuredPowerPointLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain
import google.generativeai as genai

# Set your Google API key
os.environ["GOOGLE_API_KEY"] = "AIzaSyCKhfAvlwzxF0DONzqGWbs6goMvn8iEkwE"

# Configure Gemini
genai.configure(api_key=os.environ["GOOGLE_API_KEY"])


class DocumentManager:
    def __init__(self, folder_path: str):
        self.folder_path = folder_path
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        self.processed_files = []
        self.qa_chain = None
        self.chat_history = []

    def process_file(self, file_path: str) -> List[str]:
        """Process a single file and return its chunks"""
        print(f"\nAttempting to process: {file_path}")
        try:
            if file_path.endswith('.pdf'):
                print("Processing as PDF...")
                loader = PyPDFLoader(file_path)
            elif file_path.endswith('.docx'):
                print("Processing as DOCX...")
                loader = Docx2txtLoader(file_path)
            elif file_path.endswith(('.pptx', '.ppt')):
                print("Processing as PowerPoint...")
                loader = UnstructuredPowerPointLoader(file_path)
            else:
                print(f"Unsupported file type: {file_path}")
                return []

            print("Loading document...")
            documents = loader.load()
            print(f"Document loaded, splitting into chunks...")
            chunks = self.text_splitter.split_documents(documents)
            print(f"Created {len(chunks)} chunks")
            self.processed_files.append(file_path)
            print(f"Successfully processed {file_path}")
            return chunks

        except Exception as e:
            print(f"Error processing {file_path}: {str(e)}")
            print(f"Error type: {type(e)}")
            return []

    def setup_qa_system(self):
        """Initialize the QA system with processed documents"""
        print("\nSetting up QA system...")
        print(f"Looking for files in: {self.folder_path}")
        print("Files found:")
        for root, _, files in os.walk(self.folder_path):
            for file in files:
                print(f"- {file}")

        try:
            all_chunks = []
            for root, _, files in os.walk(self.folder_path):
                for file in files:
                    if file.endswith(('.pdf', '.docx', '.pptx', '.ppt')):
                        file_path = os.path.join(root, file)
                        chunks = self.process_file(file_path)
                        all_chunks.extend(chunks)

            print(f"\nTotal files processed: {len(self.processed_files)}")

            if not all_chunks:
                print("No documents were successfully processed!")
                return False

            print("Initializing embeddings...")
            embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

            # Create a persistent directory for Chroma
            persist_directory = os.path.join(os.path.dirname(self.folder_path), 'chroma_db')
            os.makedirs(persist_directory, exist_ok=True)

            print("Creating vector store...")
            vectorstore = Chroma.from_documents(
                documents=all_chunks,
                embedding=embeddings,
                persist_directory=persist_directory
            )

            print("Setting up QA chain...")
            self.qa_chain = ConversationalRetrievalChain.from_llm(
                ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.7),
                vectorstore.as_retriever(),
                return_source_documents=True
            )
            print("QA system setup complete!")
            return True

        except Exception as e:
            print(f"Error setting up QA system: {str(e)}")
            print(f"Error type: {type(e)}")
            return False

    def ask_question(self, question: str) -> Dict:
        """Ask a question and get a response with source information"""
        if not self.qa_chain:
            return {"error": "QA system not initialized. Run setup_qa_system() first."}

        try:
            # Using the new invoke method instead of calling directly
            result = self.qa_chain.invoke({
                "question": question,
                "chat_history": self.chat_history
            })

            # Format source information
            sources = []
            for doc in result["source_documents"]:
                source_file = doc.metadata.get('source', 'Unknown source')
                if source_file not in sources:
                    sources.append(source_file)

            # Update chat history
            self.chat_history.append((question, result["answer"]))

            return {
                "answer": result["answer"],
                "sources": sources
            }

        except Exception as e:
            return {"error": f"Error processing question: {str(e)}"}


def main():
    st.set_page_config(page_title="FolderFlow Document Assistant", page_icon="📚")

    st.title("FolderFlow Document Assistant 🤖")

    # Sidebar for folder selection
    st.sidebar.header("Settings")
    folder_path = st.sidebar.text_input(
        "Enter Documents Folder Path",
        value="/Users/rajat74/Downloads/OneDrive_1_1-6-2025"  # Your default path
    )

    # Initialize session state
    if 'manager' not in st.session_state:
        st.session_state.manager = None
        st.session_state.system_ready = False

    # Setup button
    if st.sidebar.button("Initialize System"):
        with st.spinner("Setting up the document management system..."):
            manager = DocumentManager(folder_path)
            if manager.setup_qa_system():
                st.session_state.manager = manager
                st.session_state.system_ready = True
                st.sidebar.success("System initialized successfully!")
            else:
                st.sidebar.error("Failed to initialize the system. Please check the folder path.")

    # Main chat interface
    if st.session_state.system_ready:
        # Display processed files
        if st.session_state.manager.processed_files:
            with st.expander("Processed Files"):
                for file in st.session_state.manager.processed_files:
                    st.text(f"✓ {os.path.basename(file)}")

        # Chat interface
        st.markdown("### Ask me anything about your documents!")

        # Initialize chat history in session state if it doesn't exist
        if 'chat_history' not in st.session_state:
            st.session_state.chat_history = []

        # Display chat history
        for q, a in st.session_state.chat_history:
            st.text_area("Question", q, height=50, disabled=True)
            st.text_area("Answer", a, height=100, disabled=True)
            st.markdown("---")

        # Question input
        question = st.text_input("Your question:", key="question_input")

        if st.button("Ask"):
            if question:
                with st.spinner("Thinking..."):
                    response = st.session_state.manager.ask_question(question)

                    if "error" in response:
                        st.error(response["error"])
                    else:
                        # Display answer
                        st.text_area("Answer", response["answer"], height=150)

                        # Display sources
                        st.markdown("**Sources:**")
                        for source in response["sources"]:
                            st.markdown(f"- {os.path.basename(source)}")

                        # Add to chat history
                        st.session_state.chat_history.append((question, response["answer"]))

                        # Clear the question input
                        st.session_state.question_input = ""
    else:
        st.info("Please initialize the system using the sidebar controls.")


if __name__ == "__main__":
    main()