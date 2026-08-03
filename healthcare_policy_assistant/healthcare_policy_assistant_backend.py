#Importing libraries
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
#from langchain_community.chains.conversational_retrieval import ConversationalRetrievalChain
from langchain_community.embeddings import HuggingFaceEmbeddings
#from langchain_huggingface import HuggingFaceEmbeddings
from langchain_mistralai import ChatMistralAI
from langchain_classic.chains.retrieval import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
import os

load_dotenv()

mistral_api_key = os.getenv("MISTRAL_API_KEY")

#file_path = "cms_medicare_policy.pdf"
embedding_model = "sentence-transformers/all-MiniLM-L6-v2"

#Loading the pdf document and splitting it into manageable chunks
def load_and_chunk_document(file_path):
    loader = PyPDFLoader(file_path)
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200       # overlap helps with context at boundaries
    )

    chunks = splitter.split_documents(docs)
    return chunks

#Embeddings & Vector Store: Convert text to vectors and store them
def vector_store(chunks):
  embeddings = HuggingFaceEmbeddings(model_name=embedding_model)
  vectorstore = FAISS.from_documents(chunks, embeddings)

  return vectorstore


#Creating prompt template
SYSTEM_PROMPT = """
You are a helpful healthcare policy analyst assistant.
Answer the question using ONLY the context provided below.
If the answer is not found in the context, say:
"I could not find this information in the uploaded policy document.Kindly ask me question related to the policy document"
Be concise, accurate, and cite specific details when possible.

Context:
{context}
"""

prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    ("human", "{input}"),
])


#Building the chain
def get_qa_chain(vectorstore):
    llm = ChatMistralAI(model="mistral-small-latest",temperature=0.2,mistral_api_key=mistral_api_key)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
    document_chain = create_stuff_documents_chain(llm,prompt) #Sending prompt to LLM
    retrieval_chain = create_retrieval_chain(retriever, document_chain) #Retrieve the data that is closely related to the user question from the vector store and provide it to the LLM as part of the prompt.
    return retrieval_chain


# chunks = load_and_chunk_document(file_path)
# vectorstore = vector_store(chunks)
# retrieval_chain = get_qa_chain(vectorstore)

# response = retrieval_chain.invoke({"input":"What kind of services are included and excluded from Inpatient hospital services?"})
# print (response["answer"])