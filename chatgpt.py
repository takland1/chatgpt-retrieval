import os
import sys

from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.document_loaders import DirectoryLoader, TextLoader, JSONLoader
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain.indexes import VectorstoreIndexCreator
from langchain.indexes.vectorstore import VectorStoreIndexWrapper
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.prompts import MessagesPlaceholder, ChatPromptTemplate

# Create a file nearby called constants.py and add it to .gitignore
# In this file, write:
# APIKEY = "<your key>"
# Get yours here: https://platform.openai.com/api-keys
import constants

os.environ["OPENAI_API_KEY"] = constants.APIKEY

# List available models - these are changed quite often
# Installation - pip install openai
# from openai import OpenAI
# for model in OpenAI().models.list().data:
#    print(model.id)
# sys.exit()

### Configuration

# Enable to save to disk & reuse the model (for repeated queries on the same data)
PERSIST = False
persistDir = "Bin/ChromaLangchainDb"

# Start with small models, then you can improve
modelForEmbeddings = "text-embedding-3-small"  # to text-embedding-3-large
modelForQA = "gpt-4o-mini"  # to gpt-4 or gpt-4o

### First stage - preparing the index

embeddings = OpenAIEmbeddings(model=modelForEmbeddings)

if PERSIST and os.path.exists(persistDir):
    print("Reusing index...\n")
    vectorstore = Chroma(persist_directory=persistDir, embedding_function=embeddings)
    index = VectorStoreIndexWrapper(vectorstore=vectorstore)
else:
    print("Building index...\n")

    # GLOB note: GLOB doesn't work as expected in DirectoryLoader.
    # Pattern like "**/obj/**" excludes only files and first-level subfolder,
    #     but leaving the subfolder content included.
    # To work around that, I had to repeat this pattern for all levels
    srcLoader = DirectoryLoader("Source/",
                                exclude=["*.sln", "*.Config", "*.DotSettings", "*.suo", "*.user", "*.cache", "*.json",
                                         "*.png",
                                         "*.svg", "*.ico", "*.drawio", "*.nswag",
                                         "**/.vs/**", "**/.vs/*", "**/.vs/**/*", "**/.vs/**/**/*", "**/.vs/**/**/**/*",
                                         "**/.vs/**/**/**/**/*",
                                         "**/obj/**", "**/obj/*", "**/obj/**/*", "**/obj/**/**/*", "**/obj/**/**/**/*",
                                         "**/obj/**/**/**/**/*",
                                         "**/bin/**", "**/bin/*", "**/bin/**/*", "**/bin/**/**/*", "**/bin/**/**/**/*",
                                         "**/bin/**/**/**/**/*"],
                                use_multithreading=True)
    ciLoader = DirectoryLoader("CI/", exclude=["*.json"], use_multithreading=True)
    docsLoader = DirectoryLoader("Docs/", exclude=["*.svg", "*.zip", "*.drawio"], use_multithreading=True)
    testDataLoader = DirectoryLoader("TestData/", exclude=["*.mdf", "*.zip", "*.http", "*.json"],
                                     use_multithreading=True)
    testDataJsonLoader = DirectoryLoader("TestData/", glob="*.json", loader_cls=JSONLoader, use_multithreading=True)
    readmeLoader = TextLoader("README.md")

    if PERSIST:
        index = VectorstoreIndexCreator(embedding=embeddings, vectorstore_cls=Chroma,
                                        vectorstore_kwargs={"persist_directory": persistDir}).from_loaders(
            [srcLoader, readmeLoader])
    else:
        index = VectorstoreIndexCreator(embedding=embeddings).from_loaders(
            [srcLoader, ciLoader, docsLoader, testDataLoader, testDataJsonLoader])

### Building the conversational chain
# Basically, it consists of two parts:
# 1. Retrieving relevant documents
# 2. Answering the question based on the retrieved documents

# For the whole chain we will use this model
llm = ChatOpenAI(model=modelForQA, temperature=0)  # temperature: 0.0 means deterministic, 1.0 means random

# I have 'Contributing.md' in the docs, so I ask ChatGpt to follow it when appropriate
retriever_system_prompt = (
    "Given a chat history and the latest user question "
    "which might reference context in the chat history, "
    "formulate a standalone question which can be understood "
    "without the chat history. If the question is about code, include the whole classes. "
    "If the question asks about code, include the contribution guide. "
    "Do NOT answer the question, "
    "just reformulate it if needed and otherwise return it as is."
)

retriever_combined_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", retriever_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

# Configure the retriever with history awareness
history_aware_retriever = create_history_aware_retriever(
    llm,
    # k: Amount of documents to return (Default: 4), 1 means only the most relevant document
    index.vectorstore.as_retriever(search_kwargs={"k": 4}),
    retriever_combined_prompt
)

# Configure the question-answering system. Here we also instruct the model to use the contribution guide
qa_system_prompt = (
    "You are an assistant for question-answering tasks related to the codebase of the application"
    "Use the following pieces of retrieved context to answer "
    "the question. If you don't know the answer, say that you "
    "don't know. Use three sentences maximum and keep the "
    "answer concise, but if you asked to write code, write as much as you need"
    "and by default write in C# using the latest language features and follow the contributing guide."
    "\n\n"
    "{context}"
)

qa_combined_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", qa_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

question_answer_chain = create_stuff_documents_chain(llm, qa_combined_prompt)

# In-memory store for chat histories
store = {}


def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]


# Assembling the chain
conversational_rag_chain = RunnableWithMessageHistory(
    create_retrieval_chain(history_aware_retriever, question_answer_chain),
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history",
    output_messages_key="answer",
)

### Running the chat
query = None
if len(sys.argv) > 1:
    query = sys.argv[1]

while True:
    if not query:
        query = input("Prompt: ")
    if query in ['quit', 'q', 'exit']:
        sys.exit()
    result = conversational_rag_chain.invoke({"input": query}, config={
        "configurable": {"session_id": "abc123"}
    })
    print(result['answer'])
    print()

    query = None