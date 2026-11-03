
import sys
from pathlib import Path

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()


# print. This line makes the output UTF-8.
sys.stdout.reconfigure(encoding="utf-8")

HERE = Path(__file__).parent
STORE_PATH = HERE / "chroma_store"

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

#1. Load the documents

documents = []
for path in sorted((HERE / "data").glob("*.md")):
    documents.append(
        Document(
            page_content=path.read_text(encoding="utf-8"), metadata={"source": path.name}
        )
    )

#2. split the documents into chunks

chunks = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50).split_documents(documents)

#3. embedding the chunks and store it to chroma db

store = Chroma.from_documents(
    documents = chunks,
    embedding = embeddings,
    collection_name="rag_demo",
    persist_directory=str(STORE_PATH)
)

retriever = store.as_retriever(search_kwargs={"k":3})

print(f"Indexed {len(documents)} documents as {len(chunks)} chunks")
print()

prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are a support agent. Answer only from the context below. "
     "If the context does not cover the question, say you do not have that "
     "information and offer to connect a human. Keep it to three lines.\n\n"
     "Context:\n{context}"),
    ("human", "{question}"),
])

model = ChatOpenAI(model="gpt-4o-mini", temperature=0)

def format_context(found):
    """
    Turn retrieved Documents into one block of text.

    The Retriever gives us:

        List[Document]

    But the prompt needs normal text.

    So this function converts:

        Document 1
        Document 2
        Document 3

    into something like:

        [refunds.md] Refunds take 5 to 7 working days...

        [shipping.md] Delivery to small towns takes...

    This formatted text becomes the {context} in the prompt.
    """

    return "\n\n".join(
        f"[{doc.metadata['source']}] {doc.page_content}"
        for doc in found
    )

chain = (
    {
        "context": retriever | format_context,
        "question": RunnablePassthrough()  # keep the original question unchanged
    }
    # question --> retriever -->   List[Document]

    # Insert context + question into our prompt template.
    | prompt

    # Send the completed prompt to the LLM.
    | model

    # Convert the LLM response into a normal string.
    | StrOutputParser()
)

questions = [

    # Should retrieve refund policy.
    "How long does a refund take?",

    # Should retrieve shipping policy.
    "I live in a small town, when will my order arrive?",

    # Should retrieve warranty information.
    "My charger broke after eight months, is that covered?",

    # Nothing in our documents talks about EMI.
    # This is intentionally an unsupported question.
    "Do you sell laptops on EMI?",
]

# question --> retriver --> relevant chunks --> format_context() --> prompt --> openLLM -->Stroutparser --> answer printing
for question in questions:
    print("Q:", question)
    print("A:", chain.invoke(question))
    sources = {
        doc.metadata["source"]
        for doc in retriever.invoke(question)
    }


    print(
        "   sources:",
        ", ".join(sorted(sources))
    )

    print("-" * 70)
