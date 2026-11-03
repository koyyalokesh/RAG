import sys
from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

sys.stdout.reconfigure(encoding="utf-8")

HERE = Path(__file__).parent

#print(HERE)

#load the documents

documents = []
for path in sorted((HERE / "data").glob("*.md")):
    documents.append(
        Document(
            page_content=path.read_text(encoding="utf-8"), metadata={"source":path.name}
        )
    )

print(f"Loaded {len(documents)} documents")

for document in documents:
    print(
        f" {document.metadata['source']:15}"
        f"{len(document.page_content)} characters"
    )

print()

splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=50,
    separators=["\n\n", "\n", ". ", " ", ""],
)

chunks = splitter.split_documents(documents)

print(f"{len(documents)} documents became {len(chunks)} chunks")

for index, chunk in enumerate(chunks[:4], start = 1):
    print(f"---- chunk {index} from {chunk.metadata['source']}, {len(chunk.page_content)} chars")
    print(chunk.page_content)
    print()

