import os

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

DOCUMENTS_DIR = "documents"
CHUNKS_DIR = "chunks"

os.makedirs(CHUNKS_DIR, exist_ok=True)

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1500,
    chunk_overlap=250,
    separators=["\n\n", "\n", ". ", " ", ""],
    add_start_index=True
)

chunk_counter = 1

for filename in sorted(os.listdir(DOCUMENTS_DIR)):

    if filename.endswith(".pdf"):

        file_path = os.path.join(DOCUMENTS_DIR, filename)
        document_name = os.path.splitext(filename)[0]

        print(f"Обробка: {filename}")

        loader = PyPDFLoader(file_path)
        pages = loader.load()

        chunks = text_splitter.split_documents(pages)

        document_chunks_dir = os.path.join(
            CHUNKS_DIR, document_name
        )

        os.makedirs(document_chunks_dir, exist_ok=True)

        for chunk in chunks:

            chunk_id = f"chunk_{chunk_counter:03d}"

            output_path = os.path.join(
                document_chunks_dir,
                chunk_id + ".txt"
            )

            with open(output_path, "w", encoding="utf-8") as file:

                file.write(f"Chunk ID: {chunk_id}\n")
                file.write(f"Source: {filename}\n")
                file.write(
                    f"Page: {chunk.metadata.get('page', 0) + 1}\n\n"
                )
                file.write(chunk.page_content)

            chunk_counter += 1

        print(f"Створено chunks: {len(chunks)}")

print("\nОбробку завершено!")
print(f"Загальна кількість chunks: {chunk_counter - 1}")
