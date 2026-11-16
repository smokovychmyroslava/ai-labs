import os
from langchain_core.documents import Document
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma

CHUNKS_DIR = "chunks"
DB_DIR = "./vector_db"

print("Ініціалізація embedding-моделі (all-MiniLM-L6-v2)...")
embeddings_model = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

print("Зчитування чанків та їхніх метаданих з файлів...")
documents = []

for root, dirs, files in os.walk(CHUNKS_DIR):
    for file in sorted(files):
        if file.endswith(".txt"):
            file_path = os.path.join(root, file)
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
            
            lines = content.split("\n")
            source_file = "unknown"
            page_num = 0
            
            for line in lines:
                if line.startswith("Source:"):
                    source_file = line.replace("Source:", "").strip()
                elif line.startswith("Page:"):
                    try:
                        page_num = int(line.replace("Page:", "").strip())
                    except ValueError:
                        page_num = 0

            parts = content.split("\n\n", 3)
            text_content = parts[3] if len(parts) > 3 else content

            doc = Document(
                page_content=text_content,
                metadata={"source": source_file, "page": page_num, "chunk_file": file}
            )
            documents.append(doc)

print(f"Знайдено та підготовлено чанків для бази: {len(documents)}")

print("Створення векторного сховища Chroma та збереження у базі...")
vectorstore = Chroma.from_documents(
    documents=documents,
        embedding=embeddings_model,
    persist_directory=DB_DIR
)

print(f"\nГотово! Векторне сховище успішно створене та збережене у папці: {os.path.abspath(DB_DIR)}")