import os
from langchain_community.embeddings import SentenceTransformerEmbeddings

CHUNKS_DIR = "chunks"
EMBEDDINGS_DIR = "embeddings"

os.makedirs(EMBEDDINGS_DIR, exist_ok=True)

print("Ініціалізація embedding-моделі (all-MiniLM-L6-v2)...")
embeddings_model = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

print("\nГенерація embeddings та форматування (по 10 чисел у рядку)...\n")

total_embeddings = 0

for root, dirs, files in os.walk(CHUNKS_DIR):
    for file in sorted(files):
        if file.endswith(".txt"):
            file_path = os.path.join(root, file)
            
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
            
            vector = embeddings_model.embed_query(content)
            
            rounded_vector = [round(float(val), 2) for val in vector]
            
            relative_path = os.path.relpath(root, CHUNKS_DIR)
            output_subfolder = os.path.join(EMBEDDINGS_DIR, relative_path)
            os.makedirs(output_subfolder, exist_ok=True)
            
            output_filename = os.path.splitext(file)[0] + ".json"
            output_path = os.path.join(output_subfolder, output_filename)
            
            items_per_line = 10
            lines = []
            for i in range(0, len(rounded_vector), items_per_line):
                chunk_slice = rounded_vector[i:i + items_per_line]
                lines.append("        " + ", ".join(map(str, chunk_slice)))
            
            embedding_block = "[\n" + ",\n".join(lines) + "\n    ]"
            
            json_content = f"""{{\n    "source_chunk": "{file}",\n    "dimension": {len(rounded_vector)},\n    "embedding": {embedding_block}\n}}"""
            
            with open(output_path, "w", encoding="utf-8") as out_f:
                out_f.write(json_content)
                
            print(f"Збережено: {file}")
            total_embeddings += 1

print(f"\nГотово! Успішно опрацьовано та збережено embeddings: {total_embeddings}")