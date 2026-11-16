import os
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma

DB_DIR = "./vector_db"
RESULTS_DIR = "search_results"

os.makedirs(RESULTS_DIR, exist_ok=True)

print("Ініціалізація embedding-моделі...")
embeddings_model = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

print("Завантаження векторного сховища з диска...")
vectorstore = Chroma(
    persist_directory=DB_DIR,
    embedding_function=embeddings_model
)

queries = [
    "Як штучний інтелект допомагає оптимізувати рутинні завдання викладача ЗВО?",
    "Які існують ґрунтовні відмінності між штучним інтелектом та машинним навчання?",
    "Які особливості використання штучного інтелекту в діяльності підприємств?",
    "Які правові та етичні аспекти використання ШІ в контексті міжнародного врегулювання?",
    "Які виклики чи бар'єри виникають при інтеграції ШІ у навчальний процес та бізнес?"
]

print(" ПОШУК ТА ЗБЕРЕЖЕННЯ РЕЗУЛЬТАТІВ")

for i, query in enumerate(queries, 1):
    print(f"Запит {i}: \"{query}\"")
    
    results = vectorstore.similarity_search(query, k=2)
    
    file_content = f"Запит {i}: {query}\n\nРезультати пошуку:\n" + "="*50 + "\n\n"
    
    for idx, doc in enumerate(results, 1):
        source = doc.metadata.get("source", "Невідомо")
        page = doc.metadata.get("page", 0)
        chunk_file = doc.metadata.get("chunk_file", "Невідомо")
        
        print(f"  [{idx}] Джерело: {source} (стор. {page}, файл чанка: {chunk_file})")
        print(f"      Текст фрагмента: \"{doc.page_content.strip()[:180]}...\"\n")
        
        file_content += f"[{idx}] Джерело: {source} (стор. {page}, файл чанка: {chunk_file})\n"
        file_content += f"Текст фрагмента:\n{doc.page_content.strip()}\n\n" + "-"*50 + "\n\n"
    
    output_filename = f"query_{i}.txt"
    output_path = os.path.join(RESULTS_DIR, output_filename)
    
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(file_content)
        
    print(f"Збережено у файл: {output_path}")

print(f"\nГотово! Усі результати успішно збережено у папку: {os.path.abspath(RESULTS_DIR)}")