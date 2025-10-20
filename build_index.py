import os
import json
import time
from typing import List, Dict
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import pickle
from langchain_text_splitters import RecursiveCharacterTextSplitter
import datetime

class VectorIndexBuilder:
    def __init__(self, model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"):
        self.model = SentenceTransformer(model_name)
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
            length_function=len,
            separators=["\n\n", "\n", ". ", "! ", "? ", " ", ""]
        )

    def load_documents(self, knowledge_base_path: str) -> List[Dict]:
        """Загрузка документов из папки"""
        documents = []
        for filename in os.listdir(knowledge_base_path):
            if filename.endswith('.txt'):
                filepath = os.path.join(knowledge_base_path, filename)
                with open(filepath, 'r', encoding='utf-8') as f:
                    content = f.read()
                    documents.append({
                        'content': content,
                        'source': filename,
                        'title': filename.replace('.txt', '')
                    })
        return documents

    def split_documents(self, documents: List[Dict]) -> List[Dict]:
        """Разбиение документов на чанки"""
        all_chunks = []

        for doc in documents:
            chunks = self.text_splitter.split_text(doc['content'])

            for i, chunk in enumerate(chunks):
                chunk_metadata = {
                    'source': doc['source'],
                    'title': doc['title'],
                    'chunk_id': i,
                    'chunk_count': len(chunks),
                    'word_count': len(chunk.split())
                }
                all_chunks.append({
                    'text': chunk,
                    'metadata': chunk_metadata
                })

        return all_chunks

    def generate_embeddings(self, chunks: List[Dict]):
        """Генерация эмбеддингов для чанков"""
        texts = [chunk['text'] for chunk in chunks]
        print(f"Генерация эмбеддингов для {len(texts)} чанков...")

        start_time = time.time()
        embeddings = self.model.encode(texts, show_progress_bar=True)
        end_time = time.time()

        generation_time = end_time - start_time
        print(f"Эмбеддинги сгенерированы за {generation_time:.2f} секунд")

        return embeddings, generation_time

    def build_faiss_index(self, embeddings: np.ndarray):
        """Создание FAISS индекса"""
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatIP(dimension)

        # Нормализуем векторы для косинусного сходства
        faiss.normalize_L2(embeddings)
        index.add(embeddings)

        return index

    def save_index(self, index, chunks: List[Dict], output_path: str, total_time: float, embedding_time: float):
        """Сохранение индекса и метаданных"""
        # Сохраняем FAISS индекс
        faiss.write_index(index, f"{output_path}/faiss.index")

        # Сохраняем метаданные
        with open(f"{output_path}/metadata.pkl", 'wb') as f:
            pickle.dump(chunks, f)

        # Сохраняем информацию о модели и времени
        index_info = {
            'model_name': 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2',
            'embedding_dimension': 384,
            'total_chunks': len(chunks),
            'build_time': time.time(),
            'total_generation_time_seconds': total_time,
            'embedding_generation_time_seconds': embedding_time,
            'chunk_size': 500,
            'chunk_overlap': 50,
            'completion_time': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }

        with open(f"{output_path}/index_info.json", 'w', encoding='utf-8') as f:
            json.dump(index_info, f, indent=2, ensure_ascii=False)

        print(f"✅ Индекс сохранён в {output_path}/")
        print(f"📊 Статистика: {len(chunks)} чанков, размерность 384")
        print(f"⏱️  Общее время создания: {total_time:.2f} секунд")
        print(f"🔢 Время генерации эмбеддингов: {embedding_time:.2f} секунд")

    def search(self, index, chunks: List[Dict], query: str, k: int = 3):
        """Поиск в индексе"""
        query_embedding = self.model.encode([query])
        faiss.normalize_L2(query_embedding)

        distances, indices = index.search(query_embedding, k)

        results = []
        for i, idx in enumerate(indices[0]):
            if idx < len(chunks):
                results.append({
                    'text': chunks[idx]['text'],
                    'metadata': chunks[idx]['metadata'],
                    'score': distances[0][i]
                })

        return results

def main():
    # Конфигурация
    KNOWLEDGE_BASE_PATH = "knowledge_base"
    OUTPUT_PATH = "vector_index"

    # Создаем папку для индекса
    os.makedirs(OUTPUT_PATH, exist_ok=True)

    # Начало общего времени
    total_start_time = time.time()

    # Инициализация билдера
    print("🔄 Инициализация модели эмбеддингов...")
    builder = VectorIndexBuilder()

    # Загрузка документов
    print("📚 Загрузка документов...")
    documents = builder.load_documents(KNOWLEDGE_BASE_PATH)
    print(f"Загружено документов: {len(documents)}")

    # Разбиение на чанки
    print("✂️ Разбиение на чанки...")
    chunks = builder.split_documents(documents)
    print(f"Создано чанков: {len(chunks)}")

    # Генерация эмбеддингов
    print("🔢 Генерация эмбеддингов...")
    embeddings, embedding_time = builder.generate_embeddings(chunks)

    # Построение индекса
    print("🏗️ Построение векторного индекса...")
    index = builder.build_faiss_index(embeddings)

    # Конец общего времени
    total_end_time = time.time()
    total_time = total_end_time - total_start_time

    # Сохранение индекса
    print("💾 Сохранение индекса...")
    builder.save_index(index, chunks, OUTPUT_PATH, total_time, embedding_time)

    # Тестовый поиск
    print("\n🔍 Тестовый поиск...")
    test_queries = [
        "Какое существо ищет блестящие предметы?",
        "Кто изучает магических существ?",
        "Что такое Темносила?"
    ]

    for query in test_queries:
        print(f"\nЗапрос: '{query}'")
        results = builder.search(index, chunks, query, k=2)
        for i, result in enumerate(results):
            print(f"Результат {i+1} (сходство: {result['score']:.4f}):")
            print(f"Источник: {result['metadata']['source']}")
            print(f"Текст: {result['text'][:200]}...")
            print("-" * 50)

if __name__ == "__main__":
    main()