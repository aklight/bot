import os
import json
import faiss
import pickle
import requests
from typing import List, Dict
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

load_dotenv()

class YandexGPTClient:
    def __init__(self):
        self.api_key = os.getenv('YANDEX_API_KEY')
        self.folder_id = os.getenv('YANDEX_FOLDER_ID')
        self.url = "https://llm.api.cloud.yandex.net/foundationModels/v1/completion"

    def generate_response(self, prompt: str, temperature: float = 0.3) -> str:
        """Генерация ответа с помощью YandexGPT"""
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Api-Key {self.api_key}"
        }

        data = {
            "modelUri": f"gpt://{self.folder_id}/yandexgpt-lite",
            "completionOptions": {
                "stream": False,
                "temperature": temperature,
                "maxTokens": 1000
            },
            "messages": [
                {
                    "role": "system",
                    "text": "Ты помощник, который всегда размышляет шаг за шагом перед ответом. Отвечай только на основе предоставленной информации."
                },
                {
                    "role": "user",
                    "text": prompt
                }
            ]
        }

        try:
            response = requests.post(self.url, headers=headers, json=data, timeout=30)
            response.raise_for_status()

            result = response.json()
            return result['result']['alternatives'][0]['message']['text']

        except requests.exceptions.RequestException as e:
            return f"Ошибка при обращении к YandexGPT: {e}"
        except KeyError as e:
            return f"Ошибка при обработке ответа от YandexGPT: {e}"
        except Exception as e:
            return f"Неожиданная ошибка: {e}"

class RAGBot:
    def __init__(self, index_path: str, model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"):
        # Загрузка модели эмбеддингов
        self.embedding_model = SentenceTransformer(model_name)

        # Загрузка векторного индекса
        self.index = faiss.read_index(f"{index_path}/faiss.index")

        # Загрузка метаданных
        with open(f"{index_path}/metadata.pkl", 'rb') as f:
            self.chunks = pickle.load(f)

        # Загрузка информации об индексе
        with open(f"{index_path}/index_info.json", 'r') as f:
            self.index_info = json.load(f)

        # Инициализация YandexGPT клиента
        self.llm_client = YandexGPTClient()

        # Few-shot примеры
        self.few_shot_examples = [
            {
                "question": "Какое существо ищет блестящие предметы?",
                "reasoning": "1. Сначала я поищу информацию о существах, которые ищут блестящие предметы\n2. В документах нашел упоминание о Блескоискателе\n3. В файле 'Блескоискатель.txt' указано, что это существо обладает магической способностью находить блестящие предметы",
                "answer": "Блескоискатель - небольшое чёрное пушистое существо с длинным хоботком, обладающее магической способностью находить блестящие предметы."
            },
            {
                "question": "Кто такой Артем Землеход?",
                "reasoning": "1. Найду информацию об Артеме Землеходе в базе знаний\n2. В документах обнаружено, что он является известным магозоологом\n3. Указано, что он автор фундаментального труда о фантастических существах",
                "answer": "Артем Землеход - известный магозоолог, автор фундаментального труда о фантастических существах. Он изучает магических существ и носит с собой волшебный чемодан с расширенным пространством."
            }
        ]

    def search_similar_chunks(self, query: str, k: int = 3) -> List[Dict]:
        """Поиск похожих чанков в векторной базе"""
        query_embedding = self.embedding_model.encode([query])
        faiss.normalize_L2(query_embedding)

        distances, indices = self.index.search(query_embedding, k)

        results = []
        for i, idx in enumerate(indices[0]):
            if idx < len(self.chunks):
                results.append({
                    'text': self.chunks[idx]['text'],
                    'metadata': self.chunks[idx]['metadata'],
                    'score': distances[0][i]
                })

        return results

    def build_prompt(self, query: str, context_chunks: List[Dict]) -> str:
        """Построение промпта с Few-shot и Chain-of-Thought"""

        # Собираем контекст из найденных чанков
        context = "\n\n".join([f"Источник: {chunk['metadata']['source']}\nТекст: {chunk['text']}"
                              for chunk in context_chunks])

        # Few-shot примеры с reasoning
        few_shot_text = ""
        for example in self.few_shot_examples:
            few_shot_text += f"""
Вопрос: {example['question']}
Размышление: {example['reasoning']}
Ответ: {example['answer']}

"""

        # System prompt с Chain-of-Thought
        prompt = f"""Ты - помощник по магическим существам. Твоя задача - отвечать на вопросы на основе предоставленной базы знаний.

ВАЖНЫЕ ИНСТРУКЦИИ:
1. ВСЕГДА сначала размышляй шаг за шагом, анализируя предоставленный контекст
2. Отвечай ТОЛЬКО на основе предоставленной информации
3. Если информации для ответа недостаточно - честно говори "Я не знаю"
4. Начинай ответ с размышления, затем давай окончательный ответ
5. Используй формат:
РАЗМЫШЛЕНИЕ:
1. Шаг первый...
2. Шаг второй...
3. Шаг третий...

ОТВЕТ: [окончательный ответ]

ПРИМЕРЫ РАЗМЫШЛЕНИЙ И ОТВЕТОВ:
{few_shot_text}

КОНТЕКСТ ДЛЯ АНАЛИЗА:
{context}

ТЕКУЩИЙ ВОПРОС: {query}

Теперь проанализируй вопрос и предоставь ответ:"""

        return prompt

    def generate_response(self, prompt: str) -> str:
        """Генерация ответа с помощью YandexGPT"""
        return self.llm_client.generate_response(prompt)

    def process_query(self, query: str) -> Dict:
        """Основной метод обработки запроса"""
        print(f"\n🔍 Поиск информации для запроса: '{query}'")

        # Поиск релевантных чанков
        context_chunks = self.search_similar_chunks(query, k=3)

        # Проверка релевантности
        if not context_chunks or all(chunk['score'] < 0.3 for chunk in context_chunks):
            return {
                "answer": "РАЗМЫШЛЕНИЕ:\n1. Проанализировал запрос и поискал релевантную информацию в базе знаний\n2. Не найдено достаточно релевантных документов для ответа на вопрос\n3. Информация в базе знаний не покрывает данный запрос\n\nОТВЕТ: Я не знаю ответ на этот вопрос. В моей базе знаний нет релевантной информации.",
                "sources": [],
                "reasoning": "Не найдено достаточно релевантных документов в базе знаний."
            }

        print(f"📚 Найдено релевантных фрагментов: {len(context_chunks)}")
        for i, chunk in enumerate(context_chunks):
            print(f"  {i+1}. {chunk['metadata']['source']} (сходство: {chunk['score']:.3f})")

        # Построение промпта
        prompt = self.build_prompt(query, context_chunks)

        # Генерация ответа
        print("🤖 Генерация ответа с помощью YandexGPT...")
        answer = self.generate_response(prompt)

        return {
            "answer": answer,
            "sources": [chunk['metadata']['source'] for chunk in context_chunks],
            "reasoning": "Ответ сгенерирован с использованием Chain-of-Thought подхода."
        }

def main():
    # Инициализация бота
    print("🤖 Инициализация RAG-бота с YandexGPT...")

    # Проверка наличия необходимых переменных окружения
    if not os.getenv('YANDEX_API_KEY') or not os.getenv('YANDEX_FOLDER_ID'):
        print("❌ Ошибка: Не найдены переменные окружения YANDEX_API_KEY и YANDEX_FOLDER_ID")
        print("Добавьте их в файл .env:")
        print("YANDEX_API_KEY=your_yandex_api_key")
        print("YANDEX_FOLDER_ID=your_folder_id")
        return

    try:
        bot = RAGBot("vector_index")
        print("✅ Бот успешно инициализирован!")
    except Exception as e:
        print(f"❌ Ошибка при инициализации бота: {e}")
        return

    print("=" * 60)
    print("Добро пожаловать в RAG-бот по магическим существам!")
    print("Используется YandexGPT для генерации ответов")
    print("Задавайте вопросы о магических существах, персонажах и организациях")
    print("Для выхода введите 'выход' или 'exit'")
    print("=" * 60)

    # Примеры тестовых запросов
    test_queries = [
        "Какое существо ищет блестящие предметы?",
        "Кто изучает магических существ?",
        "Что такое Темносила?",
        "Какие способности у Громоперого?",
        "Где находится Вершинная Академия?",
        "Что делает Невидимкошерст когда чувствует опасность?"
    ]

    print("\n📋 Примеры вопросов для тестирования:")
    for i, query in enumerate(test_queries, 1):
        print(f"{i}. {query}")

    # REPL-цикл
    while True:
        try:
            user_input = input("\n🎯 Ваш вопрос: ").strip()

            if user_input.lower() in ['выход', 'exit', 'quit']:
                print("До свидания!")
                break

            if not user_input:
                continue

            # Обработка запроса
            result = bot.process_query(user_input)

            # Вывод результата
            print(f"\n{'='*50}")
            print("🤖 ОТВЕТ БОТА:")
            print(result["answer"])
            print(f"{'='*50}")

            if result["sources"]:
                print(f"\n📚 Использованные источники:")
                for source in result["sources"]:
                    print(f"  - {source}")

        except KeyboardInterrupt:
            print("\n\nДо свидания!")
            break
        except Exception as e:
            print(f"❌ Произошла ошибка: {e}")

if __name__ == "__main__":
    main()