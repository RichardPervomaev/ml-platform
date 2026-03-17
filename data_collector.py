import os
import json
import redis

MAX_WINDOW_SIZE = int(os.getenv("MAX_WINDOW_SIZE", "1000"))
REDIS_HOST = os.getenv("REDIS_HOST", "redis")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
REDIS_DB = int(os.getenv("REDIS_DB", "0"))
REDIS_KEY = os.getenv("REDIS_COLLECTOR_KEY", "production_data")


class ProductionDataCollector:
    """
    Redis-backed collector production-данных.

    Что делает:
    - add(features): добавляет одну запись
    - get_all(): читает весь window
    - size(): размер window
    - clear(): очищает буфер

    Почему Redis:
    - API и drift-worker живут в разных контейнерах
    - нужен общий state
    - Redis даёт быстрый shared buffer
    """

    def __init__(self):
        self.client = redis.Redis(
            host=REDIS_HOST,
            port=REDIS_PORT,
            db=REDIS_DB,
            decode_responses=True
        )

    def add(self, features):
        """
        Добавляем запись вида [x1, x2, x3] в Redis list.
        Храним как JSON-строку.
        """
        if not isinstance(features, list) or len(features) != 3:
            return

        payload = json.dumps(features)

        # кладём в конец списка
        self.client.rpush(REDIS_KEY, payload)

        # обрезаем список до последних MAX_WINDOW_SIZE записей
        current_size = self.client.llen(REDIS_KEY)
        if current_size > MAX_WINDOW_SIZE:
            self.client.ltrim(REDIS_KEY, -MAX_WINDOW_SIZE, -1)

    def get_all(self):
        """
        Читаем весь текущий window и возвращаем list of lists.
        """
        raw_items = self.client.lrange(REDIS_KEY, 0, -1)
        return [json.loads(item) for item in raw_items]

    def size(self):
        """
        Возвращаем размер текущего окна.
        """
        return self.client.llen(REDIS_KEY)

    def clear(self):
        """
        Полностью очищаем буфер.
        Удобно для тестов.
        """
        self.client.delete(REDIS_KEY)


collector = ProductionDataCollector()
