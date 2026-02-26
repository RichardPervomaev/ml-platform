import threading
from collections import deque

MAX_WINDOW_SIZE = 1000

class ProductionDataCollector:

    def __init__(self):
        self.buffer = deque(maxlen=MAX_WINDOW_SIZE)
        self.lock = threading.Lock()

    def add(self, features):
        with self.lock:
            self.buffer.append(features)

    def get_all(self):
        with self.lock:
            return list(self.buffer)

    def size(self):
        with self.lock:
            return len(self.buffer)


collector = ProductionDataCollector()
