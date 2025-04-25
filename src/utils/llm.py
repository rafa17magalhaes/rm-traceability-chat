import os, logging, os, multiprocessing
from threading import Lock
from typing import Optional
from gpt4all import GPT4All

_MODEL_FILE = os.getenv("MODEL_FILE", "/app/model/model.gguf")

_logger = logging.getLogger("llm")
_logger.setLevel(logging.INFO)
if not _logger.handlers:
    _logger.addHandler(logging.StreamHandler())

_lock   = Lock()
_model: Optional[GPT4All] = None

def get_model() -> GPT4All:
    global _model
    with _lock:
        if _model is None:
            _logger.info("Carregando LLM principal %s …", _MODEL_FILE)
            _model = GPT4All(_MODEL_FILE,
                             n_threads=multiprocessing.cpu_count() or 4)
        return _model
