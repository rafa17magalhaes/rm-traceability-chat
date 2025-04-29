import os
import requests
import logging
from typing import Optional
from joblib import load

logger = logging.getLogger("MLService")
logger.setLevel(logging.DEBUG)
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    ))
    logger.addHandler(handler)

class MLService:
    _action_model = None

    @staticmethod
    def get_company_id_for_user(user_id: str) -> Optional[str]:
        backend_url = os.getenv("BACKEND_URL", "http://rm_traceability_app:3001")
        url = f"{backend_url}/orchestration/full-data/{user_id}"
        logger.debug(f"[get_company_id_for_user] GET {url}")
        try:
            resp = requests.get(url, timeout=5)
            resp.raise_for_status()
            data = resp.json()
            return data.get("user", {}).get("companyId")
        except Exception:
            logger.error("[get_company_id_for_user] falha", exc_info=True)
            return None

    @staticmethod
    def _load_action_model():
        if MLService._action_model is None:
            model_path = os.path.join(
                os.path.dirname(__file__), "..", "ml", "model.pkl"
            )
            MLService._action_model = load(model_path)
            logger.info(f"Action model carregado de {model_path}")
        return MLService._action_model

    @staticmethod
    def predict_next_action(n_codes: int, n_events: int) -> str:
        """
        Prediz a próxima ação com base nos contadores de códigos e eventos.
        """
        model = MLService._load_action_model()
        pred = model.predict([[n_codes, n_events]])[0]
        logger.debug(f"[predict_next_action] features=({n_codes},{n_events}) -> {pred}")
        return pred

    @staticmethod
    def get_inventory_quantity(resource_name: str, company_id: str) -> int:
        backend = os.getenv("BACKEND_URL", "http://rm_traceability_app:3001")
        url = (
            f"{backend}/orchestration/inventory-quantity"
            f"?companyId={company_id}&resourceName={resource_name}"
        )
        logger.debug(f"[get_inventory_quantity] GET {url}")
        try:
            resp = requests.get(url, timeout=5)
            resp.raise_for_status()
            return resp.json().get("amount", 0)
        except Exception:
            logger.error("[get_inventory_quantity] falha", exc_info=True)
            return 0

    @staticmethod
    def fetch_inventory_for_product(resource_name: str, company_id: str) -> str:
        qtd = MLService.get_inventory_quantity(resource_name, company_id)
        return f"\n[Resposta do BD: Você tem {qtd} unidades de {resource_name} no estoque.]"

    @staticmethod
    def fetch_codes_for_product(resource_name: str, company_id: str) -> list[str]:
        backend = os.getenv("BACKEND_URL", "http://rm_traceability_app:3001")
        url = (
            f"{backend}/orchestration/inventory-codes"
            f"?companyId={company_id}&resourceName={resource_name}"
        )
        logger.debug(f"[fetch_codes_for_product] GET {url}")
        try:
            resp = requests.get(url, timeout=5)
            resp.raise_for_status()
            return resp.json().get("codes", [])
        except Exception:
            logger.error("[fetch_codes_for_product] falha", exc_info=True)
            return []
