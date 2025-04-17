import os
import requests
import logging
from typing import Optional

logger = logging.getLogger("MLService")
logger.setLevel(logging.DEBUG)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)

class MLService:
    @staticmethod
    def get_company_id_for_user(user_id: str) -> Optional[str]:
        backend_url = os.getenv("BACKEND_URL", "http://rm_traceability_app:3001")
        url = f"{backend_url}/orchestration/full-data/{user_id}"
        logger.debug(f"[get_company_id_for_user] Chamando endpoint: {url}")
        try:
            resp = requests.get(url, timeout=5)
            resp.raise_for_status()
            data = resp.json()
            user_obj = data.get("user")
            if user_obj and user_obj.get("companyId"):
                return user_obj["companyId"]
            logger.error(f"[get_company_id_for_user] user.companyId não encontrado. user_obj={user_obj}")
        except Exception:
            logger.error("[get_company_id_for_user] Erro ao buscar companyId", exc_info=True)
        return None

    @staticmethod
    def get_inventory_quantity(resource_name: str, company_id: str) -> int:
        backend = os.getenv("BACKEND_URL", "http://rm_traceability_app:3001")
        url = (
            f"{backend}/orchestration/inventory-quantity"
            f"?companyId={company_id}&resourceName={resource_name}"
        )
        logger.debug(f"[get_inventory_quantity] Chamando endpoint: {url}")
        try:
            resp = requests.get(url, timeout=5)
            resp.raise_for_status()
            data = resp.json()
            return data.get("amount", 0)
        except Exception:
            logger.error("[get_inventory_quantity] Erro ao consultar inventory-quantity", exc_info=True)
            return 0

    @staticmethod
    def fetch_inventory_for_product(resource_name: str, company_id: str) -> str:
        qtd = MLService.get_inventory_quantity(resource_name, company_id)
        return f"\n[Resposta do BD: Você tem {qtd} unidades de {resource_name} no estoque.]"

    @staticmethod
    def predict_next_action(user_id: str) -> str:
        logger.debug(f"[predict_next_action] Chamando modelo de ML para user_id: {user_id}")
        return "inventario"

    @staticmethod
    def fetch_codes_for_product(resource_name: str, company_id: str) -> list[str]:
        """Retorna a lista de códigos (value) para um resource no inventário."""
        backend = os.getenv("BACKEND_URL", "http://rm_traceability_app:3001")
        url = (
            f"{backend}/orchestration/inventory-codes"
            f"?companyId={company_id}&resourceName={resource_name}"
        )
        logger.debug(f"[fetch_codes_for_product] Chamando endpoint: {url}")
        try:
            resp = requests.get(url, timeout=5)
            resp.raise_for_status()
            return resp.json().get("codes", [])
        except Exception:
            logger.error("[fetch_codes_for_product] Erro ao buscar códigos", exc_info=True)
            return []
