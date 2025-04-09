import os
import requests
import logging
from typing import Any, Dict, Optional

logger = logging.getLogger("MLService")
logger.setLevel(logging.DEBUG)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
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
            else:
                logger.error(f"[get_company_id_for_user] user.companyId não encontrado. user_obj={user_obj}")
                return None
        except Exception as e:
            logger.error("[get_company_id_for_user] Erro ao buscar companyId:", exc_info=True)
            return None

    @staticmethod
    def get_inventory_quantity(resource_name: str, company_id: str) -> int:
        backend_url = os.getenv("BACKEND_URL", "http://rm_traceability_app:3001")
        url = f"{backend_url}/orchestration/inventory-quantity?companyId={company_id}&resourceName={resource_name}"
        logger.debug(f"[get_inventory_quantity] Chamando endpoint: {url}")

        try:
            resp = requests.get(url, timeout=5)
            resp.raise_for_status()
            data = resp.json()
            logger.info(f"[get_inventory_quantity] Response from {url}: {data}")
            return data.get("quantidade", 0)
        except Exception as e:
            logger.error("[get_inventory_quantity] Erro ao consultar inventory-quantity:", exc_info=True)
            return 0

    @staticmethod
    def fetch_inventory_for_product(resource_name: str, company_id: str) -> str:
        qtd = MLService.get_inventory_quantity(resource_name, company_id)
        return f"\n[Resposta do BD: Você tem {qtd} unidades de {resource_name} no estoque.]"

    @staticmethod
    def predict_next_action(user_id: str) -> str:
        logger.debug(f"[predict_next_action] Chamando modelo de ML para user_id: {user_id}")
        return "inventario"
