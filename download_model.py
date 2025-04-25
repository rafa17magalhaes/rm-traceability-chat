import os, sys, glob, shutil, logging
from dotenv import load_dotenv
from huggingface_hub import snapshot_download

load_dotenv()
REPO   = os.getenv("MODEL_REPO")
TOKEN  = os.getenv("HF_TOKEN") or None

if not REPO:
    print("❌  Defina MODEL_REPO no .env", file=sys.stderr)
    sys.exit(1)

logging.basicConfig(level=logging.INFO)
logging.info("🔽 Baixando %s …", REPO)

snapshot_download(
    repo_id   = REPO,
    local_dir = "model",
    token     = TOKEN,
    local_dir_use_symlinks=False,
)

#
#  Escolhe *Q5_0.gguf se existir; senão cai para o maior arquivo
#
candidates = glob.glob("model/**/*.Q5_0.gguf", recursive=True)
if not candidates:
    candidates = glob.glob("model/**/*.gguf", recursive=True)

if not candidates:
    print("❌  Nenhum .gguf encontrado em model/", file=sys.stderr)
    sys.exit(1)

best = max(candidates, key=os.path.getsize)
dst  = "model/model.gguf"

shutil.move(best, dst)
logging.info("✅  Copiado %s → %s", best, dst)
