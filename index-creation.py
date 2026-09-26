import os
import logging
from faiss_storage import FaissVectorStorage

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s %(levelname)s %(message)s",
    force=True,
)
log = logging.getLogger("index_creation")
log.setLevel(logging.INFO)
logging.getLogger("faiss_storage").setLevel(logging.INFO)

faiss_dir = os.environ["FAISS_DATA_DIR"]
collection = os.environ["COLLECTION"]
store = FaissVectorStorage(faiss_dir, device="gpu", nlist=4096)


count = store.build_index(collection)
print(collection, count)
