import logging
import os
from pathlib import Path

import pandas as pd
from sentence_transformers import SentenceTransformer
from vector_storage import VectorStorage
import torch

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s %(levelname)s %(message)s",
    force=True,
)
log = logging.getLogger("embedding_job")
log.setLevel(logging.INFO)
logging.getLogger("vector_storage").setLevel(logging.INFO)

# set vector storage directory
storage_dir = Path(os.environ["CHROMA_DATA_DIR"]).expanduser()
if not storage_dir.is_absolute():
    raise ValueError("CHROMA_DATA_DIR must be an absolute path")
store = VectorStorage(storage_dir)

log.info("Checking existing embeddings; Chroma directory: %s", storage_dir)
model = None

src_2_chunks = pd.read_csv(
    Path(__file__).resolve().parent / "dataset/train/train_source2.tsv",
    sep="\t",
    usecols=["entity_id", "business_name", "business_address"],
    dtype={"entity_id": str, "business_name": str, "business_address": str},
    chunksize=1024,
)

# embeddings of source 2
processed_rows = 0
chunk_index = 0
skipped_chunks = 0
for chunk_index, src_2_df in enumerate(src_2_chunks, start=1):
    src_2_df[["business_name", "business_address"]] = src_2_df[
        ["business_name", "business_address"]
    ].fillna("")
    entity_ids_2 = src_2_df["entity_id"].to_numpy()
    missing_names = store.missing_ids("business_name_2", entity_ids_2)
    missing_addresses = store.missing_ids("business_address_2", entity_ids_2)

    if not missing_names and not missing_addresses:
        skipped_chunks += 1
        processed_rows += len(src_2_df)
        log.info("Chunk %d already saved in both collections; skipped", chunk_index)
        continue

    log.info(
        "Chunk %d started (%d rows; %d names and %d addresses remaining)",
        chunk_index,
        len(src_2_df),
        len(missing_names),
        len(missing_addresses),
    )
    if model is None:
        log.info("Loading multilingual-e5-large on CUDA")
        model = SentenceTransformer(
            "intfloat/multilingual-e5-large",
            device="cuda",
            model_kwargs={"torch_dtype": torch.float16},
        )
        log.info("Model ready")

    if missing_names:
        name_rows = src_2_df[src_2_df["entity_id"].isin(missing_names)]
        business_names_2 = ("query: " + name_rows["business_name"]).to_numpy()
        embeddings_bn_2 = model.encode(
            business_names_2,
            batch_size=128,
            normalize_embeddings=True,
            show_progress_bar=False,
        ).astype("float16")
        store.store_embeddings(
            "business_name_2",
            ids=name_rows["entity_id"].to_numpy(),
            embeddings=embeddings_bn_2,
        )
        del embeddings_bn_2

    if missing_addresses:
        address_rows = src_2_df[src_2_df["entity_id"].isin(missing_addresses)]
        business_address_2 = ("query: " + address_rows["business_address"]).to_numpy()
        embeddings_ba_2 = model.encode(
            business_address_2,
            batch_size=128,
            normalize_embeddings=True,
            show_progress_bar=False,
        ).astype("float16")
        store.store_embeddings(
            "business_address_2",
            ids=address_rows["entity_id"].to_numpy(),
            embeddings=embeddings_ba_2,
        )
        del embeddings_ba_2

    processed_rows += len(src_2_df)
    log.info("Chunk %d complete (%d rows scanned)", chunk_index, processed_rows)

log.info(
    "Finished %d chunks (%d rows scanned; %d chunks skipped)",
    chunk_index,
    processed_rows,
    skipped_chunks,
)
