import logging
import os
import time
from pathlib import Path

import pandas as pd
from sentence_transformers import SentenceTransformer
from faiss_storage import FaissVectorStorage
import torch

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s %(levelname)s %(message)s",
    force=True,
)
log = logging.getLogger("embedding_job_1")
log.setLevel(logging.INFO)
logging.getLogger("faiss_storage").setLevel(logging.INFO)

# set source 1 FAISS storage directory
storage_dir_1 = Path(os.environ["FAISS_DATA_DIR"]).expanduser()
if not storage_dir_1.is_absolute():
    raise ValueError("FAISS_DATA_DIR must be an absolute path")
store_1 = FaissVectorStorage(storage_dir_1, device="gpu")

log.info("Checking existing source 1 embeddings; FAISS directory: %s", storage_dir_1)
model_1 = None

src_1_chunks = pd.read_csv(
    Path(__file__).resolve().parent / "dataset/train/train_source1.tsv",
    sep="\t",
    usecols=["entity_id", "business_name", "business_address"],
    dtype={"entity_id": str, "business_name": str, "business_address": str},
    chunksize=10000,
)

# embeddings of source 1
processed_rows_1 = 0
chunk_index_1 = 0
skipped_chunks_1 = 0
for chunk_index_1, src_1_df in enumerate(src_1_chunks, start=1):
    src_1_df[["business_name", "business_address"]] = src_1_df[
        ["business_name", "business_address"]
    ].fillna("")
    entity_ids_1 = src_1_df["entity_id"].to_numpy()
    check_started = time.perf_counter()
    missing_names_1 = store_1.missing_ids("business_name_1", entity_ids_1)
    log.info(
        "Source 1 chunk %d business_name_1 missing_ids: %d/%d missing in %.3fs",
        chunk_index_1,
        len(missing_names_1),
        len(entity_ids_1),
        time.perf_counter() - check_started,
    )
    check_started = time.perf_counter()
    missing_addresses_1 = store_1.missing_ids("business_address_1", entity_ids_1)
    log.info(
        "Source 1 chunk %d business_address_1 missing_ids: %d/%d missing in %.3fs",
        chunk_index_1,
        len(missing_addresses_1),
        len(entity_ids_1),
        time.perf_counter() - check_started,
    )

    if not missing_names_1 and not missing_addresses_1:
        skipped_chunks_1 += 1
        processed_rows_1 += len(src_1_df)
        log.info("Source 1 chunk %d already saved in both collections; skipped", chunk_index_1)
        continue

    log.info(
        "Source 1 chunk %d started (%d rows; %d names and %d addresses remaining)",
        chunk_index_1,
        len(src_1_df),
        len(missing_names_1),
        len(missing_addresses_1),
    )
    if model_1 is None:
        log.info("Loading multilingual-e5-large on CUDA for source 1")
        model_1 = SentenceTransformer(
            "intfloat/multilingual-e5-large",
            device="cuda",
            model_kwargs={"torch_dtype": torch.float16},
        )
        log.info("Source 1 model ready")

    if missing_names_1:
        name_rows_1 = src_1_df[src_1_df["entity_id"].isin(missing_names_1)]
        business_names_1 = ("query: " + name_rows_1["business_name"]).to_numpy()
        encode_started = time.perf_counter()
        embeddings_bn_1 = model_1.encode(
            business_names_1,
            batch_size=128,
            normalize_embeddings=True,
            show_progress_bar=False,
        ).astype("float16")
        log.info(
            "Source 1 chunk %d business_name_1 encode+fp16: %d rows in %.3fs",
            chunk_index_1,
            len(name_rows_1),
            time.perf_counter() - encode_started,
        )
        store_started = time.perf_counter()
        store_1.store_embeddings(
            "business_name_1",
            ids=name_rows_1["entity_id"].to_numpy(),
            embeddings=embeddings_bn_1,
        )
        log.info(
            "Source 1 chunk %d business_name_1 store_embeddings: %d rows in %.3fs",
            chunk_index_1,
            len(name_rows_1),
            time.perf_counter() - store_started,
        )
        del embeddings_bn_1

    if missing_addresses_1:
        address_rows_1 = src_1_df[src_1_df["entity_id"].isin(missing_addresses_1)]
        business_address_1 = ("query: " + address_rows_1["business_address"]).to_numpy()
        encode_started = time.perf_counter()
        embeddings_ba_1 = model_1.encode(
            business_address_1,
            batch_size=128,
            normalize_embeddings=True,
            show_progress_bar=False,
        ).astype("float16")
        log.info(
            "Source 1 chunk %d business_address_1 encode+fp16: %d rows in %.3fs",
            chunk_index_1,
            len(address_rows_1),
            time.perf_counter() - encode_started,
        )
        store_started = time.perf_counter()
        store_1.store_embeddings(
            "business_address_1",
            ids=address_rows_1["entity_id"].to_numpy(),
            embeddings=embeddings_ba_1,
        )
        log.info(
            "Source 1 chunk %d business_address_1 store_embeddings: %d rows in %.3fs",
            chunk_index_1,
            len(address_rows_1),
            time.perf_counter() - store_started,
        )
        del embeddings_ba_1

    processed_rows_1 += len(src_1_df)
    log.info("Source 1 chunk %d complete (%d rows scanned)", chunk_index_1, processed_rows_1)

log.info(
    "Source 1 finished %d chunks (%d rows scanned; %d chunks skipped)",
    chunk_index_1,
    processed_rows_1,
    skipped_chunks_1,
)
