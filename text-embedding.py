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
log = logging.getLogger("embedding_job_3")
log.setLevel(logging.INFO)
logging.getLogger("faiss_storage").setLevel(logging.INFO)

# set source 3 FAISS storage directory
storage_dir_3 = Path(os.environ["FAISS_DATA_DIR"]).expanduser()
if not storage_dir_3.is_absolute():
    raise ValueError("FAISS_DATA_DIR must be an absolute path")
store_3 = FaissVectorStorage(storage_dir_3, device="gpu")

log.info("Checking existing source 3 embeddings; FAISS directory: %s", storage_dir_3)
model_3 = None

src_3_chunks = pd.read_csv(
    Path(__file__).resolve().parent / "dataset/train/train_source3.tsv",
    sep="\t",
    usecols=["entity_id", "business_name", "business_address"],
    dtype={"entity_id": str, "business_name": str, "business_address": str},
    chunksize=10000,
)

# embeddings of source 3
processed_rows_3 = 0
chunk_index_3 = 0
skipped_chunks_3 = 0
for chunk_index_3, src_3_df in enumerate(src_3_chunks, start=1):
    src_3_df[["business_name", "business_address"]] = src_3_df[
        ["business_name", "business_address"]
    ].fillna("")
    entity_ids_3 = src_3_df["entity_id"].to_numpy()
    check_started = time.perf_counter()
    missing_names_3 = store_3.missing_ids("business_name_3", entity_ids_3)
    log.info(
        "Source 3 chunk %d business_name_3 missing_ids: %d/%d missing in %.3fs",
        chunk_index_3,
        len(missing_names_3),
        len(entity_ids_3),
        time.perf_counter() - check_started,
    )
    check_started = time.perf_counter()
    missing_addresses_3 = store_3.missing_ids("business_address_3", entity_ids_3)
    log.info(
        "Source 3 chunk %d business_address_3 missing_ids: %d/%d missing in %.3fs",
        chunk_index_3,
        len(missing_addresses_3),
        len(entity_ids_3),
        time.perf_counter() - check_started,
    )

    if not missing_names_3 and not missing_addresses_3:
        skipped_chunks_3 += 1
        processed_rows_3 += len(src_3_df)
        log.info("Source 3 chunk %d already saved in both collections; skipped", chunk_index_3)
        continue

    log.info(
        "Source 3 chunk %d started (%d rows; %d names and %d addresses remaining)",
        chunk_index_3,
        len(src_3_df),
        len(missing_names_3),
        len(missing_addresses_3),
    )
    if model_3 is None:
        log.info("Loading multilingual-e5-large on CUDA for source 3")
        model_3 = SentenceTransformer(
            "intfloat/multilingual-e5-large",
            device="cuda",
            model_kwargs={"torch_dtype": torch.float16},
        )
        log.info("Source 3 model ready")

    if missing_names_3:
        name_rows_3 = src_3_df[src_3_df["entity_id"].isin(missing_names_3)]
        business_names_3 = ("query: " + name_rows_3["business_name"]).to_numpy()
        encode_started = time.perf_counter()
        embeddings_bn_3 = model_3.encode(
            business_names_3,
            batch_size=128,
            normalize_embeddings=True,
            show_progress_bar=False,
        ).astype("float16")
        log.info(
            "Source 3 chunk %d business_name_3 encode+fp16: %d rows in %.3fs",
            chunk_index_3,
            len(name_rows_3),
            time.perf_counter() - encode_started,
        )
        store_started = time.perf_counter()
        store_3.store_embeddings(
            "business_name_3",
            ids=name_rows_3["entity_id"].to_numpy(),
            embeddings=embeddings_bn_3,
        )
        log.info(
            "Source 3 chunk %d business_name_3 store_embeddings: %d rows in %.3fs",
            chunk_index_3,
            len(name_rows_3),
            time.perf_counter() - store_started,
        )
        del embeddings_bn_3

    if missing_addresses_3:
        address_rows_3 = src_3_df[src_3_df["entity_id"].isin(missing_addresses_3)]
        business_address_3 = ("query: " + address_rows_3["business_address"]).to_numpy()
        encode_started = time.perf_counter()
        embeddings_ba_3 = model_3.encode(
            business_address_3,
            batch_size=128,
            normalize_embeddings=True,
            show_progress_bar=False,
        ).astype("float16")
        log.info(
            "Source 3 chunk %d business_address_3 encode+fp16: %d rows in %.3fs",
            chunk_index_3,
            len(address_rows_3),
            time.perf_counter() - encode_started,
        )
        store_started = time.perf_counter()
        store_3.store_embeddings(
            "business_address_3",
            ids=address_rows_3["entity_id"].to_numpy(),
            embeddings=embeddings_ba_3,
        )
        log.info(
            "Source 3 chunk %d business_address_3 store_embeddings: %d rows in %.3fs",
            chunk_index_3,
            len(address_rows_3),
            time.perf_counter() - store_started,
        )
        del embeddings_ba_3

    processed_rows_3 += len(src_3_df)
    log.info("Source 3 chunk %d complete (%d rows scanned)", chunk_index_3, processed_rows_3)

log.info(
    "Source 3 finished %d chunks (%d rows scanned; %d chunks skipped)",
    chunk_index_3,
    processed_rows_3,
    skipped_chunks_3,
)
