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
log = logging.getLogger("embedding_job_2")
log.setLevel(logging.INFO)
logging.getLogger("faiss_storage").setLevel(logging.INFO)

# set source 2 FAISS storage directory
storage_dir_2 = Path(os.environ["FAISS_DATA_DIR"]).expanduser()
if not storage_dir_2.is_absolute():
    raise ValueError("FAISS_DATA_DIR must be an absolute path")
store_2 = FaissVectorStorage(storage_dir_2, device="gpu")

log.info("Checking existing source 2 embeddings; FAISS directory: %s", storage_dir_2)
model_2 = None

src_2_chunks = pd.read_csv(
    Path(__file__).resolve().parent / "dataset/train/train_source2.tsv",
    sep="\t",
    usecols=["entity_id", "business_name", "business_address"],
    dtype={"entity_id": str, "business_name": str, "business_address": str},
    chunksize=10000,
)

# embeddings of source 2
processed_rows_2 = 0
chunk_index_2 = 0
skipped_chunks_2 = 0
for chunk_index_2, src_2_df in enumerate(src_2_chunks, start=1):
    src_2_df[["business_name", "business_address"]] = src_2_df[
        ["business_name", "business_address"]
    ].fillna("")
    entity_ids_2 = src_2_df["entity_id"].to_numpy()
    check_started = time.perf_counter()
    missing_names_2 = store_2.missing_ids("business_name_2", entity_ids_2)
    log.info(
        "Source 2 chunk %d business_name_2 missing_ids: %d/%d missing in %.3fs",
        chunk_index_2,
        len(missing_names_2),
        len(entity_ids_2),
        time.perf_counter() - check_started,
    )
    check_started = time.perf_counter()
    missing_addresses_2 = store_2.missing_ids("business_address_2", entity_ids_2)
    log.info(
        "Source 2 chunk %d business_address_2 missing_ids: %d/%d missing in %.3fs",
        chunk_index_2,
        len(missing_addresses_2),
        len(entity_ids_2),
        time.perf_counter() - check_started,
    )

    if not missing_names_2 and not missing_addresses_2:
        skipped_chunks_2 += 1
        processed_rows_2 += len(src_2_df)
        log.info("Source 2 chunk %d already saved in both collections; skipped", chunk_index_2)
        continue

    log.info(
        "Source 2 chunk %d started (%d rows; %d names and %d addresses remaining)",
        chunk_index_2,
        len(src_2_df),
        len(missing_names_2),
        len(missing_addresses_2),
    )
    if model_2 is None:
        log.info("Loading multilingual-e5-large on CUDA for source 2")
        model_2 = SentenceTransformer(
            "intfloat/multilingual-e5-large",
            device="cuda",
            model_kwargs={"torch_dtype": torch.float16},
        )
        log.info("Source 2 model ready")

    if missing_names_2:
        name_rows_2 = src_2_df[src_2_df["entity_id"].isin(missing_names_2)]
        business_names_2 = ("query: " + name_rows_2["business_name"]).to_numpy()
        encode_started = time.perf_counter()
        embeddings_bn_2 = model_2.encode(
            business_names_2,
            batch_size=128,
            normalize_embeddings=True,
            show_progress_bar=False,
        ).astype("float16")
        log.info(
            "Source 2 chunk %d business_name_2 encode+fp16: %d rows in %.3fs",
            chunk_index_2,
            len(name_rows_2),
            time.perf_counter() - encode_started,
        )
        store_started = time.perf_counter()
        store_2.store_embeddings(
            "business_name_2",
            ids=name_rows_2["entity_id"].to_numpy(),
            embeddings=embeddings_bn_2,
        )
        log.info(
            "Source 2 chunk %d business_name_2 store_embeddings: %d rows in %.3fs",
            chunk_index_2,
            len(name_rows_2),
            time.perf_counter() - store_started,
        )
        del embeddings_bn_2

    if missing_addresses_2:
        address_rows_2 = src_2_df[src_2_df["entity_id"].isin(missing_addresses_2)]
        business_address_2 = ("query: " + address_rows_2["business_address"]).to_numpy()
        encode_started = time.perf_counter()
        embeddings_ba_2 = model_2.encode(
            business_address_2,
            batch_size=128,
            normalize_embeddings=True,
            show_progress_bar=False,
        ).astype("float16")
        log.info(
            "Source 2 chunk %d business_address_2 encode+fp16: %d rows in %.3fs",
            chunk_index_2,
            len(address_rows_2),
            time.perf_counter() - encode_started,
        )
        store_started = time.perf_counter()
        store_2.store_embeddings(
            "business_address_2",
            ids=address_rows_2["entity_id"].to_numpy(),
            embeddings=embeddings_ba_2,
        )
        log.info(
            "Source 2 chunk %d business_address_2 store_embeddings: %d rows in %.3fs",
            chunk_index_2,
            len(address_rows_2),
            time.perf_counter() - store_started,
        )
        del embeddings_ba_2

    processed_rows_2 += len(src_2_df)
    log.info("Source 2 chunk %d complete (%d rows scanned)", chunk_index_2, processed_rows_2)

log.info(
    "Source 2 finished %d chunks (%d rows scanned; %d chunks skipped)",
    chunk_index_2,
    processed_rows_2,
    skipped_chunks_2,
)
