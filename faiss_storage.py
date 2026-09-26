"""Persistent FP16 vectors with a FAISS index for approximate cosine search.

The caller supplies unit-normalized, two-dimensional ``numpy.float16`` arrays.
Vectors are saved unchanged in a little-endian FP16 file, while SQLite maps
string IDs to rows in that file. ``build_index`` creates or updates the separate
FP16 FAISS search index after writes. One process should write a collection at a
time; readers should not access a collection while it is being written.

Example::

    store = FaissVectorStorage("/absolute/path/to/faiss_data", device="auto")
    store.store_embeddings("business_name_1", ["S1-123"], vectors_fp16)
    store.build_index("business_name_1")
    matches = store.search_similar("business_name_1", query_fp16, k=10)
    saved = store.get_embeddings("business_name_1", ["S1-123"])

FAISS is imported only for indexing and search. FP16 writes and ID lookups do
not require it. CPU index operations use temporary float32 arrays; GPU index
building transfers FP16 tensors and converts them to float32 on CUDA. Neither
path changes the persisted FP16 vectors.
"""

import importlib
import logging
import os
import re
import sqlite3
import time
import uuid
from contextlib import closing, nullcontext
from pathlib import Path

import numpy as np


log = logging.getLogger(__name__)

_NAME_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]*\Z")
_SQL_BATCH_SIZE = 900  # Works with SQLite builds limited to 999 parameters.
_FP16_DISK_DTYPE = np.dtype("<f2")


class FaissVectorStorage:
    """Store FP16 vectors and expose ID lookup and approximate top-k search.

    ``device='auto'`` selects GPU when the installed FAISS build exposes one,
    otherwise CPU. ``device='gpu'`` fails rather than silently using CPU.
    The GPU number refers to the devices visible to the process, so a single
    ``CUDA_VISIBLE_DEVICES`` MIG slice is normally device 0.
    """

    def __init__(
        self,
        directory,
        *,
        device="auto",
        gpu_id=0,
        nlist=4096,
        nprobe=32,
        training_size=200_000,
        add_batch_size=8192,
    ):
        if device not in {"auto", "cpu", "gpu"}:
            raise ValueError("device must be 'auto', 'cpu', or 'gpu'")
        if gpu_id < 0 or nlist < 1 or nprobe < 1 or training_size < 1 or add_batch_size < 1:
            raise ValueError("gpu_id must be nonnegative and index settings must be positive")

        self.directory = Path(directory).expanduser()
        self.directory.mkdir(parents=True, exist_ok=True)
        self.device = device
        self.gpu_id = gpu_id
        self.nlist = nlist
        self.nprobe = nprobe
        self.training_size = training_size
        self.add_batch_size = add_batch_size
        # Retain GPU resources alongside each loaded index.
        self._indexes = {}
        log.info("FAISS storage ready at %s (device=%s)", self.directory, device)

    def missing_ids(self, name, ids):
        """Return IDs absent from the collection, preserving input order."""
        ids_list = self._check_ids(ids)
        if not ids_list:
            return []
        collection = self._collection(name)
        db_path = collection / "ids.sqlite3"
        if not db_path.exists():
            return ids_list
        with closing(sqlite3.connect(db_path)) as connection:
            existing = self._rows_for_ids(connection, ids_list)
        return [item_id for item_id in ids_list if item_id not in existing]

    def store_embeddings(self, name, ids, embeddings):
        """Append new FP16 vectors; return the number added.

        Existing IDs with identical FP16 vectors are skipped. An existing ID
        with different vector values is rejected; use another collection when
        changing the embedding model or its inputs.
        """
        ids_list = self._check_ids(ids)
        vectors = self._check_vectors(embeddings, ndim=2)
        if len(ids_list) != len(vectors):
            raise ValueError("ids and embeddings must have the same length")
        if vectors.shape[1] < 1:
            raise ValueError("embeddings must have at least one dimension")
        if not ids_list:
            return 0

        first_positions = {}
        unique_positions = []
        for position, item_id in enumerate(ids_list):
            first = first_positions.setdefault(item_id, position)
            if first == position:
                unique_positions.append(position)
            elif not np.array_equal(
                vectors[first].view(np.uint16), vectors[position].view(np.uint16)
            ):
                raise ValueError(f"conflicting vectors for duplicate ID {item_id!r}")

        collection = self._collection(name)
        collection.mkdir(parents=True, exist_ok=True)
        db_path = collection / "ids.sqlite3"
        raw_path = collection / "embeddings.f16"
        if not db_path.exists() and raw_path.exists() and raw_path.stat().st_size:
            raise RuntimeError(
                f"collection {name!r} has FP16 vectors but no ID database"
            )
        started = time.monotonic()

        with closing(sqlite3.connect(db_path)) as connection:
            connection.execute("PRAGMA synchronous=FULL")
            with connection:
                connection.execute(
                    "CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL)"
                )
                connection.execute(
                    'CREATE TABLE IF NOT EXISTS vectors (id TEXT PRIMARY KEY, "row" INTEGER NOT NULL UNIQUE)'
                )
                saved_dimension = connection.execute(
                    "SELECT value FROM metadata WHERE key = 'dimension'"
                ).fetchone()
                if saved_dimension is None:
                    connection.execute(
                        "INSERT INTO metadata(key, value) VALUES ('dimension', ?)",
                        (str(vectors.shape[1]),),
                    )
                elif int(saved_dimension[0]) != vectors.shape[1]:
                    raise ValueError(
                        f"collection {name!r} has dimension {saved_dimension[0]}, "
                        f"received {vectors.shape[1]}"
                    )
                if connection.execute(
                    "SELECT 1 FROM metadata WHERE key = 'next_row'"
                ).fetchone() is None:
                    row_count = connection.execute("SELECT COUNT(*) FROM vectors").fetchone()[0]
                    connection.execute(
                        "INSERT INTO metadata(key, value) VALUES ('next_row', ?)",
                        (str(row_count),),
                    )

            dimension, count = self._state(connection, raw_path)
            expected_bytes = count * dimension * _FP16_DISK_DTYPE.itemsize
            self._trim_uncommitted_tail(raw_path, expected_bytes)
            unique_ids = [ids_list[position] for position in unique_positions]
            existing = self._rows_for_ids(connection, unique_ids)

            if existing:
                saved = np.memmap(
                    raw_path, dtype=_FP16_DISK_DTYPE, mode="r", shape=(count, dimension)
                )
                for position in unique_positions:
                    item_id = ids_list[position]
                    if item_id in existing and not np.array_equal(
                        vectors[position].view(np.uint16),
                        saved[existing[item_id]].view(np.uint16),
                    ):
                        raise ValueError(f"conflicting vector for existing ID {item_id!r}")
                del saved

            new_positions = [
                position for position in unique_positions if ids_list[position] not in existing
            ]
            if not new_positions:
                log.info("All %d IDs already saved in %s", len(ids_list), name)
                return 0

            new_vectors = np.ascontiguousarray(vectors[new_positions], dtype=_FP16_DISK_DTYPE)
            with raw_path.open("ab") as stream:
                stream.write(new_vectors.tobytes(order="C"))
                stream.flush()
                os.fsync(stream.fileno())

            # Commit ID visibility only after all associated vector bytes are durable.
            with connection:
                connection.executemany(
                    'INSERT INTO vectors(id, "row") VALUES (?, ?)',
                    (
                        (ids_list[position], count + offset)
                        for offset, position in enumerate(new_positions)
                    ),
                )
                connection.execute(
                    "UPDATE metadata SET value = ? WHERE key = 'next_row'",
                    (str(count + len(new_positions)),),
                )

        self._indexes.pop(name, None)
        log.info(
            "Saved %d FP16 vectors to %s (%d existing; %.2fs)",
            len(new_positions),
            name,
            len(ids_list) - len(new_positions),
            time.monotonic() - started,
        )
        return len(new_positions)

    def get_embeddings(self, name, ids):
        """Fetch exact saved FP16 vectors by ID in request order.

        Raises ``KeyError`` when any requested ID is missing.
        """
        ids_list = self._check_ids(ids)
        collection = self._collection(name)
        db_path = collection / "ids.sqlite3"
        if not db_path.exists():
            if ids_list:
                raise KeyError(ids_list[0])
            return np.empty((0, 0), dtype=np.float16)

        raw_path = collection / "embeddings.f16"
        with closing(sqlite3.connect(db_path)) as connection:
            dimension, count = self._state(connection, raw_path)
            if not ids_list:
                return np.empty((0, dimension), dtype=np.float16)
            rows = self._rows_for_ids(connection, ids_list)
            for item_id in ids_list:
                if item_id not in rows:
                    raise KeyError(item_id)
            row_numbers = [rows[item_id] for item_id in ids_list]

        saved = np.memmap(
            raw_path, dtype=_FP16_DISK_DTYPE, mode="r", shape=(count, dimension)
        )
        result = np.array(saved[row_numbers], dtype=np.float16, copy=True)
        del saved
        log.info("Fetched %d FP16 vectors by ID from %s", len(ids_list), name)
        return result

    def build_index(self, name):
        """Train or extend the persistent IVF/SQfp16 index; return its size.

        A partially written temporary index is ignored on the next call. The
        prior complete index remains usable until the new one is published.
        """
        collection = self._collection(name)
        db_path = collection / "ids.sqlite3"
        if not db_path.exists():
            raise ValueError(f"collection {name!r} does not exist")
        raw_path = collection / "embeddings.f16"
        index_path = collection / "index.faiss"

        with closing(sqlite3.connect(db_path)) as connection:
            dimension, count = self._state(connection, raw_path)
        if count == 0:
            raise ValueError(f"collection {name!r} has no vectors")

        faiss = self._faiss()
        device = self._resolved_device(faiss)
        started = time.monotonic()
        sample_count = min(count, self.training_size)
        desired_lists = min(self.nlist, max(1, sample_count // 40))
        cpu_index = None
        if index_path.exists():
            cpu_index = faiss.read_index(str(index_path))
            self._check_index(faiss, cpu_index, dimension, count)
            if int(cpu_index.nlist) < desired_lists:
                log.info(
                    "Rebuilding %s index with %d lists (was %d)",
                    name,
                    desired_lists,
                    cpu_index.nlist,
                )
                cpu_index = None
            else:
                start_row = int(cpu_index.ntotal)
                if start_row == count:
                    log.info("Index for %s already contains all %d vectors", name, count)
                    return count
                log.info("Extending %s index from %d to %d vectors", name, start_row, count)
        if cpu_index is None:
            cpu_index = faiss.IndexIVFScalarQuantizer(
                faiss.IndexFlatIP(dimension),
                dimension,
                desired_lists,
                faiss.ScalarQuantizer.QT_fp16,
                faiss.METRIC_INNER_PRODUCT,
                False,  # Store FP16 vector components, not centroid residuals.
            )
            start_row = 0
            log.info(
                "Creating %s IVF/SQfp16 index: %d vectors, %d dimensions, %d lists, %s",
                name,
                count,
                dimension,
                desired_lists,
                device,
            )

        if device == "gpu":
            torch, cuda_device = self._gpu_tensor_support()
            log.info(
                "Building %s using CUDA tensor data path on %s: "
                "stored dtype=float16, transfer dtype=float16, FAISS input dtype=float32",
                name,
                cuda_device,
            )

        index, resources = self._to_device(faiss, cpu_index, device)
        if device == "gpu" and (
            not hasattr(index, "getDevice")
            or int(index.getDevice()) != self.gpu_id
            or not any(
                all(hasattr(index, method) for method in methods)
                for methods in (("train_ex", "add_with_ids_ex"), ("train_c", "add_with_ids_c"))
            )
        ):
            raise RuntimeError(
                "FAISS GPU index does not support CUDA tensor train/add_with_ids "
                f"on device {self.gpu_id}"
            )
        saved = np.memmap(
            raw_path, dtype=_FP16_DISK_DTYPE, mode="r", shape=(count, dimension)
        )
        # torch_utils uses the current PyTorch stream for FAISS GPU operations.
        # Select the same logical CUDA device as index_cpu_to_gpu above.
        with torch.cuda.device(cuda_device) if device == "gpu" else nullcontext():
            if not index.is_trained:
                if sample_count == count:
                    sample_vectors = saved
                else:
                    sample_rows = np.sort(
                        np.random.default_rng(0).choice(count, size=sample_count, replace=False)
                    )
                    sample_vectors = saved[sample_rows]
                if device == "gpu":
                    log.info(
                        "Training FAISS GPU index on %d vectors using GPU-side "
                        "FP16->FP32 conversion",
                        sample_count,
                    )
                    sample_gpu_fp32 = self._fp16_numpy_to_cuda_fp32(
                        torch, sample_vectors, cuda_device
                    )
                    index.train(sample_gpu_fp32)
                    del sample_gpu_fp32
                else:
                    sample = np.ascontiguousarray(sample_vectors, dtype=np.float32)
                    index.train(sample)
                    del sample
                del sample_vectors
                log.info("Trained %s index on %d sampled vectors", name, sample_count)

            for first in range(start_row, count, self.add_batch_size):
                last = min(first + self.add_batch_size, count)
                if device == "gpu":
                    batch_gpu_fp32 = self._fp16_numpy_to_cuda_fp32(
                        torch, saved[first:last], cuda_device
                    )
                    labels_gpu = torch.arange(
                        first, last, dtype=torch.int64, device=cuda_device
                    )
                    index.add_with_ids(batch_gpu_fp32, labels_gpu)
                    del batch_gpu_fp32, labels_gpu
                else:
                    batch = np.ascontiguousarray(saved[first:last], dtype=np.float32)
                    labels = np.arange(first, last, dtype=np.int64)
                    index.add_with_ids(batch, labels)
            if device == "gpu":
                torch.cuda.synchronize(cuda_device)
        del saved
        if int(index.ntotal) != count:
            raise RuntimeError(
                f"FAISS indexed {index.ntotal} vectors but {count} were expected"
            )

        disk_index = faiss.index_gpu_to_cpu(index) if device == "gpu" else index
        self._check_index(faiss, disk_index, dimension, count)
        temporary_path = index_path.with_name(f"index.{uuid.uuid4().hex}.tmp")
        try:
            faiss.write_index(disk_index, str(temporary_path))
            with temporary_path.open("rb") as stream:
                os.fsync(stream.fileno())
            os.replace(temporary_path, index_path)
        finally:
            temporary_path.unlink(missing_ok=True)

        self._indexes[name] = (index, resources, count)
        log.info(
            "Built %s index with %d FP16 vectors on %s in %.2fs",
            name,
            count,
            device,
            time.monotonic() - started,
        )
        return count

    def search_similar(self, name, query_vector, k=10, *, nprobe=None):
        """Return the top-k ``(ID, inner-product score)`` matches for one vector."""
        query = self._check_vectors(query_vector, ndim=1)
        return self.search_similar_batch(name, query[np.newaxis, :], k, nprobe=nprobe)[0]

    def search_similar_batch(self, name, query_vectors, k=10, *, nprobe=None):
        """Search many FP16 queries together and return matches per query."""
        queries = self._check_vectors(query_vectors, ndim=2)
        if k < 1 or k > 1024:
            raise ValueError("k must be between 1 and 1024")
        probes = self.nprobe if nprobe is None else nprobe
        if probes < 1:
            raise ValueError("nprobe must be positive")

        index, dimension, db_path = self._search_index(name)
        if queries.shape[1] != dimension:
            raise ValueError(
                f"collection {name!r} has dimension {dimension}, "
                f"received {queries.shape[1]}"
            )
        if len(queries) == 0:
            return []

        index.nprobe = min(probes, index.nlist)
        started = time.monotonic()
        scores, labels = index.search(
            np.ascontiguousarray(queries, dtype=np.float32), k
        )
        found_rows = list({int(row) for row in labels.flat if row >= 0})
        with closing(sqlite3.connect(db_path)) as connection:
            found_ids = self._ids_for_rows(connection, found_rows)
        if len(found_ids) != len(found_rows):
            raise RuntimeError(f"index and ID map disagree for collection {name!r}")

        result = [
            [
                (found_ids[int(row)], float(score))
                for score, row in zip(query_scores, query_labels)
                if row >= 0
            ]
            for query_scores, query_labels in zip(scores, labels)
        ]
        log.info(
            "Searched %d queries in %s (k=%d, nprobe=%d, %.3fs)",
            len(queries),
            name,
            k,
            index.nprobe,
            time.monotonic() - started,
        )
        return result

    def _search_index(self, name):
        collection = self._collection(name)
        db_path = collection / "ids.sqlite3"
        index_path = collection / "index.faiss"
        if not db_path.exists():
            raise ValueError(f"collection {name!r} does not exist")
        with closing(sqlite3.connect(db_path)) as connection:
            dimension, count = self._state(connection, collection / "embeddings.f16")
        if not index_path.exists():
            raise RuntimeError(f"build_index({name!r}) must run before similarity search")
        cached = self._indexes.get(name)
        if cached is not None and cached[2] == count:
            return cached[0], dimension, db_path

        faiss = self._faiss()
        cpu_index = faiss.read_index(str(index_path))
        self._check_index(faiss, cpu_index, dimension, count)
        if int(cpu_index.ntotal) != count:
            raise RuntimeError(
                f"index for {name!r} is stale; run build_index after storing vectors"
            )
        device = self._resolved_device(faiss)
        index, resources = self._to_device(faiss, cpu_index, device)
        self._indexes[name] = (index, resources, count)
        log.info("Loaded %s index with %d vectors on %s", name, count, device)
        return index, dimension, db_path

    def _collection(self, name):
        if not isinstance(name, str) or not _NAME_PATTERN.fullmatch(name):
            raise ValueError("collection name must contain only letters, digits, '_' or '-'")
        return self.directory / name

    @staticmethod
    def _check_ids(ids):
        result = list(ids)
        if any(not isinstance(item_id, str) or not item_id for item_id in result):
            raise ValueError("all IDs must be nonempty strings")
        return result

    @staticmethod
    def _check_vectors(vectors, *, ndim):
        result = np.asarray(vectors)
        if result.dtype != np.float16 or result.ndim != ndim:
            raise ValueError(f"vectors must be a {ndim}D numpy.float16 array")
        if not np.isfinite(result).all():
            raise ValueError("vectors must contain only finite values")
        return result

    @staticmethod
    def _state(connection, raw_path):
        found = connection.execute(
            "SELECT value FROM metadata WHERE key = 'dimension'"
        ).fetchone()
        if found is None:
            raise RuntimeError("collection is missing its vector dimension")
        dimension = int(found[0])
        saved_count = connection.execute(
            "SELECT value FROM metadata WHERE key = 'next_row'"
        ).fetchone()
        if saved_count is None:
            raise RuntimeError("collection is missing its vector count")
        count = int(saved_count[0])
        expected_bytes = count * dimension * _FP16_DISK_DTYPE.itemsize
        actual_bytes = raw_path.stat().st_size if raw_path.exists() else 0
        if actual_bytes < expected_bytes:
            raise RuntimeError(
                f"FP16 vector file is truncated: {actual_bytes} < {expected_bytes} bytes"
            )
        return dimension, count

    @staticmethod
    def _trim_uncommitted_tail(raw_path, expected_bytes):
        if not raw_path.exists():
            return
        actual_bytes = raw_path.stat().st_size
        if actual_bytes > expected_bytes:
            with raw_path.open("r+b") as stream:
                stream.truncate(expected_bytes)
                stream.flush()
                os.fsync(stream.fileno())
            log.warning(
                "Discarded %d uncommitted bytes from %s",
                actual_bytes - expected_bytes,
                raw_path,
            )

    @staticmethod
    def _rows_for_ids(connection, ids):
        rows = {}
        for first in range(0, len(ids), _SQL_BATCH_SIZE):
            chunk = ids[first : first + _SQL_BATCH_SIZE]
            placeholders = ",".join("?" for _ in chunk)
            query = f'SELECT id, "row" FROM vectors WHERE id IN ({placeholders})'
            rows.update((item_id, int(row)) for item_id, row in connection.execute(query, chunk))
        return rows

    @staticmethod
    def _ids_for_rows(connection, rows):
        ids = {}
        for first in range(0, len(rows), _SQL_BATCH_SIZE):
            chunk = rows[first : first + _SQL_BATCH_SIZE]
            placeholders = ",".join("?" for _ in chunk)
            query = f'SELECT "row", id FROM vectors WHERE "row" IN ({placeholders})'
            ids.update((int(row), item_id) for row, item_id in connection.execute(query, chunk))
        return ids

    @staticmethod
    def _faiss():
        try:
            return importlib.import_module("faiss")
        except ImportError as error:
            raise RuntimeError(
                "FAISS is required for build_index and similarity search; "
                "install a compatible faiss-cpu or faiss-gpu package"
            ) from error

    def _resolved_device(self, faiss):
        if self.device == "cpu":
            return "cpu"
        supported = all(
            hasattr(faiss, symbol)
            for symbol in ("get_num_gpus", "StandardGpuResources", "index_cpu_to_gpu")
        )
        try:
            available = supported and faiss.get_num_gpus() > self.gpu_id
        except Exception:
            available = False
        if available:
            return "gpu"
        if self.device == "gpu":
            raise RuntimeError(
                f"FAISS GPU device {self.gpu_id} is unavailable in this process"
            )
        return "cpu"

    def _to_device(self, faiss, cpu_index, device):
        if device == "cpu":
            return cpu_index, None
        resources = faiss.StandardGpuResources()
        return faiss.index_cpu_to_gpu(resources, self.gpu_id, cpu_index), resources

    def _gpu_tensor_support(self):
        """Load the optional CUDA tensor bridge only for GPU index building."""
        try:
            torch = importlib.import_module("torch")
            importlib.import_module("faiss.contrib.torch_utils")
        except Exception as error:
            raise RuntimeError(
                "GPU-optimized index building requires compatible CUDA PyTorch "
                "and faiss.contrib.torch_utils; no CPU FP32 fallback is used"
            ) from error

        try:
            cuda_device = torch.device(f"cuda:{self.gpu_id}")
            with torch.cuda.device(cuda_device):
                torch.empty(0, dtype=torch.float16, device=cuda_device)
        except Exception as error:
            raise RuntimeError(
                f"GPU-optimized index building cannot initialize PyTorch on cuda:{self.gpu_id}"
            ) from error
        return torch, cuda_device

    @staticmethod
    def _fp16_numpy_to_cuda_fp32(torch, vectors, cuda_device):
        """Transfer FP16 to CUDA before converting it to FAISS input FP32."""
        fp16_array = np.ascontiguousarray(vectors, dtype=np.float16)
        if not fp16_array.flags.writeable:
            # torch.from_numpy cannot safely wrap the read-only memmap view.
            fp16_array = fp16_array.copy()
        cpu_fp16 = torch.from_numpy(fp16_array)
        cuda_fp16 = cpu_fp16.to(device=cuda_device)
        cuda_fp32 = cuda_fp16.float()
        del cuda_fp16, cpu_fp16
        return cuda_fp32

    @staticmethod
    def _check_index(faiss, index, dimension, count):
        if (
            not isinstance(index, faiss.IndexIVFScalarQuantizer)
            or int(index.d) != dimension
            or index.metric_type != faiss.METRIC_INNER_PRODUCT
            or index.sq.qtype != faiss.ScalarQuantizer.QT_fp16
            or index.by_residual
            or int(index.ntotal) > count
        ):
            raise RuntimeError("saved FAISS index is incompatible with FP16 IVF collection")
