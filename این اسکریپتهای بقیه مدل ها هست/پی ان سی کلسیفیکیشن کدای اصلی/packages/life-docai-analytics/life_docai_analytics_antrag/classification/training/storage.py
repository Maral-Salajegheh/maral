"""On-disk stores that never load everything into RAM.

ClipFeatureStore   CLIP vector per image_path, as write-once .npy shards plus an index.
                   Adding pages writes new shards only; reading memory-maps the shards.
Stage1Features     frozen Stage 1 page features of one run: features.npy (memory-mapped),
                   pages.csv (one row per page, stack order) and metadata.json.
"""

import csv
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch


def _sha256(path, chunk=1024 * 1024):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


class ClipFeatureStore:
    SHARD_SIZE = 4096

    def __init__(self, directory):
        self.directory = Path(directory)
        self.index_file = self.directory / "index.jsonl"
        self._location = {}          # image_path -> (shard number, row)
        self._shards = []            # shard file names, in index order
        self._open = {}              # shard number -> memmap (opened lazily, per process)
        self._pending_keys, self._pending_vectors = [], []
        if self.index_file.exists():
            for line in self.index_file.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                entry = json.loads(line)
                keys = json.loads((self.directory / f"{entry['shard']}.keys.json").read_text(encoding="utf-8"))
                number = len(self._shards)
                self._shards.append(entry["shard"])
                for row, key in enumerate(keys):
                    self._location[key] = (number, row)

    def __len__(self):
        return len(self._location) + len(self._pending_keys)

    def __contains__(self, image_path):
        return image_path in self._location

    def __getitem__(self, image_path):
        number, row = self._location[image_path]
        if number not in self._open:
            self._open[number] = np.load(self.directory / f"{self._shards[number]}.npy", mmap_mode="r")
        return torch.from_numpy(np.array(self._open[number][row], dtype=np.float32))

    def __getstate__(self):          # DataLoader workers re-open the memmaps themselves
        state = dict(self.__dict__); state["_open"] = {}
        return state

    # ---- writing ----------------------------------------------------------------
    def add(self, image_path, vector):
        self._pending_keys.append(image_path)
        self._pending_vectors.append(np.asarray(vector, dtype=np.float32))
        if len(self._pending_keys) >= self.SHARD_SIZE:
            self.flush()

    def flush(self):
        if not self._pending_keys:
            return
        self.directory.mkdir(parents=True, exist_ok=True)
        name = f"shard_{len(self._shards):05d}"
        array = np.stack(self._pending_vectors)
        tmp = self.directory / f"{name}.tmp.npy"
        np.save(tmp, array); os.replace(tmp, self.directory / f"{name}.npy")
        keys_tmp = self.directory / f"{name}.keys.json.tmp"
        keys_tmp.write_text(json.dumps(self._pending_keys), encoding="utf-8")
        os.replace(keys_tmp, self.directory / f"{name}.keys.json")
        with self.index_file.open("a", encoding="utf-8") as handle:   # the shard counts once it is indexed
            handle.write(json.dumps({"shard": name, "count": len(self._pending_keys)}) + "\n")
            handle.flush(); os.fsync(handle.fileno())
        number = len(self._shards)
        self._shards.append(name)
        for row, key in enumerate(self._pending_keys):
            self._location[key] = (number, row)
        self._pending_keys, self._pending_vectors = [], []


class Stage1Features:
    FORMAT = "stage1_features_stack"

    def __init__(self, directory):
        self.directory = Path(directory)
        self.metadata_file = self.directory / "metadata.json"
        self.metadata = json.loads(self.metadata_file.read_text(encoding="utf-8"))
        if self.metadata.get("format") != self.FORMAT:
            raise ValueError(f"Not a Stage 1 feature directory of format {self.FORMAT}: {directory}")
        with (self.directory / "pages.csv").open(newline="", encoding="utf-8") as handle:
            self.pages = list(csv.DictReader(handle))
        self._features = None

    @property
    def features(self):              # opened on first use, per process
        if self._features is None:
            self._features = np.load(self.directory / "features.npy", mmap_mode="r")
        return self._features

    def rows(self, start, end):
        return torch.from_numpy(np.array(self.features[start:end], dtype=np.float32))

    def __getstate__(self):
        state = dict(self.__dict__); state["_features"] = None
        return state

    def metadata_sha256(self):
        return _sha256(self.metadata_file)

    @classmethod
    def create(cls, directory, page_count, dim):
        """Pre-allocate features.npy; returns (writer memmap, directory)."""
        directory = Path(directory)
        if directory.exists():
            raise FileExistsError(f"Output already exists: {directory}; use another --run-name or --output")
        directory.mkdir(parents=True)
        return np.lib.format.open_memmap(directory / "features.npy", mode="w+", dtype=np.float32,
                                         shape=(page_count, dim))

    @classmethod
    def finish(cls, directory, writer, pages, metadata):
        directory = Path(directory)
        writer.flush(); del writer
        fields = list(pages[0]) if pages else ["row"]
        with (directory / "pages.csv").open("w", newline="", encoding="utf-8") as handle:
            out = csv.DictWriter(handle, fieldnames=fields); out.writeheader(); out.writerows(pages)
        metadata = {**metadata, "format": cls.FORMAT, "features_sha256": _sha256(directory / "features.npy")}
        tmp = directory / "metadata.json.tmp"
        tmp.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        os.replace(tmp, directory / "metadata.json")    # written last: a directory without it is incomplete
