"""Content digests used to compare outputs with the stored references."""

import hashlib
import json

import torch


def state_dict_digest(path) -> str:
    """sha256 over the checkpoint's tensors (names and raw bytes, in key order)."""
    sd = torch.load(path, map_location="cpu", weights_only=False)["state_dict"]
    h = hashlib.sha256()
    for name, t in sd.items():
        h.update(name.encode())
        h.update(t.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def json_digest(path) -> str:
    """sha256 of a JSON file's parsed content (independent of whitespace)."""
    data = json.load(open(path))
    return hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()


def file_digest(path) -> str:
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def checkpoint_digest(path) -> str:
    """sha256 over every tensor stored in a checkpoint (state_dict and extra
    tensors such as normaliser statistics) plus its JSON-serialised cfg."""
    blob = torch.load(path, map_location="cpu", weights_only=False)
    h = hashlib.sha256()
    for key in sorted(blob):
        if key == "cfg":
            h.update(json.dumps(blob[key], sort_keys=True).encode())
            continue
        items = blob[key].items() if isinstance(blob[key], dict) else [(key, blob[key])]
        for name, t in items:
            h.update(str(name).encode())
            h.update(t.detach().contiguous().numpy().tobytes())
    return h.hexdigest()
