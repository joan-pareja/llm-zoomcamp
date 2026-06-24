import numpy as np
import onnxruntime as ort
from pathlib import Path
from typing import overload

from tokenizers import Tokenizer


class Embedder:
    def __init__(self, path: str | Path | None = None):
        repo_root = Path(__file__).resolve().parent.parent

        if path is None:
            path = repo_root / "models" / "Xenova" / "all-MiniLM-L6-v2"
        else:
            path = Path(path)
            if not path.is_absolute():
                path = repo_root / path

        self.tokenizer = Tokenizer.from_file(str(path / "tokenizer.json"))
        self.session = ort.InferenceSession(
            str(path / "model.onnx"), providers=["CPUExecutionProvider"]
        )
        self.input_names = {inp.name for inp in self.session.get_inputs()}

    @overload
    def encode(self, text: str, normalize: bool = True) -> np.ndarray:
        ...

    @overload
    def encode(self, text: list[str], normalize: bool = True) -> np.ndarray:
        ...

    def encode(self, text: str | list[str], normalize: bool = True) -> np.ndarray:
        if isinstance(text, str):
            return self.encode_batch([text], normalize=normalize)[0]

        return self.encode_batch(text, normalize=normalize)

    def encode_batch(
        self,
        texts: list[str],
        normalize: bool = True,
    ) -> np.ndarray:
        self.tokenizer.enable_padding()
        encoded = self.tokenizer.encode_batch(texts)
        feed = {}
        if "input_ids" in self.input_names:
            feed["input_ids"] = np.array([e.ids for e in encoded], dtype=np.int64)
        if "attention_mask" in self.input_names:
            feed["attention_mask"] = np.array(
                [e.attention_mask for e in encoded], dtype=np.int64
            )
        if "token_type_ids" in self.input_names:
            feed["token_type_ids"] = np.array(
                [e.type_ids for e in encoded], dtype=np.int64
            )
        hidden = self.session.run(None, feed)[0]
        mask = feed["attention_mask"][..., None]
        pooled = (hidden * mask).sum(axis=1) / mask.sum(axis=1)
        if normalize:
            pooled = pooled / np.linalg.norm(pooled, axis=1, keepdims=True)
        return pooled
