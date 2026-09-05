"""
GraphLoom Embedding Module
==========================
Multimodal embedding using a deployed Qwen3-VL-Embedding-2B Modal endpoint.
"""

import base64
import os
from io import BytesIO
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import requests
from PIL import Image


class Qwen3VLEmbedder:
    """
    Qwen3-VL-Embedding wrapper for a deployed Modal endpoint.

    Supports text, images, and multimodal (text + image) inputs.
    """

    DEFAULT_MODAL_URL = "https://ullahimran914--qwen3-vl-embedding-serve.modal.run"
    DEFAULT_MODEL_NAME = "Qwen/Qwen3-VL-Embedding-2B"
    _EXTENSION_TO_MIME = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".gif": "image/gif",
        ".webp": "image/webp",
        ".bmp": "image/bmp",
    }
    _PIL_FORMAT_TO_MIME = {
        "JPEG": "image/jpeg",
        "JPG": "image/jpeg",
        "PNG": "image/png",
        "GIF": "image/gif",
        "WEBP": "image/webp",
        "BMP": "image/bmp",
    }

    def __init__(
        self,
        model_name: Optional[str] = None,
        embedding_dim: int = 2048,
        modal_url: Optional[str] = None,
        request_timeout: int = 60,
    ):
        """
        Initialize the remote embedding client.

        Args:
            model_name: Embedding model name passed to Modal endpoint
            embedding_dim: Expected output embedding dimension
            modal_url: Base URL for deployed embedding endpoint
            request_timeout: Timeout in seconds for each embedding request
        """
        self.model_name = model_name or self.DEFAULT_MODEL_NAME
        self.embedding_dim = embedding_dim
        self.request_timeout = request_timeout
        self.modal_url = (
            modal_url
            or os.getenv("GRAPHLOOM_EMBEDDING_URL")
            or self.DEFAULT_MODAL_URL
        ).rstrip("/")

        self._session: Optional[requests.Session] = None
        self._loaded = False

    def load(self):
        """Initialize HTTP session for embedding requests."""
        if self._loaded:
            return
        self._session = requests.Session()
        self._loaded = True

    def _encode_image_as_data_url(self, image_source: Union[str, Path, Image.Image]) -> str:
        """Convert local image path or PIL image to a data URL."""
        if isinstance(image_source, Path):
            image_source = str(image_source)

        if isinstance(image_source, str):
            if image_source.startswith("data:"):
                return image_source
            if image_source.startswith(("http://", "https://")):
                return image_source

            image_path = Path(image_source)
            if not image_path.exists():
                raise FileNotFoundError(f"Image not found: {image_source}")

            image_data = base64.b64encode(image_path.read_bytes()).decode("utf-8")
            mime_type = self._EXTENSION_TO_MIME.get(image_path.suffix.lower(), "image/jpeg")
            return f"data:{mime_type};base64,{image_data}"

        if isinstance(image_source, Image.Image):
            buffer = BytesIO()
            image_format = (image_source.format or "PNG").upper()
            image_source.save(buffer, format=image_format)
            image_data = base64.b64encode(buffer.getvalue()).decode("utf-8")
            mime_type = self._PIL_FORMAT_TO_MIME.get(image_format, "image/png")
            return f"data:{mime_type};base64,{image_data}"

        raise TypeError(f"Unsupported image type: {type(image_source)}")

    def _build_payload(
        self,
        text: str,
        image: Optional[Union[str, Path, Image.Image]],
        instruction: str,
    ) -> Dict:
        """Build request payload for a single embedding call."""
        user_content = []

        if image is not None:
            data_url = self._encode_image_as_data_url(image)
            user_content.append(
                {"type": "image_url", "image_url": {"url": data_url}}
            )

        if text:
            user_content.append({"type": "text", "text": text})

        if not user_content:
            user_content = [{"type": "text", "text": ""}]

        return {
            "messages": [
                {
                    "role": "system",
                    "content": [{"type": "text", "text": instruction}],
                },
                {
                    "role": "user",
                    "content": user_content,
                },
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": ""}],
                },
            ],
            "model": self.model_name,
            "encoding_format": "float",
            "continue_final_message": True,
            "add_special_tokens": True,
        }

    def _request_embedding(self, payload: Dict) -> np.ndarray:
        """Send a request to Modal embedding endpoint and parse response."""
        if self._session is None:
            self.load()

        response = self._session.post(
            f"{self.modal_url}/v1/embeddings",
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=self.request_timeout,
        )
        response.raise_for_status()

        body = response.json()
        if "data" not in body or not body["data"]:
            raise ValueError(f"Unexpected embedding response payload: {body}")

        embedding = body["data"][0].get("embedding")
        if embedding is None:
            raise ValueError(f"Missing embedding field in response: {body}")

        return np.asarray(embedding, dtype=np.float32)

    def encode(
        self,
        inputs: List[Dict],
        batch_size: int = 8,
        normalize: bool = True,
        show_progress: bool = True,
    ) -> np.ndarray:
        """
        Encode inputs to embeddings.

        Args:
            inputs: List of dicts with "text" and/or "image" keys
            batch_size: Unused (kept for API compatibility)
            normalize: Whether to L2-normalize embeddings
            show_progress: Whether to show a progress bar for large inputs

        Returns:
            numpy array of embeddings [N, embedding_dim]
        """
        _ = batch_size
        self.load()

        if not inputs:
            return np.empty((0, self.embedding_dim), dtype=np.float32)

        iterator = inputs
        if show_progress and len(inputs) > 1:
            try:
                from tqdm import tqdm

                iterator = tqdm(inputs, desc="Computing embeddings")
            except Exception:
                iterator = inputs

        embeddings = []
        for item in iterator:
            instruction = item.get("instruction", "Represent the user's input.")
            text = item.get("text", "")
            image = item.get("image")
            payload = self._build_payload(text=text, image=image, instruction=instruction)
            embeddings.append(self._request_embedding(payload))

        arr = np.vstack(embeddings)
        self.embedding_dim = int(arr.shape[1])

        if normalize:
            norms = np.linalg.norm(arr, axis=1, keepdims=True)
            arr = arr / (norms + 1e-8)

        return arr

    def encode_text(self, texts: List[str], **kwargs) -> np.ndarray:
        """Encode text-only inputs."""
        inputs = [
            {"text": t, "instruction": "Represent the user's input."}
            for t in texts
        ]
        return self.encode(inputs, **kwargs)

    def encode_images(self, images: List[Union[str, Path, Image.Image]], **kwargs) -> np.ndarray:
        """Encode image-only inputs."""
        inputs = [
            {"image": img, "instruction": "Represent the user's input."}
            for img in images
        ]
        return self.encode(inputs, **kwargs)

    def encode_multimodal(
        self,
        texts: List[str],
        images: List[Union[str, Path, Image.Image]],
        **kwargs,
    ) -> np.ndarray:
        """Encode multimodal (text + image) inputs."""
        if len(texts) != len(images):
            raise ValueError("texts and images must have same length")
        inputs = [
            {
                "text": text,
                "image": image,
                "instruction": "Represent the user's input.",
            }
            for text, image in zip(texts, images)
        ]
        return self.encode(inputs, **kwargs)

    def similarity(self, embeddings1: np.ndarray, embeddings2: np.ndarray) -> np.ndarray:
        """Compute cosine similarity between two sets of embeddings."""
        norm1 = np.linalg.norm(embeddings1, axis=1, keepdims=True)
        norm2 = np.linalg.norm(embeddings2, axis=1, keepdims=True)
        emb1_normalized = embeddings1 / (norm1 + 1e-8)
        emb2_normalized = embeddings2 / (norm2 + 1e-8)
        return emb1_normalized @ emb2_normalized.T

    def unload(self):
        """Release HTTP resources."""
        if self._session is not None:
            self._session.close()
            self._session = None
        self._loaded = False


def create_embedder(config) -> Qwen3VLEmbedder:
    """Create an embedder from config."""
    return Qwen3VLEmbedder(
        model_name=getattr(config.embedding, "model_name", None),
        embedding_dim=config.embedding.embedding_dim,
        modal_url=getattr(config.embedding, "modal_url", None),
        request_timeout=getattr(config.embedding, "request_timeout", 60),
    )


def get_local_multimodal_embedding(
    image_path: str,
    text: str,
    modal_url: str = Qwen3VLEmbedder.DEFAULT_MODAL_URL,
) -> List[float]:
    """
    Get embedding for local image + text using the deployed Modal endpoint.

    Args:
        image_path: Path to local image file
        text: Text to combine with the image
        modal_url: Modal endpoint base URL

    Returns:
        Embedding vector as a list of floats
    """
    embedder = Qwen3VLEmbedder(modal_url=modal_url)
    embedding = embedder.encode(
        [{"text": text, "image": image_path}],
        normalize=False,
        show_progress=False,
    )[0]
    embedder.unload()
    return embedding.tolist()
