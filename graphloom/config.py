"""
GraphLoom Configuration
=======================
Configuration settings for the GraphLoom RAG system.
"""

import os
from dataclasses import dataclass, field
from typing import Optional, List


@dataclass
class EmbeddingConfig:
    """Configuration for embedding endpoint and local model device hints."""
    model_name: str = "Qwen/Qwen3-VL-Embedding-2B"
    embedding_dim: int = 2048
    max_length: int = 32768
    modal_url: str = "https://ullahimran914--qwen3-vl-embedding-serve.modal.run"
    request_timeout: int = 60
    device: Optional[str] = None  # e.g., "cuda:0", "cuda:1", or "cpu"
    torch_dtype: str = "float16"  # "float16", "bfloat16", or "float32"
    use_flash_attention: bool = False


@dataclass
class VLMConfig:
    """Configuration for the Vision-Language Model (OpenRouter)."""
    api_key: Optional[str] = None
    base_url: str = "https://openrouter.ai/api/v1"
    model: str = "qwen/qwen-2.5-vl-7b-instruct"
    max_tokens: int = 2000
    temperature: float = 0.7
    site_url: str = ""  # Optional: for OpenRouter rankings
    site_name: str = "GraphLoom"


@dataclass
class AnswerGenConfig:
    """Configuration for the Answer Generation Model (Groq)."""
    api_key: Optional[str] = None
    model: str = "meta-llama/llama-4-scout-17b-16e-instruct"
    max_tokens: int = 1024
    temperature: float = 0.7
    top_p: float = 0.9


@dataclass
class KnowledgeGraphConfig:
    """Configuration for Knowledge Graph construction and retrieval."""
    # Triple extraction
    rebel_model: str = "Babelscape/rebel-large"
    extraction_confidence_threshold: float = 0.5  # tau_edge
    
    # External KG enrichment
    kb_confidence_threshold: float = 0.7  # tau_KB
    max_enrichment_hops: int = 2
    
    # Subgraph retrieval
    seed_size: int = 15  # K
    expansion_hops: int = 2  # m
    degree_cap: int = 5  # d_max
    edge_budget: int = 50  # N_max


@dataclass
class RouterConfig:
    """Configuration for the reliability-calibrated slot router."""
    top_k: int = 4  # k
    activation_threshold: float = 0.3  # epsilon
    reliability_bias: float = 2.0  # beta
    gate_bias: float = -1.0  # b_g


@dataclass
class RevisionConfig:
    """Configuration for verification and revision."""
    nli_model: str = "facebook/bart-large-mnli"  # Public model, no auth required
    entailment_threshold: float = 0.7  # tau_ent
    adequacy_threshold: float = 0.6  # tau_eval


@dataclass
class GraphLoomConfig:
    """Main configuration for GraphLoom."""
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    vlm: VLMConfig = field(default_factory=VLMConfig)
    answer_gen: AnswerGenConfig = field(default_factory=AnswerGenConfig)
    kg: KnowledgeGraphConfig = field(default_factory=KnowledgeGraphConfig)
    router: RouterConfig = field(default_factory=RouterConfig)
    revision: RevisionConfig = field(default_factory=RevisionConfig)
    
    # Data paths
    data_dir: str = "./data"
    images_dir: str = "./data/images/final_dataset_images"
    triples_path: str = "./data/triples.json"
    qa_dataset_path: str = "./data/qa_dataset.json"
    
    @classmethod
    def from_env(cls) -> "GraphLoomConfig":
        """Create configuration from environment variables."""
        config = cls()
        
        # Load API keys from environment
        config.vlm.api_key = os.getenv("OPENROUTER_API_KEY")
        config.answer_gen.api_key = os.getenv("GROQ_API_KEY")
        
        # Load device configuration
        device = os.getenv("GRAPHLOOM_DEVICE")
        if device:
            config.embedding.device = device
        
        return config
    
    def set_device(self, device: str):
        """Set the device for local models (e.g., 'cuda:0', 'cuda:1', 'cpu')."""
        self.embedding.device = device
    
    def set_gpu(self, gpu_id: int):
        """Set GPU by ID (e.g., 0, 1, 2)."""
        self.embedding.device = f"cuda:{gpu_id}"
