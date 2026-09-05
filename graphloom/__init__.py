"""
GraphLoom - Multimodal KG-RAG Framework
=======================================

A framework for grounded multimodal question answering using
knowledge graph retrieval-augmented generation.

Based on the paper: "GraphLoom: Controlled Evidence Injection for
Interleaved KG-RAG in Multimodal QA"

Quick Start:
    ```python
    from graphloom import GraphLoom
    
    # Initialize with API keys
    gl = GraphLoom(
        openrouter_api_key="your-openrouter-key",
        groq_api_key="your-groq-key",
    )
    
    # Load existing knowledge
    gl.load_triples("data/triples.json")
    
    # Answer a question
    result = gl.answer(
        question="What is shown in the image?",
        image="path/to/image.jpg",
    )
    print(result["answer"])
    ```

Components:
    - GraphLoom: Main class for the complete pipeline
    - Qwen3VLEmbedder: Multimodal embedding model
    - VLMClient: Vision-language model client (OpenRouter)
    - AnswerGenerator: Answer generation client (Groq)
    - MultimodalKnowledgeGraph: Knowledge graph storage and retrieval
    - HieraSlotMemory: Hierarchical slot memories
    - ReliabilityCalibratedRouter: Slot routing with reliability calibration
"""

__version__ = "0.1.0"
__author__ = "GraphLoom Team"

from .config import (
    GraphLoomConfig,
    EmbeddingConfig,
    VLMConfig,
    AnswerGenConfig,
    KnowledgeGraphConfig,
    RouterConfig,
    RevisionConfig,
)

from .embedding import (
    Qwen3VLEmbedder,
    create_embedder,
)

from .vlm import (
    VLMClient,
    create_vlm_client,
)

from .answer_gen import (
    AnswerGenerator,
    create_answer_generator,
)

from .knowledge_graph import (
    Triple,
    TripleExtractor,
    MultimodalKnowledgeGraph,
    SubgraphRetriever,
)

from .slot_memory import (
    SlotMemory,
    HieraSlotMemory,
    ReliabilityCalibratedRouter,
    EvidenceMemoryFusion,
)

from .verification import (
    ClaimExtractor,
    NLIVerifier,
    RetrievalQualityEvaluator,
    AnswerVerifierReviser,
)

from .graphloom import (
    GraphLoom,
    create_graphloom,
)

__all__ = [
    # Main class
    "GraphLoom",
    "create_graphloom",
    
    # Config
    "GraphLoomConfig",
    "EmbeddingConfig",
    "VLMConfig",
    "AnswerGenConfig",
    "KnowledgeGraphConfig",
    "RouterConfig",
    "RevisionConfig",
    
    # Embedding
    "Qwen3VLEmbedder",
    "create_embedder",
    
    # VLM
    "VLMClient",
    "create_vlm_client",
    
    # Answer Generation
    "AnswerGenerator",
    "create_answer_generator",
    
    # Knowledge Graph
    "Triple",
    "TripleExtractor",
    "MultimodalKnowledgeGraph",
    "SubgraphRetriever",
    
    # Slot Memory
    "SlotMemory",
    "HieraSlotMemory",
    "ReliabilityCalibratedRouter",
    "EvidenceMemoryFusion",
    
    # Verification
    "ClaimExtractor",
    "NLIVerifier",
    "RetrievalQualityEvaluator",
    "AnswerVerifierReviser",
]
