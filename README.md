# GraphLoom

**Multimodal KG-RAG Framework for Grounded Question Answering**

GraphLoom is a Python implementation of the paper "GraphLoom: Reliability-Calibrated Graph Evidence Routing for Multimodal KG-RAG". It combines retrieval-augmented generation with instance-level multimodal knowledge graphs for accurate, evidence-grounded answers.

## Features

- **Multimodal Embedding**: Uses deployed Qwen3-VL-Embedding-2B endpoint for unified text and image embeddings
- **Vision-Language Understanding**: Integrates Qwen-2.5-VL via OpenRouter for scene description
- **Fast Answer Generation**: Uses Llama-4-Scout via Groq for efficient answer generation
- **Knowledge Graph RAG**: Builds and retrieves from multimodal knowledge graphs
- **HieraSlot Memories**: Structured slot memories with reliability-calibrated routing
- **Verification & Revision**: NLI-based answer verification and correction
- **Modal-only Embedding Backend**: Embeddings always use the deployed Modal endpoint

## Installation

### Quick Install (Recommended)

```bash
# Install system dependencies (Linux)
apt-get update -qq
apt-get install -y ffmpeg libgl1 ninja-build

# Install with uv (fastest)
cd graphloom
uv pip install -e .

# Or install with pip
pip install -e .

# Download spaCy model (for entity linking)
python -m spacy download en_core_web_sm
```

### Alternative: Install from requirements.txt

```bash
cd graphloom
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### Requirements

- Python 3.9+
- CUDA-capable GPU (optional, for local REBEL/NLI models)
- API keys for OpenRouter and Groq

## Quick Start

```python
from graphloom import GraphLoom

# Initialize with API keys
gl = GraphLoom(
    openrouter_api_key="your-openrouter-key",
    groq_api_key="your-groq-key",
)

# Load existing knowledge triples
gl.load_triples("data/triples.json")

# Build embeddings for retrieval
gl.build_embeddings()

# Answer a question with an image
result = gl.answer(
    question="What is the distance to the trail?",
    image="path/to/image.jpg",
    use_multihop=True,
    use_verification=True,
)

print(result["answer"])
print(result["evidence"])  # Evidence is always returned
```

### Modal Embedding Endpoint (Required)

GraphLoom embeddings are resolved through your deployed Modal endpoint.

```bash
export GRAPHLOOM_EMBEDDING_URL="https://ullahimran914--qwen3-vl-embedding-serve.modal.run"
```

## Detailed Usage

### 1. Using Existing Knowledge (Load from JSON)

If you already have a knowledge base in JSON format, you can load it directly:

```python
from graphloom import GraphLoom

gl = GraphLoom(
    openrouter_api_key="your-key",
    groq_api_key="your-key",
)

# Load existing triples from JSON file
gl.load_triples("data/triples.json")

# Build embeddings for similarity-based retrieval
# This computes vector embeddings for all triples via the deployed Modal endpoint
gl.build_embeddings()

# Now you can answer questions
result = gl.answer("What is the capital of France?")
print(f"Answer: {result['answer']}")
print(f"Evidence used: {len(result['evidence'])} triples")
```

### 2. Adding Knowledge Programmatically

You can add triples directly in your code:

```python
from graphloom import GraphLoom

gl = GraphLoom(
    openrouter_api_key="your-key",
    groq_api_key="your-key",
)

# Add triples as dictionaries
triples = [
    {"subject": "Paris", "relation": "is capital of", "object": "France", "confidence": 1.0},
    {"subject": "France", "relation": "is a", "object": "country", "confidence": 1.0},
    {"subject": "Eiffel Tower", "relation": "located in", "object": "Paris", "confidence": 0.95},
    {"subject": "Eiffel Tower", "relation": "height", "object": "330 meters", "confidence": 0.9},
]
gl.add_triples(triples)

# Build embeddings after adding triples
gl.build_embeddings()

# Save for later use
gl.save_triples("my_knowledge.json")
```

### 3. Extracting Knowledge from Images

GraphLoom can automatically extract triples from images using VLM:

```python
from graphloom import GraphLoom

gl = GraphLoom(
    openrouter_api_key="your-key",
    groq_api_key="your-key",
)

# Extract triples from an image
# This uses Qwen-2.5-VL direct triple extraction, then REBEL augmentation
triples = gl.extract_triples_from_image(
    image="path/to/image.jpg",
    question="What objects are in this image?"  # Optional context
)

print(f"Extracted {len(triples)} triples:")
for t in triples:
    print(f"  {t.subject} --[{t.relation}]--> {t.object}")

# Add extracted triples to knowledge graph
gl.kg.add_triples(triples)
gl.build_embeddings()
```

### 4. Answering Questions with Images (Multimodal QA)

```python
from graphloom import GraphLoom

gl = GraphLoom(
    openrouter_api_key="your-key",
    groq_api_key="your-key",
)

# Load knowledge
gl.load_triples("data/triples.json")
gl.build_embeddings()

# Answer with image context
result = gl.answer(
    question="What distance is shown on the sign?",
    image="data/images/trail_sign.jpg",
    use_multihop=True,      # Enable multi-hop reasoning
    use_verification=True,  # Enable answer verification
    extract_from_image=True,  # Also extract new triples from image
)

print(f"Question: {result['question']}")
print(f"Answer: {result['answer']}")
print(f"\nEvidence ({len(result['evidence'])} triples):")
for e in result['evidence'][:5]:
    print(f"  - {e['subject']} {e['relation']} {e['object']}")
```

### 5. Multiple Choice Questions

```python
from graphloom import GraphLoom

gl = GraphLoom(
    openrouter_api_key="your-key",
    groq_api_key="your-key",
)

gl.load_triples("data/triples.json")
gl.build_embeddings()

# Answer multiple choice question
result = gl.answer(
    question="What is the tallest mountain in the world?",
    options=["K2", "Mount Everest", "Kangchenjunga", "Lhotse"],
)

print(f"Selected answer: {result['answer']}")
```

### 6. Batch Evaluation on QA Dataset

```python
from graphloom import GraphLoom
import json

gl = GraphLoom(
    openrouter_api_key="your-key",
    groq_api_key="your-key",
)

gl.load_triples("data/triples.json")
gl.build_embeddings()

# Load QA dataset
with open("data/qa_dataset.json") as f:
    qa_dataset = json.load(f)

# Evaluate on dataset
metrics = gl.evaluate(
    qa_dataset,
    use_multihop=True,
    use_verification=True,
)

print(f"Accuracy: {metrics['accuracy']:.2%}")
print(f"Correct: {metrics['correct']}/{metrics['total']}")

# Access detailed results with evidence
for r in metrics['results'][:3]:
    print(f"\nID: {r['id']}")
    print(f"Question: {r['question']}")
    print(f"Ground Truth: {r['ground_truth']}")
    print(f"Predicted: {r['predicted']}")
    print(f"Correct: {r['is_correct']}")
    print(f"Evidence: {len(r['evidence'])} triples")
```

### 7. Text-Only QA (No Image)

```python
from graphloom import GraphLoom

gl = GraphLoom(
    openrouter_api_key="your-key",
    groq_api_key="your-key",
)

# Add some knowledge
gl.add_triples([
    {"subject": "Python", "relation": "is a", "object": "programming language"},
    {"subject": "Python", "relation": "created by", "object": "Guido van Rossum"},
    {"subject": "Python", "relation": "first released", "object": "1991"},
])
gl.build_embeddings()

# Text-only question
result = gl.answer("Who created Python?")
print(f"Answer: {result['answer']}")
print(f"Evidence: {result['evidence']}")
```

### 8. Controlling Retrieval and Verification

```python
from graphloom import GraphLoom

gl = GraphLoom(
    openrouter_api_key="your-key",
    groq_api_key="your-key",
)

gl.load_triples("data/triples.json")
gl.build_embeddings()

# Fast mode: disable multi-hop and verification
result_fast = gl.answer(
    question="What is shown?",
    image="image.jpg",
    use_multihop=False,     # Single-hop retrieval only
    use_verification=False,  # Skip NLI verification
)

# Full mode: enable all features
result_full = gl.answer(
    question="What is shown?",
    image="image.jpg",
    use_multihop=True,      # Multi-hop with hypothesis generation
    use_verification=True,  # NLI-based verification and revision
)
```

### 9. Working with the Knowledge Graph Directly

```python
from graphloom import GraphLoom

gl = GraphLoom(
    openrouter_api_key="your-key",
    groq_api_key="your-key",
)

gl.load_triples("data/triples.json")

# Access the knowledge graph
print(f"Total triples: {len(gl.kg)}")
print(f"Entities: {gl.kg.get_entities()}")

# Get triples for a specific entity
dog_triples = gl.kg.get_triples_for_entity("dog")
for t in dog_triples:
    print(f"  {t.subject} --[{t.relation}]--> {t.object}")

# Get neighbors in the graph
neighbors = gl.kg.get_neighbors("dog")
print(f"Neighbors of 'dog': {neighbors}")
```

### 10. Command Line Usage

```bash
# Set API keys
export OPENROUTER_API_KEY="your-key"
export GROQ_API_KEY="your-key"

# Answer a single question
python run.py -q "What is a dog?" -t data/triples.json

# Answer with an image
python run.py -q "What is in this image?" -i image.jpg -t data/triples.json

# Use specific GPU
python run.py -q "Question?" --gpu 1

# Evaluate on dataset
python run.py --evaluate data/qa_dataset.json -t data/triples.json --output results.json

# Disable multi-hop for faster inference
python run.py -q "Question?" --no-multihop --no-verification
```

## Configuration

### Optional Local Device Selection (REBEL/NLI only)

You can specify which GPU to use for local REBEL/NLI models.  
Embeddings always use the Modal endpoint configured by `GRAPHLOOM_EMBEDDING_URL`.

```python
# Using device string
gl = GraphLoom(device="cuda:0")  # First GPU
gl = GraphLoom(device="cuda:1")  # Second GPU
gl = GraphLoom(device="cpu")     # CPU only

# Using GPU ID
gl = GraphLoom(gpu_id=0)  # First GPU
gl = GraphLoom(gpu_id=1)  # Second GPU
```

### Custom Configuration

```python
from graphloom import GraphLoom, GraphLoomConfig

config = GraphLoomConfig()

# Embedding settings
config.embedding.model_name = "Qwen/Qwen3-VL-Embedding-2B"
config.embedding.embedding_dim = 2048
config.embedding.modal_url = "https://ullahimran914--qwen3-vl-embedding-serve.modal.run"
config.embedding.request_timeout = 60

# Optional local device for non-embedding stages (REBEL/NLI)
config.embedding.device = "cuda:0"

# VLM settings (OpenRouter)
config.vlm.model = "qwen/qwen-2.5-vl-7b-instruct"
config.vlm.max_tokens = 2000

# Answer generation settings (Groq)
config.answer_gen.model = "meta-llama/llama-4-scout-17b-16e-instruct"
config.answer_gen.max_tokens = 1024

# Knowledge graph settings
config.kg.seed_size = 15
config.kg.expansion_hops = 2
config.kg.edge_budget = 50

# Router settings
config.router.top_k = 4
config.router.activation_threshold = 0.3

gl = GraphLoom(config=config)
```

## Data Format

### Triples (triples.json)

```json
[
  {
    "subject": "Dipsea Trail",
    "relation": "distance",
    "object": "7.9 miles",
    "confidence": 0.95,
    "source": "extracted"
  },
  {
    "subject": "trail sign",
    "relation": "shows",
    "object": "distance information",
    "confidence": 0.9
  }
]
```

### QA Dataset (qa_dataset.json)

```json
[
  {
    "id": "d82d381af11d76d8a70601251ecb8ca5",
    "question": "What distance was the AP Warrior fast race at the Del Mar Racetrack?",
    "ground_truth": "Seven Furlongs",
    "image": "data/images/final_dataset_images/117d500aaa630023c4038b8268b309c0.png"
  },
  {
    "id": "example_002",
    "question": "What animal is shown?",
    "ground_truth": "dog",
    "image": "data/images/final_dataset_images/beach_dog.jpg"
  }
]
```

## API Reference

### GraphLoom

Main class for the complete pipeline.

```python
class GraphLoom:
    def __init__(
        self,
        config: Optional[GraphLoomConfig] = None,
        openrouter_api_key: Optional[str] = None,
        groq_api_key: Optional[str] = None,
        device: Optional[str] = None,
        gpu_id: Optional[int] = None,
    )
    
    def load_triples(self, path: str)
    def save_triples(self, path: str)
    def add_triples(self, triples: List[Dict])
    def build_embeddings(self)
    
    def answer(
        self,
        question: str,
        image: Optional[str] = None,
        options: Optional[List[str]] = None,
        use_multihop: bool = True,
        use_verification: bool = True,
        extract_from_image: bool = False,
    ) -> Dict  # Always returns evidence in result
    
    def batch_answer(self, questions: List[Dict], **kwargs) -> List[Dict]
    def evaluate(self, qa_dataset: List[Dict], **kwargs) -> Dict
```

### Components

- **Qwen3VLEmbedder**: Multimodal embedding model
- **VLMClient**: Vision-language model client (OpenRouter)
- **AnswerGenerator**: Answer generation client (Groq)
- **MultimodalKnowledgeGraph**: Knowledge graph storage and retrieval
- **SubgraphRetriever**: Evidence subgraph retrieval
- **HieraSlotMemory**: Hierarchical slot memories
- **ReliabilityCalibratedRouter**: Slot routing
- **AnswerVerifierReviser**: Answer verification and revision

## Environment Variables

```bash
export OPENROUTER_API_KEY="your-openrouter-api-key"
export GROQ_API_KEY="your-groq-api-key"
export GRAPHLOOM_EMBEDDING_URL="https://ullahimran914--qwen3-vl-embedding-serve.modal.run"
export GRAPHLOOM_DEVICE="cuda:0"  # Optional, local REBEL/NLI stages only
```

## Architecture

GraphLoom follows the architecture from the paper:

1. **Scene Description**: Qwen-2.5-VL generates unified scene descriptions from images
2. **Triple Extraction**: REBEL extracts structured triples from descriptions
3. **Knowledge Graph**: Instance-level MMKG stores and indexes triples
4. **Subgraph Retrieval**: Qwen3-VL-Embedding (served on Modal) retrieves relevant evidence
5. **HieraSlot Memories**: Triples converted to structured key-value memories
6. **Reliability Routing**: Top-k slots selected based on utility and reliability
7. **Answer Generation**: Llama-4-Scout generates evidence-grounded answers
8. **Verification**: DeBERTa-based NLI verifies claims against evidence
9. **Revision**: Unsupported claims are corrected using retrieved evidence

## Citation

If you use GraphLoom in your research, please cite:

```bibtex
@inproceedings{ali2026graphloom,
  title     = {GraphLoom: Reliability-Calibrated Graph Evidence Routing for Multimodal KG-RAG},
  author    = {Ali, Zafar and Khan, Asad and Malik, Aalia and Kefalas, Pavlos},
  booktitle = {Proceedings of the 35th ACM International Conference on Information and Knowledge Management (CIKM '26)},
  year      = {2026}
}
```

## License

Apache 2.0
