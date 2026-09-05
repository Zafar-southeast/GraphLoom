#!/usr/bin/env python3
"""
GraphLoom Example Usage
=======================

This script demonstrates how to use GraphLoom for multimodal question answering.

Before running:
1. Install requirements: pip install -r requirements.txt
2. Set environment variables:
   - OPENROUTER_API_KEY: Your OpenRouter API key
   - GROQ_API_KEY: Your Groq API key
3. Prepare your data in the data/ directory:
   - data/triples.json: Knowledge graph triples
   - data/qa_dataset.json: QA dataset
   - data/images/: Image files
"""

import os
import json
from pathlib import Path

# Add parent directory to path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from graphloom import GraphLoom, Triple


def example_basic_usage():
    """Basic usage example with existing triples."""
    print("=" * 60)
    print("Example 1: Basic Usage with Existing Triples")
    print("=" * 60)
    
    # Initialize GraphLoom
    # You can specify device for local models:
    # - device="cuda:0" for first GPU
    # - device="cuda:1" for second GPU
    # - device="cpu" for CPU
    # Or use gpu_id parameter:
    # - gpu_id=0 for first GPU
    # - gpu_id=1 for second GPU
    
    gl = GraphLoom(
        openrouter_api_key=os.getenv("OPENROUTER_API_KEY"),
        groq_api_key=os.getenv("GROQ_API_KEY"),
        device="cuda:0",  # Change to your preferred device
    )
    
    # Load existing triples
    triples_path = "data/triples.json"
    if os.path.exists(triples_path):
        gl.load_triples(triples_path)
    else:
        # Add some example triples manually
        example_triples = [
            {"subject": "dog", "relation": "is a", "object": "animal", "confidence": 0.9},
            {"subject": "dog", "relation": "has", "object": "four legs", "confidence": 0.95},
            {"subject": "beach", "relation": "has", "object": "sand", "confidence": 0.9},
            {"subject": "sunset", "relation": "occurs at", "object": "beach", "confidence": 0.8},
        ]
        gl.add_triples(example_triples)
        print(f"Added {len(example_triples)} example triples")
    
    # Build embeddings for retrieval
    gl.build_embeddings()
    
    # Answer a question (text-only)
    result = gl.answer(
        question="What kind of animal is a dog?",
        use_multihop=False,
        use_verification=False,
    )
    
    print(f"\nQuestion: {result['question']}")
    print(f"Answer: {result['answer']}")
    print()


def example_multimodal_qa():
    """Multimodal QA example with image."""
    print("=" * 60)
    print("Example 2: Multimodal QA with Image")
    print("=" * 60)
    
    gl = GraphLoom(
        openrouter_api_key=os.getenv("OPENROUTER_API_KEY"),
        groq_api_key=os.getenv("GROQ_API_KEY"),
        gpu_id=0,  # Use first GPU
    )
    
    # Load triples
    if os.path.exists("data/triples.json"):
        gl.load_triples("data/triples.json")
        gl.build_embeddings()
    
    # Find an image
    image_dir = Path("data/images/final_dataset_images")
    if image_dir.exists():
        images = list(image_dir.glob("*.jpg")) + list(image_dir.glob("*.JPG"))
        if images:
            image_path = str(images[0])
            print(f"Using image: {image_path}")
            
            # Answer with image (evidence is always returned)
            result = gl.answer(
                question="What is shown in this image?",
                image=image_path,
                extract_from_image=True,  # Extract triples from image
                use_multihop=True,
                use_verification=True,
            )
            
            print(f"\nQuestion: {result['question']}")
            print(f"Answer: {result['answer']}")
            print(f"Evidence triples: {len(result['evidence'])}")
            for e in result['evidence'][:5]:
                print(f"  - {e['subject']} {e['relation']} {e['object']}")
            if result['verification']:
                print(f"Verification: {result['verification']}")
        else:
            print("No images found in data/images/final_dataset_images/")
    else:
        print("Image directory not found. Skipping multimodal example.")
    print()


def example_multiple_choice():
    """Multiple choice QA example."""
    print("=" * 60)
    print("Example 3: Multiple Choice QA")
    print("=" * 60)
    
    gl = GraphLoom(
        openrouter_api_key=os.getenv("OPENROUTER_API_KEY"),
        groq_api_key=os.getenv("GROQ_API_KEY"),
        device="cuda:0",
    )
    
    # Add some knowledge
    triples = [
        {"subject": "Earth", "relation": "is a", "object": "planet", "confidence": 1.0},
        {"subject": "Earth", "relation": "orbits", "object": "Sun", "confidence": 1.0},
        {"subject": "Moon", "relation": "orbits", "object": "Earth", "confidence": 1.0},
        {"subject": "Mars", "relation": "is a", "object": "planet", "confidence": 1.0},
    ]
    gl.add_triples(triples)
    gl.build_embeddings()
    
    # Multiple choice question
    result = gl.answer(
        question="What does the Moon orbit?",
        options=["The Sun", "Earth", "Mars", "Jupiter"],
        use_verification=False,
    )
    
    print(f"\nQuestion: {result['question']}")
    print(f"Answer: {result['answer']}")
    print()


def example_batch_evaluation():
    """Batch evaluation example with user's QA dataset format."""
    print("=" * 60)
    print("Example 4: Batch Evaluation")
    print("=" * 60)
    
    gl = GraphLoom(
        openrouter_api_key=os.getenv("OPENROUTER_API_KEY"),
        groq_api_key=os.getenv("GROQ_API_KEY"),
        device="cuda:0",
    )
    
    # Load QA dataset
    # Expected format: {"question": str, "ground_truth": str, "image": str, "id": str}
    qa_path = "data/qa_dataset.json"
    if os.path.exists(qa_path):
        with open(qa_path, "r") as f:
            qa_dataset = json.load(f)
        
        # Load triples
        if os.path.exists("data/triples.json"):
            gl.load_triples("data/triples.json")
            gl.build_embeddings()
        
        # Evaluate on first 5 questions
        subset = qa_dataset[:5] if len(qa_dataset) > 5 else qa_dataset
        
        metrics = gl.evaluate(
            subset,
            use_multihop=True,
            use_verification=True,
        )
        
        print(f"\nEvaluation Results:")
        print(f"  Accuracy: {metrics['accuracy']:.2%}")
        print(f"  Correct: {metrics['correct']}/{metrics['total']}")
        
        # Show detailed results with evidence
        print("\nDetailed Results:")
        for r in metrics['results'][:3]:
            print(f"\n  ID: {r['id']}")
            print(f"  Question: {r['question']}")
            print(f"  Ground Truth: {r['ground_truth']}")
            print(f"  Predicted: {r['predicted']}")
            print(f"  Correct: {r['is_correct']}")
            print(f"  Evidence ({len(r['evidence'])} triples):")
            for e in r['evidence'][:3]:
                print(f"    - {e['subject']} {e['relation']} {e['object']}")
    else:
        print("QA dataset not found at data/qa_dataset.json")
    print()


def example_custom_config():
    """Example with custom configuration."""
    print("=" * 60)
    print("Example 5: Custom Configuration")
    print("=" * 60)
    
    from graphloom import GraphLoomConfig
    
    # Create custom config
    config = GraphLoomConfig()
    
    # Customize embedding settings
    config.embedding.model_name = "Qwen/Qwen3-VL-Embedding-2B"
    config.embedding.embedding_dim = 2048
    config.embedding.device = "cuda:0"
    config.embedding.torch_dtype = "float16"
    
    # Customize VLM settings
    config.vlm.model = "qwen/qwen-2.5-vl-7b-instruct"
    config.vlm.max_tokens = 2000
    config.vlm.temperature = 0.7
    
    # Customize answer generation settings
    config.answer_gen.model = "meta-llama/llama-4-scout-17b-16e-instruct"
    config.answer_gen.max_tokens = 1024
    
    # Customize knowledge graph settings
    config.kg.seed_size = 15
    config.kg.expansion_hops = 2
    config.kg.degree_cap = 5
    config.kg.edge_budget = 50
    
    # Customize router settings
    config.router.top_k = 4
    config.router.activation_threshold = 0.3
    
    # Set API keys
    config.vlm.api_key = os.getenv("OPENROUTER_API_KEY")
    config.answer_gen.api_key = os.getenv("GROQ_API_KEY")
    
    # Create GraphLoom with custom config
    gl = GraphLoom(config=config)
    
    print("GraphLoom initialized with custom configuration:")
    print(f"  Embedding model: {config.embedding.model_name}")
    print(f"  VLM model: {config.vlm.model}")
    print(f"  Answer model: {config.answer_gen.model}")
    print(f"  Device: {config.embedding.device}")
    print()


def main():
    """Run all examples."""
    print("\n" + "=" * 60)
    print("GraphLoom Examples")
    print("=" * 60 + "\n")
    
    # Check for API keys
    if not os.getenv("OPENROUTER_API_KEY"):
        print("Warning: OPENROUTER_API_KEY not set")
    if not os.getenv("GROQ_API_KEY"):
        print("Warning: GROQ_API_KEY not set")
    
    # Run examples
    try:
        example_custom_config()  # This doesn't need API calls
    except Exception as e:
        print(f"Example 5 failed: {e}")
    
    # Only run API-dependent examples if keys are set
    if os.getenv("OPENROUTER_API_KEY") and os.getenv("GROQ_API_KEY"):
        try:
            example_basic_usage()
        except Exception as e:
            print(f"Example 1 failed: {e}")
        
        try:
            example_multiple_choice()
        except Exception as e:
            print(f"Example 3 failed: {e}")
        
        try:
            example_multimodal_qa()
        except Exception as e:
            print(f"Example 2 failed: {e}")
        
        try:
            example_batch_evaluation()
        except Exception as e:
            print(f"Example 4 failed: {e}")
    else:
        print("\nSkipping API-dependent examples (API keys not set)")
    
    print("\n" + "=" * 60)
    print("Examples completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
