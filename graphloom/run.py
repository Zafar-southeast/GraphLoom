#!/usr/bin/env python3
"""
GraphLoom - Simple Run Script
=============================

A simple script to run GraphLoom for multimodal question answering.

Usage:
    python run.py --question "What is shown in the image?" --image path/to/image.jpg
    python run.py --question "How far is the trail?" --triples data/triples.json
    python run.py --evaluate data/qa_dataset.json --triples data/triples.json

Environment Variables:
    OPENROUTER_API_KEY: Your OpenRouter API key
    GROQ_API_KEY: Your Groq API key
"""

import argparse
import json
import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def main():
    parser = argparse.ArgumentParser(
        description="GraphLoom - Multimodal KG-RAG for Question Answering",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Answer a single question
  python run.py -q "What is a dog?" -t data/triples.json

  # Answer with an image
  python run.py -q "What is in this image?" -i image.jpg -t data/triples.json

  # Multiple choice question
  python run.py -q "What animal is shown?" -i image.jpg --options "cat,dog,bird"

  # Evaluate on dataset
  python run.py --evaluate data/qa_dataset.json -t data/triples.json

  # Use specific GPU
  python run.py -q "Question?" --gpu 1
        """
    )
    
    # Question answering arguments
    parser.add_argument("-q", "--question", type=str, help="Question to answer")
    parser.add_argument("-i", "--image", type=str, help="Path to image file")
    parser.add_argument("-t", "--triples", type=str, default="data/triples.json",
                        help="Path to triples JSON file (default: data/triples.json)")
    parser.add_argument("--options", type=str, help="Comma-separated answer options for MCQ")
    
    # Evaluation arguments
    parser.add_argument("--evaluate", type=str, help="Path to QA dataset for evaluation")
    
    # Model configuration
    parser.add_argument("--gpu", type=int, default=0, help="GPU ID to use (default: 0)")
    parser.add_argument("--device", type=str, help="Device string (e.g., 'cuda:0', 'cpu')")
    parser.add_argument("--no-multihop", action="store_true", help="Disable multi-hop retrieval")
    parser.add_argument("--no-verification", action="store_true", help="Disable answer verification")
    parser.add_argument("--extract-from-image", action="store_true", 
                        help="Extract triples from image before answering")
    
    # Output options
    parser.add_argument("--output", type=str, help="Output file for results (JSON)")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    
    # Validate arguments
    if not args.question and not args.evaluate:
        parser.error("Either --question or --evaluate is required")
    
    # Check API keys
    if not os.getenv("OPENROUTER_API_KEY"):
        print("Error: OPENROUTER_API_KEY environment variable not set")
        print("Set it with: export OPENROUTER_API_KEY='your-key'")
        sys.exit(1)
    
    if not os.getenv("GROQ_API_KEY"):
        print("Error: GROQ_API_KEY environment variable not set")
        print("Set it with: export GROQ_API_KEY='your-key'")
        sys.exit(1)
    
    # Import GraphLoom
    from graphloom import GraphLoom
    
    # Determine device
    device = args.device if args.device else f"cuda:{args.gpu}"
    
    if args.verbose:
        print(f"Initializing GraphLoom on {device}...")
    
    # Initialize GraphLoom
    gl = GraphLoom(
        openrouter_api_key=os.getenv("OPENROUTER_API_KEY"),
        groq_api_key=os.getenv("GROQ_API_KEY"),
        device=device,
    )
    
    # Load triples
    if os.path.exists(args.triples):
        if args.verbose:
            print(f"Loading triples from {args.triples}...")
        gl.load_triples(args.triples)
        gl.build_embeddings()
    else:
        print(f"Warning: Triples file not found: {args.triples}")
    
    # Run evaluation or single question
    if args.evaluate:
        # Evaluation mode
        if not os.path.exists(args.evaluate):
            print(f"Error: QA dataset not found: {args.evaluate}")
            sys.exit(1)
        
        with open(args.evaluate, "r") as f:
            qa_dataset = json.load(f)
        
        print(f"Evaluating on {len(qa_dataset)} questions...")
        
        metrics = gl.evaluate(
            qa_dataset,
            use_multihop=not args.no_multihop,
            use_verification=not args.no_verification,
        )
        
        print("\nEvaluation Results:")
        print(f"  Accuracy: {metrics['accuracy']:.2%}")
        print(f"  Correct: {metrics['correct']}/{metrics['total']}")
        
        if args.output:
            with open(args.output, "w") as f:
                json.dump(metrics, f, indent=2)
            print(f"\nResults saved to {args.output}")
    
    else:
        # Single question mode
        options = args.options.split(",") if args.options else None
        
        if args.verbose:
            print(f"\nQuestion: {args.question}")
            if args.image:
                print(f"Image: {args.image}")
            if options:
                print(f"Options: {options}")
        
        result = gl.answer(
            question=args.question,
            image=args.image,
            options=options,
            use_multihop=not args.no_multihop,
            use_verification=not args.no_verification,
            extract_from_image=args.extract_from_image,
        )
        
        print(f"\nAnswer: {result['answer']}")
        
        # Evidence is always returned
        if result.get("evidence"):
            print(f"\nEvidence ({len(result['evidence'])} triples):")
            for e in result["evidence"][:10]:
                print(f"  - {e['subject']} {e['relation']} {e['object']}")
        
        if args.verbose and result.get("metadata"):
            print(f"\nMetadata: {result['metadata']}")
        
        if args.output:
            with open(args.output, "w") as f:
                json.dump(result, f, indent=2, default=str)
            print(f"\nResult saved to {args.output}")


if __name__ == "__main__":
    main()
