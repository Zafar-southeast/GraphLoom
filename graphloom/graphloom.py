"""
GraphLoom - Main Module
=======================
Multimodal KG-RAG framework for grounded question answering.

Based on the paper: "GraphLoom: Controlled Evidence Injection for
Interleaved KG-RAG in Multimodal QA"
"""

import os
import json
from typing import Optional, Union, List, Dict, Tuple
from PIL import Image
import numpy as np

from .config import GraphLoomConfig
from .embedding import Qwen3VLEmbedder, create_embedder
from .vlm import VLMClient, create_vlm_client
from .answer_gen import AnswerGenerator, create_answer_generator
from .knowledge_graph import (
    Triple,
    TripleExtractor,
    MultimodalKnowledgeGraph,
    SubgraphRetriever,
)
from .slot_memory import (
    HieraSlotMemory,
    ReliabilityCalibratedRouter,
    EvidenceMemoryFusion,
)
from .verification import (
    RetrievalQualityEvaluator,
    AnswerVerifierReviser,
)


class GraphLoom:
    """
    GraphLoom: Multimodal KG-RAG Framework.
    
    A complete pipeline for grounded multimodal question answering that:
    1. Constructs instance-level multimodal knowledge graphs
    2. Retrieves compact evidence subgraphs
    3. Uses HieraSlot memories with reliability-calibrated routing
    4. Performs verification and revision for faithful answers
    
    Example usage:
        ```python
        from graphloom import GraphLoom
        
        # Initialize with API keys
        gl = GraphLoom(
            openrouter_api_key="your-key",
            groq_api_key="your-key",
            device="cuda:0",  # GPU for local models
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
    """
    
    def __init__(
        self,
        config: Optional[GraphLoomConfig] = None,
        openrouter_api_key: Optional[str] = None,
        groq_api_key: Optional[str] = None,
        device: Optional[str] = None,
        gpu_id: Optional[int] = None,
    ):
        """
        Initialize GraphLoom.
        
        Args:
            config: GraphLoomConfig object (optional)
            openrouter_api_key: OpenRouter API key for VLM
            groq_api_key: Groq API key for answer generation
            device: Device for local models (e.g., "cuda:0", "cpu")
            gpu_id: GPU ID (alternative to device, e.g., 0, 1, 2)
        """
        # Initialize config
        self.config = config or GraphLoomConfig.from_env()
        
        # Override with provided values
        if openrouter_api_key:
            self.config.vlm.api_key = openrouter_api_key
        if groq_api_key:
            self.config.answer_gen.api_key = groq_api_key
        if device:
            self.config.set_device(device)
        elif gpu_id is not None:
            self.config.set_gpu(gpu_id)
        
        # Initialize components (lazy loading)
        self._embedder: Optional[Qwen3VLEmbedder] = None
        self._vlm: Optional[VLMClient] = None
        self._answer_gen: Optional[AnswerGenerator] = None
        self._triple_extractor: Optional[TripleExtractor] = None
        
        # Knowledge graph
        self.kg = MultimodalKnowledgeGraph()
        
        # Retriever (initialized after embedder)
        self._retriever: Optional[SubgraphRetriever] = None
        
        # Slot memory components
        self.slot_memory = HieraSlotMemory(self.config.embedding.embedding_dim)
        self.router = ReliabilityCalibratedRouter(
            top_k=self.config.router.top_k,
            activation_threshold=self.config.router.activation_threshold,
            reliability_bias=self.config.router.reliability_bias,
            gate_bias=self.config.router.gate_bias,
        )
        self.memory_fusion = EvidenceMemoryFusion()
        
        # Verification components
        self.quality_evaluator = RetrievalQualityEvaluator(
            self.config.revision.adequacy_threshold
        )
        self._verifier: Optional[AnswerVerifierReviser] = None
    
    @property
    def embedder(self) -> Qwen3VLEmbedder:
        """Get or create the embedding model."""
        if self._embedder is None:
            self._embedder = create_embedder(self.config)
        return self._embedder
    
    @property
    def vlm(self) -> VLMClient:
        """Get or create the VLM client."""
        if self._vlm is None:
            self._vlm = create_vlm_client(self.config)
        return self._vlm
    
    @property
    def answer_gen(self) -> AnswerGenerator:
        """Get or create the answer generator."""
        if self._answer_gen is None:
            self._answer_gen = create_answer_generator(self.config)
        return self._answer_gen
    
    @property
    def triple_extractor(self) -> TripleExtractor:
        """Get or create the triple extractor."""
        if self._triple_extractor is None:
            self._triple_extractor = TripleExtractor(
                model_name=self.config.kg.rebel_model,
                device=self.config.embedding.device,
            )
        return self._triple_extractor
    
    @property
    def retriever(self) -> SubgraphRetriever:
        """Get or create the subgraph retriever."""
        if self._retriever is None:
            self._retriever = SubgraphRetriever(
                kg=self.kg,
                embedder=self.embedder,
                seed_size=self.config.kg.seed_size,
                expansion_hops=self.config.kg.expansion_hops,
                degree_cap=self.config.kg.degree_cap,
                edge_budget=self.config.kg.edge_budget,
            )
        return self._retriever
    
    @property
    def verifier(self) -> AnswerVerifierReviser:
        """Get or create the answer verifier."""
        if self._verifier is None:
            self._verifier = AnswerVerifierReviser(
                nli_model=self.config.revision.nli_model,
                entailment_threshold=self.config.revision.entailment_threshold,
                device=self.config.embedding.device,
            )
        return self._verifier
    
    def load_triples(self, path: str):
        """
        Load triples from a JSON file.
        
        Args:
            path: Path to JSON file with triples
        """
        self.kg.load_from_json(path)
        print(f"Loaded {len(self.kg)} triples from {path}")
    
    def save_triples(self, path: str):
        """
        Save triples to a JSON file.
        
        Args:
            path: Output path
        """
        self.kg.save_to_json(path)
        print(f"Saved {len(self.kg)} triples to {path}")
    
    def add_triples(self, triples: List[Dict]):
        """
        Add triples to the knowledge graph.
        
        Args:
            triples: List of triple dicts with subject, relation, object keys
        """
        for t in triples:
            triple = Triple.from_dict(t)
            self.kg.add_triple(triple)
    
    def build_embeddings(self, unload_after: bool = True):
        """
        Build embeddings for all triples in the knowledge graph.
        
        Args:
            unload_after: Whether to unload embedding model after to free GPU memory
        """
        if len(self.kg) == 0:
            print("No triples to embed")
            return
        
        print(f"Computing embeddings for {len(self.kg)} triples...")
        self.kg.compute_embeddings(self.embedder)
        print("Embeddings computed!")
        
        # Unload embedding model to free GPU memory for other models
        if unload_after and self._embedder is not None:
            self._embedder.unload()
    
    def extract_triples_from_image(
        self,
        image: Union[str, Image.Image],
        question: Optional[str] = None,
    ) -> List[Triple]:
        """
        Extract triples from an image using VLM triples, augmented by REBEL.
        
        Args:
            image: Image path or PIL Image
            question: Optional question for context
        
        Returns:
            List of extracted triples
        """
        threshold = self.config.kg.extraction_confidence_threshold
        merged: List[Triple] = []
        seen = set()

        # 1) Direct structured triples from VLM (better for short factual labels
        # such as sport type like "baseball" on posters).
        vlm_triples = self.vlm.extract_structured_triples(image, question=question)
        for item in vlm_triples:
            confidence = float(item.get("confidence", 0.9))
            if confidence < threshold:
                continue
            triple = Triple(
                subject=item["subject"].strip(),
                relation=item["relation"].strip(),
                object=item["object"].strip(),
                confidence=confidence,
                source="vlm",
            )
            key = (
                triple.subject.lower(),
                triple.relation.lower(),
                triple.object.lower(),
            )
            if key in seen:
                continue
            seen.add(key)
            merged.append(triple)

        # 2) Scene description + REBEL for additional textual relation coverage.
        scene_desc = self.vlm.describe_scene(image, question)
        rebel_triples = self.triple_extractor.extract(
            scene_desc,
            confidence_threshold=threshold,
        )
        for triple in rebel_triples:
            key = (
                triple.subject.strip().lower(),
                triple.relation.strip().lower(),
                triple.object.strip().lower(),
            )
            if key in seen:
                continue
            seen.add(key)
            merged.append(triple)

        return merged
    
    def retrieve_evidence(
        self,
        question: str,
        image: Optional[Union[str, Image.Image]] = None,
        use_multihop: bool = True,
    ) -> MultimodalKnowledgeGraph:
        """
        Retrieve evidence subgraph for a question.
        
        Args:
            question: Question text
            image: Optional image
            use_multihop: Whether to use multi-hop retrieval
        
        Returns:
            Evidence subgraph
        """
        if len(self.kg) == 0:
            return MultimodalKnowledgeGraph()
        
        # Ensure embeddings are computed
        if self.kg.triple_embeddings is None:
            self.build_embeddings()
        
        if use_multihop:
            # Use multi-hop retrieval with hypothesis generation
            def hypothesis_gen(q, evidence, img):
                return self.answer_gen.generate_hypothesis(q, evidence, img)
            
            subgraph = self.retriever.retrieve_multihop(
                question,
                image,
                hypothesis_generator=hypothesis_gen,
                confidence_threshold=self.config.kg.extraction_confidence_threshold,
            )
        else:
            subgraph = self.retriever.retrieve(
                question,
                image,
                confidence_threshold=self.config.kg.extraction_confidence_threshold,
            )
        
        return subgraph
    
    def answer(
        self,
        question: str,
        image: Optional[Union[str, Image.Image]] = None,
        options: Optional[List[str]] = None,
        use_multihop: bool = True,
        use_verification: bool = True,
        extract_from_image: bool = False,
    ) -> Dict:
        """
        Answer a question using the GraphLoom pipeline.
        
        Args:
            question: Question to answer
            image: Optional image for multimodal QA
            options: Answer options for multiple choice
            use_multihop: Whether to use multi-hop retrieval
            use_verification: Whether to verify and revise answer
            extract_from_image: Whether to extract triples from image
        
        Returns:
            Dictionary with answer, evidence, and metadata
        """
        result = {
            "question": question,
            "answer": None,
            "evidence": [],
            "verification": None,
            "metadata": {},
        }
        
        # Extract triples from image if requested
        if extract_from_image and image is not None:
            extracted = self.extract_triples_from_image(image, question)
            self.kg.add_triples(extracted)
            result["metadata"]["extracted_triples"] = len(extracted)
        
        # Retrieve evidence
        evidence_subgraph = self.retrieve_evidence(question, image, use_multihop)
        evidence_triples = evidence_subgraph.triples
        
        # Evaluate retrieval quality
        quality_eval = self.quality_evaluator.evaluate(
            question,
            evidence_triples,
        )
        result["metadata"]["retrieval_quality"] = quality_eval
        
        # Handle corrective actions if needed
        if not quality_eval["is_adequate"] and quality_eval["recommendation"]:
            # Try expansion boost
            if quality_eval["recommendation"] == "expansion_boost":
                # Increase edge budget and retry
                original_budget = self.retriever.edge_budget
                self.retriever.edge_budget = min(int(original_budget * 1.2), 100)
                evidence_subgraph = self.retrieve_evidence(question, image, use_multihop)
                evidence_triples = evidence_subgraph.triples
                self.retriever.edge_budget = original_budget
        
        # Build slot memories
        if evidence_triples:
            self.slot_memory.build_from_triples(evidence_triples, self.embedder)
        
        # Get query embedding for routing
        if image is not None:
            query_emb = self.embedder.encode([{"text": question, "image": image}])[0]
        else:
            query_emb = self.embedder.encode_text([question])[0]
        
        # Route to active slots
        active_triples = self.router.get_active_evidence(
            query_emb,
            self.slot_memory,
            evidence_triples,
        )
        
        # Unload embedding model to free GPU memory for NLI model
        if self._embedder is not None:
            self._embedder.unload()
        
        # Prepare evidence for answer generation
        evidence_texts = [t.to_text() for t in (active_triples or evidence_triples[:10])]
        
        # Generate answer
        answer_type = "mcq" if options else "open"
        draft_answer = self.answer_gen.answer_with_evidence(
            question=question,
            evidence=evidence_texts,
            image=image,
            answer_type=answer_type,
            options=options,
        )
        
        result["answer"] = draft_answer
        
        # Verify and revise if requested
        if use_verification and evidence_triples:
            final_answer, verification_history = self.verifier.verify_and_revise(
                draft_answer,
                evidence_triples,
                self.answer_gen,
                image,
            )
            result["answer"] = final_answer
            result["verification"] = verification_history
            
            # Unload NLI model after verification to free GPU memory
            if self._verifier is not None:
                self._verifier.nli_verifier.unload()
        
        # Always include evidence in result
        result["evidence"] = [t.to_dict() for t in evidence_triples]
        
        return result
    
    def unload_all_models(self):
        """Unload all models to free GPU memory."""
        if self._embedder is not None:
            self._embedder.unload()
        if self._verifier is not None:
            self._verifier.nli_verifier.unload()
        
        import gc
        import torch
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print("All models unloaded, GPU memory freed.")
    
    def batch_answer(
        self,
        questions: List[Dict],
        **kwargs,
    ) -> List[Dict]:
        """
        Answer multiple questions.
        
        Args:
            questions: List of dicts with "question" and optional "image"
                       Supports format: {"question": str, "image": str, "id": str, "ground_truth": str}
            **kwargs: Additional arguments passed to answer()
        
        Returns:
            List of answer results with evidence
        """
        results = []
        
        from tqdm import tqdm
        for q in tqdm(questions, desc="Answering questions"):
            result = self.answer(
                question=q["question"],
                image=q.get("image"),
                **kwargs,
            )
            # Include original item id if present
            if "id" in q:
                result["id"] = q["id"]
            if "ground_truth" in q:
                result["ground_truth"] = q["ground_truth"]
            results.append(result)
        
        return results
    
    def evaluate(
        self,
        qa_dataset: List[Dict],
        **kwargs,
    ) -> Dict:
        """
        Evaluate on a QA dataset.
        
        Args:
            qa_dataset: List of dicts with format:
                        {"question": str, "ground_truth": str, "image": str, "id": str}
            **kwargs: Additional arguments passed to answer()
        
        Returns:
            Evaluation metrics and detailed results with evidence
        """
        results = self.batch_answer(qa_dataset, **kwargs)
        
        # Compute metrics
        correct = 0
        total = len(qa_dataset)
        detailed_results = []
        
        for qa, result in zip(qa_dataset, results):
            # Support both "answer" and "ground_truth" keys
            gold = qa.get("ground_truth", qa.get("answer", "")).lower().strip()
            pred = result["answer"].lower().strip() if result["answer"] else ""
            
            # Simple exact match (can be extended)
            is_correct = gold in pred or pred in gold
            if is_correct:
                correct += 1
            
            detailed_results.append({
                "id": qa.get("id"),
                "question": qa["question"],
                "ground_truth": qa.get("ground_truth", qa.get("answer")),
                "predicted": result["answer"],
                "evidence": result["evidence"],
                "is_correct": is_correct,
            })
        
        metrics = {
            "accuracy": correct / total if total > 0 else 0,
            "total": total,
            "correct": correct,
            "results": detailed_results,
        }
        
        return metrics


def create_graphloom(
    openrouter_api_key: Optional[str] = None,
    groq_api_key: Optional[str] = None,
    device: Optional[str] = None,
    gpu_id: Optional[int] = None,
    **kwargs,
) -> GraphLoom:
    """
    Create a GraphLoom instance with simple configuration.
    
    Args:
        openrouter_api_key: OpenRouter API key
        groq_api_key: Groq API key
        device: Device for local models
        gpu_id: GPU ID (alternative to device)
        **kwargs: Additional config overrides
    
    Returns:
        Configured GraphLoom instance
    """
    return GraphLoom(
        openrouter_api_key=openrouter_api_key,
        groq_api_key=groq_api_key,
        device=device,
        gpu_id=gpu_id,
    )
