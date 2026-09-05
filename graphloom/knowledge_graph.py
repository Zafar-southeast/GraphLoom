"""
GraphLoom Knowledge Graph Module
================================
Multimodal Knowledge Graph construction and retrieval.
"""

import json
import numpy as np
import networkx as nx
from typing import List, Dict, Tuple, Optional, Set, Union
from dataclasses import dataclass, field
from collections import defaultdict


@dataclass
class Triple:
    """A knowledge graph triple (subject, relation, object)."""
    subject: str
    relation: str
    object: str
    confidence: float = 1.0
    source: str = "extracted"  # "extracted", "vg150", "conceptnet", "user"
    metadata: Dict = field(default_factory=dict)
    
    def to_text(self) -> str:
        """Convert triple to natural language text."""
        return f"{self.subject} {self.relation} {self.object}"
    
    def to_tuple(self) -> Tuple[str, str, str]:
        """Convert to tuple format."""
        return (self.subject, self.relation, self.object)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "subject": self.subject,
            "relation": self.relation,
            "object": self.object,
            "confidence": self.confidence,
            "source": self.source,
            "metadata": self.metadata,
        }
    
    @classmethod
    def from_dict(cls, d: Dict) -> "Triple":
        """Create from dictionary."""
        return cls(
            subject=d["subject"],
            relation=d["relation"],
            object=d["object"],
            confidence=d.get("confidence", 1.0),
            source=d.get("source", "extracted"),
            metadata=d.get("metadata", {}),
        )


class TripleExtractor:
    """
    Extract triples from text using REBEL model.
    """
    
    def __init__(self, model_name: str = "Babelscape/rebel-large", device: Optional[str] = None):
        """
        Initialize the triple extractor.
        
        Args:
            model_name: REBEL model name
            device: Device to use
        """
        self.model_name = model_name
        self.device = device
        self.model = None
        self.tokenizer = None
        self._loaded = False
    
    def load(self):
        """Load the REBEL model."""
        if self._loaded:
            return
        
        import torch
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        
        print(f"Loading REBEL model: {self.model_name}...")
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)
        
        if self.device:
            self.model = self.model.to(self.device)
        elif torch.cuda.is_available():
            self.model = self.model.to("cuda")
        
        self.model.eval()
        self._loaded = True
        print("REBEL model loaded!")
    
    def extract(self, text: str, confidence_threshold: float = 0.5) -> List[Triple]:
        """
        Extract triples from text.
        
        Args:
            text: Input text
            confidence_threshold: Minimum confidence for triples
        
        Returns:
            List of extracted triples
        """
        self.load()
        
        import torch
        
        # Tokenize
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            max_length=512,
            truncation=True,
            padding=True,
        )
        
        if self.device:
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
        elif torch.cuda.is_available():
            inputs = {k: v.to("cuda") for k, v in inputs.items()}
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=256,
                num_beams=5,
                num_return_sequences=1,
            )
        
        # Decode
        decoded = self.tokenizer.decode(outputs[0], skip_special_tokens=False)
        
        # Parse triples from REBEL output
        triples = self._parse_rebel_output(decoded, confidence_threshold)
        
        return triples
    
    def _parse_rebel_output(self, text: str, confidence_threshold: float) -> List[Triple]:
        """Parse REBEL model output to extract triples."""
        # REBEL linearization format is:
        # <triplet> SUBJECT <subj> OBJECT <obj> RELATION
        # In practice, REBEL may emit repeated <subj>/<obj> blocks for the same subject
        # without repeating <triplet>. Handle both variants robustly.
        triples = []
        cleaned = (
            text.replace("<s>", " ")
            .replace("</s>", " ")
            .replace("<pad>", " ")
            .strip()
        )
        tokens = cleaned.split()

        current = {"subject": [], "object": [], "relation": []}
        section = None

        def flush_current():
            subject = " ".join(current["subject"]).strip()
            obj = " ".join(current["object"]).strip()
            relation = " ".join(current["relation"]).strip().replace("_", " ")
            if subject and relation and obj:
                confidence = 0.8  # Base confidence for REBEL
                if confidence >= confidence_threshold:
                    triples.append(
                        Triple(
                            subject=subject,
                            relation=relation,
                            object=obj,
                            confidence=confidence,
                            source="extracted",
                        )
                    )

        for token in tokens:
            if token == "<triplet>":
                if any(current.values()):
                    flush_current()
                    current = {"subject": [], "object": [], "relation": []}
                section = "subject"
                continue

            if token == "<subj>":
                # Some outputs chain multiple object/relation pairs for one subject:
                # <triplet> SUBJECT <subj> OBJECT1 <obj> REL1 <subj> OBJECT2 <obj> REL2 ...
                # Flush the previous triple and keep the existing subject.
                if current["subject"] and current["object"] and current["relation"]:
                    flush_current()
                    current["object"] = []
                    current["relation"] = []
                section = "object"
                continue

            if token == "<obj>":
                section = "relation"
                continue

            if section in current:
                current[section].append(token)

        if any(current.values()):
            flush_current()

        return triples


class MultimodalKnowledgeGraph:
    """
    Multimodal Knowledge Graph for GraphLoom.
    
    Supports:
    - Triple storage and retrieval
    - Graph-based operations
    - Embedding-based similarity search
    - Subgraph extraction
    """
    
    def __init__(self):
        """Initialize the knowledge graph."""
        self.graph = nx.MultiDiGraph()
        self.triples: List[Triple] = []
        self.triple_embeddings: Optional[np.ndarray] = None
        self.entity_to_triples: Dict[str, List[int]] = defaultdict(list)
    
    def add_triple(self, triple: Triple):
        """Add a triple to the graph."""
        idx = len(self.triples)
        self.triples.append(triple)
        
        # Add to NetworkX graph
        self.graph.add_edge(
            triple.subject,
            triple.object,
            relation=triple.relation,
            confidence=triple.confidence,
            source=triple.source,
            triple_idx=idx,
        )
        
        # Update entity index
        self.entity_to_triples[triple.subject.lower()].append(idx)
        self.entity_to_triples[triple.object.lower()].append(idx)
    
    def add_triples(self, triples: List[Triple]):
        """Add multiple triples."""
        for triple in triples:
            self.add_triple(triple)
    
    def load_from_json(self, path: str):
        """
        Load triples from JSON file.
        
        Supports multiple formats:
        - List of dicts: [{"subject": ..., "relation": ..., "object": ...}, ...]
        - Dict with list value: {"key": [{"subject": ..., ...}, ...]}
        - Dict of dicts: {"id1": {"subject": ..., ...}, "id2": {...}}
        """
        with open(path, "r") as f:
            data = json.load(f)
        
        triples_to_add = []
        
        if isinstance(data, list):
            # Format: [{"subject": ..., "relation": ..., "object": ...}, ...]
            triples_to_add = data
        elif isinstance(data, dict):
            # Check if it's a dict containing a list of triples
            for key, value in data.items():
                if isinstance(value, list):
                    # Format: {"key": [{"subject": ..., ...}, ...]}
                    for item in value:
                        if isinstance(item, dict) and "subject" in item and "relation" in item and "object" in item:
                            triples_to_add.append(item)
                elif isinstance(value, dict) and "subject" in value and "relation" in value and "object" in value:
                    # Format: {"id1": {"subject": ..., ...}, "id2": {...}}
                    triples_to_add.append(value)
        
        # Add all found triples
        for item in triples_to_add:
            if isinstance(item, dict) and "subject" in item and "relation" in item and "object" in item:
                triple = Triple.from_dict(item)
                self.add_triple(triple)
    
    def save_to_json(self, path: str):
        """Save triples to JSON file."""
        data = [t.to_dict() for t in self.triples]
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
    
    def get_entities(self) -> Set[str]:
        """Get all entities in the graph."""
        return set(self.graph.nodes())
    
    def get_triples_for_entity(self, entity: str) -> List[Triple]:
        """Get all triples involving an entity."""
        entity_lower = entity.lower()
        indices = self.entity_to_triples.get(entity_lower, [])
        return [self.triples[i] for i in indices]
    
    def get_neighbors(self, entity: str, max_hops: int = 1) -> Set[str]:
        """Get neighboring entities within max_hops."""
        if entity not in self.graph:
            return set()
        
        neighbors = set()
        current = {entity}
        
        for _ in range(max_hops):
            next_level = set()
            for node in current:
                next_level.update(self.graph.successors(node))
                next_level.update(self.graph.predecessors(node))
            neighbors.update(next_level)
            current = next_level - neighbors
        
        neighbors.discard(entity)
        return neighbors
    
    def compute_embeddings(self, embedder, batch_size: int = 32):
        """
        Compute embeddings for all triples.
        
        Args:
            embedder: Embedding model with encode_text method
            batch_size: Batch size for encoding
        """
        if not self.triples:
            return
        
        texts = [t.to_text() for t in self.triples]
        self.triple_embeddings = embedder.encode_text(texts, batch_size=batch_size)
    
    def retrieve_by_similarity(
        self,
        query_embedding: np.ndarray,
        top_k: int = 15,
        confidence_threshold: float = 0.0,
    ) -> List[Tuple[Triple, float]]:
        """
        Retrieve triples by embedding similarity.
        
        Args:
            query_embedding: Query embedding vector
            top_k: Number of triples to retrieve
            confidence_threshold: Minimum triple confidence
        
        Returns:
            List of (triple, similarity_score) tuples
        """
        if self.triple_embeddings is None:
            raise ValueError("Call compute_embeddings first")
        
        # Normalize
        query_norm = query_embedding / (np.linalg.norm(query_embedding) + 1e-8)
        
        # Compute similarities
        similarities = self.triple_embeddings @ query_norm
        
        # Filter by confidence and get top-k
        results = []
        for idx in np.argsort(similarities)[::-1]:
            triple = self.triples[idx]
            if triple.confidence >= confidence_threshold:
                results.append((triple, float(similarities[idx])))
                if len(results) >= top_k:
                    break
        
        return results
    
    def extract_subgraph(
        self,
        seed_triples: List[Triple],
        expansion_hops: int = 2,
        degree_cap: int = 5,
        edge_budget: int = 50,
    ) -> "MultimodalKnowledgeGraph":
        """
        Extract a bounded subgraph around seed triples.
        
        Args:
            seed_triples: Initial seed triples
            expansion_hops: Number of expansion hops
            degree_cap: Maximum edges per node during expansion
            edge_budget: Maximum total edges in subgraph
        
        Returns:
            Subgraph as new MultimodalKnowledgeGraph
        """
        subgraph = MultimodalKnowledgeGraph()
        
        # Add seed triples
        included_indices = set()
        for triple in seed_triples:
            if triple in self.triples:
                idx = self.triples.index(triple)
                included_indices.add(idx)
                subgraph.add_triple(triple)
        
        if len(subgraph.triples) >= edge_budget:
            return subgraph
        
        # Get seed entities
        frontier = set()
        for triple in seed_triples:
            frontier.add(triple.subject)
            frontier.add(triple.object)
        
        # Bounded BFS expansion
        for _ in range(expansion_hops):
            if len(subgraph.triples) >= edge_budget:
                break
            
            next_frontier = set()
            
            for entity in frontier:
                # Get triples for this entity
                entity_triples = self.get_triples_for_entity(entity)
                
                # Sort by confidence and take top degree_cap
                entity_triples.sort(key=lambda t: t.confidence, reverse=True)
                
                added = 0
                for triple in entity_triples:
                    if added >= degree_cap:
                        break
                    if len(subgraph.triples) >= edge_budget:
                        break
                    
                    idx = self.triples.index(triple)
                    if idx not in included_indices:
                        included_indices.add(idx)
                        subgraph.add_triple(triple)
                        next_frontier.add(triple.subject)
                        next_frontier.add(triple.object)
                        added += 1
            
            frontier = next_frontier - set(subgraph.get_entities())
        
        return subgraph
    
    def to_evidence_text(self, max_triples: int = 50) -> str:
        """Convert graph to evidence text for prompting."""
        lines = []
        for triple in self.triples[:max_triples]:
            lines.append(f"- {triple.to_text()}")
        return "\n".join(lines)
    
    def __len__(self) -> int:
        return len(self.triples)
    
    def __repr__(self) -> str:
        return f"MultimodalKnowledgeGraph(triples={len(self.triples)}, entities={len(self.get_entities())})"


class SubgraphRetriever:
    """
    Retrieves evidence subgraphs from the knowledge graph.
    
    Implements the bounded retrieval algorithm from GraphLoom.
    """
    
    def __init__(
        self,
        kg: MultimodalKnowledgeGraph,
        embedder,
        seed_size: int = 15,
        expansion_hops: int = 2,
        degree_cap: int = 5,
        edge_budget: int = 50,
    ):
        """
        Initialize the retriever.
        
        Args:
            kg: Knowledge graph to retrieve from
            embedder: Embedding model
            seed_size: Number of seed triples (K)
            expansion_hops: Expansion hops (m)
            degree_cap: Max edges per node (d_max)
            edge_budget: Max total edges (N_max)
        """
        self.kg = kg
        self.embedder = embedder
        self.seed_size = seed_size
        self.expansion_hops = expansion_hops
        self.degree_cap = degree_cap
        self.edge_budget = edge_budget
    
    def retrieve(
        self,
        query: str,
        image: Optional[Union[str, "Image.Image"]] = None,
        confidence_threshold: float = 0.5,
    ) -> MultimodalKnowledgeGraph:
        """
        Retrieve evidence subgraph for a query.
        
        Args:
            query: Query text
            image: Optional query image
            confidence_threshold: Minimum triple confidence
        
        Returns:
            Evidence subgraph
        """
        # Encode query
        if image is not None:
            query_emb = self.embedder.encode([{"text": query, "image": image}])[0]
        else:
            query_emb = self.embedder.encode_text([query])[0]
        
        # Retrieve seed triples
        seed_results = self.kg.retrieve_by_similarity(
            query_emb,
            top_k=self.seed_size,
            confidence_threshold=confidence_threshold,
        )
        
        seed_triples = [t for t, _ in seed_results]
        
        # Extract bounded subgraph
        subgraph = self.kg.extract_subgraph(
            seed_triples,
            expansion_hops=self.expansion_hops,
            degree_cap=self.degree_cap,
            edge_budget=self.edge_budget,
        )
        
        return subgraph
    
    def retrieve_multihop(
        self,
        query: str,
        image: Optional[Union[str, "Image.Image"]] = None,
        hypothesis_generator=None,
        confidence_threshold: float = 0.5,
    ) -> MultimodalKnowledgeGraph:
        """
        Retrieve evidence with interleaved multi-hop retrieval.
        
        Args:
            query: Query text
            image: Optional query image
            hypothesis_generator: Function to generate bridging hypothesis
            confidence_threshold: Minimum triple confidence
        
        Returns:
            Merged evidence subgraph
        """
        # First retrieval round
        subgraph1 = self.retrieve(query, image, confidence_threshold)
        
        if hypothesis_generator is None or len(subgraph1) == 0:
            return subgraph1
        
        # Generate bridging hypothesis
        evidence = [t.to_text() for t in subgraph1.triples[:10]]
        hypothesis = hypothesis_generator(query, evidence, image)
        
        # Second retrieval round with hypothesis
        subgraph2 = self.retrieve(hypothesis, image, confidence_threshold)
        
        # Merge subgraphs
        merged = MultimodalKnowledgeGraph()
        
        # Add all triples from both subgraphs
        seen = set()
        all_triples = []
        
        for triple in subgraph1.triples + subgraph2.triples:
            key = triple.to_tuple()
            if key not in seen:
                seen.add(key)
                all_triples.append(triple)
        
        # Sort by confidence and take top edge_budget
        all_triples.sort(key=lambda t: t.confidence, reverse=True)
        
        for triple in all_triples[:self.edge_budget]:
            merged.add_triple(triple)
        
        return merged
