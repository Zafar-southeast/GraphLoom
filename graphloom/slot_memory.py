"""
GraphLoom Slot Memory Module
============================
HieraSlot memories and reliability-calibrated routing.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field


@dataclass
class SlotMemory:
    """
    A single slot memory unit.
    
    Each triple produces two slots:
    - Entity slot: key=Emb(subject), value=Emb(relation + object)
    - Relation slot: key=Emb(relation), value=Emb(subject + object)
    """
    key: np.ndarray
    value: np.ndarray
    slot_type: str  # "entity" or "relation"
    triple_idx: int
    confidence: float
    reliability_score: float = 0.5
    
    def to_dict(self) -> Dict:
        return {
            "slot_type": self.slot_type,
            "triple_idx": self.triple_idx,
            "confidence": self.confidence,
            "reliability_score": self.reliability_score,
        }


class HieraSlotMemory:
    """
    Hierarchical Slot Memory for GraphLoom.
    
    Converts retrieved triples into structured key-value slot memories
    with entity and relation slots.
    """
    
    def __init__(self, embedding_dim: int = 2048):
        """
        Initialize HieraSlot memory.
        
        Args:
            embedding_dim: Dimension of embeddings
        """
        self.embedding_dim = embedding_dim
        self.slots: List[SlotMemory] = []
        self.entity_slots: List[SlotMemory] = []
        self.relation_slots: List[SlotMemory] = []
    
    def build_from_triples(
        self,
        triples: List["Triple"],
        embedder,
        batch_size: int = 32,
    ):
        """
        Build slot memories from triples.
        
        Args:
            triples: List of Triple objects
            embedder: Embedding model with encode_text method
            batch_size: Batch size for encoding
        """
        if not triples:
            return
        
        # Prepare texts for encoding
        subjects = [t.subject for t in triples]
        relations = [t.relation for t in triples]
        objects = [t.object for t in triples]
        rel_obj = [f"{t.relation} {t.object}" for t in triples]
        subj_obj = [f"{t.subject} {t.object}" for t in triples]
        
        # Encode all texts
        all_texts = subjects + relations + objects + rel_obj + subj_obj
        all_embeddings = embedder.encode_text(all_texts, batch_size=batch_size)
        
        n = len(triples)
        subj_emb = all_embeddings[:n]
        rel_emb = all_embeddings[n:2*n]
        # obj_emb = all_embeddings[2*n:3*n]  # Not used directly
        rel_obj_emb = all_embeddings[3*n:4*n]
        subj_obj_emb = all_embeddings[4*n:5*n]
        
        # Create slots
        self.slots = []
        self.entity_slots = []
        self.relation_slots = []
        
        for i, triple in enumerate(triples):
            # Entity slot: key=subject, value=relation+object
            entity_slot = SlotMemory(
                key=subj_emb[i],
                value=rel_obj_emb[i],
                slot_type="entity",
                triple_idx=i,
                confidence=triple.confidence,
            )
            self.entity_slots.append(entity_slot)
            self.slots.append(entity_slot)
            
            # Relation slot: key=relation, value=subject+object
            relation_slot = SlotMemory(
                key=rel_emb[i],
                value=subj_obj_emb[i],
                slot_type="relation",
                triple_idx=i,
                confidence=triple.confidence,
            )
            self.relation_slots.append(relation_slot)
            self.slots.append(relation_slot)
    
    def get_keys(self) -> np.ndarray:
        """Get all slot keys as matrix."""
        if not self.slots:
            return np.array([])
        return np.vstack([s.key for s in self.slots])
    
    def get_values(self) -> np.ndarray:
        """Get all slot values as matrix."""
        if not self.slots:
            return np.array([])
        return np.vstack([s.value for s in self.slots])
    
    def __len__(self) -> int:
        return len(self.slots)


class ReliabilityCalibratedRouter:
    """
    Reliability-calibrated slot router for GraphLoom.
    
    Routes queries to the most relevant and reliable slots
    using Top-k selection with reliability bias.
    """
    
    def __init__(
        self,
        top_k: int = 4,
        activation_threshold: float = 0.3,
        reliability_bias: float = 2.0,
        gate_bias: float = -1.0,
    ):
        """
        Initialize the router.
        
        Args:
            top_k: Number of slots to activate (k)
            activation_threshold: Minimum gate probability (epsilon)
            reliability_bias: Weight for reliability score (beta)
            gate_bias: Base gate bias (b_g)
        """
        self.top_k = top_k
        self.activation_threshold = activation_threshold
        self.reliability_bias = reliability_bias
        self.gate_bias = gate_bias
    
    def compute_slot_scores(
        self,
        query: np.ndarray,
        slot_memory: HieraSlotMemory,
        visual_context: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Compute utility scores for all slots.
        
        Args:
            query: Query embedding
            slot_memory: HieraSlot memory
            visual_context: Optional visual context embedding
        
        Returns:
            Array of slot scores
        """
        if len(slot_memory) == 0:
            return np.array([])
        
        keys = slot_memory.get_keys()
        
        # Normalize query
        query_norm = query / (np.linalg.norm(query) + 1e-8)
        
        # Semantic alignment: query @ keys
        semantic_scores = keys @ query_norm
        
        # Visual context contribution (if available)
        if visual_context is not None:
            visual_norm = visual_context / (np.linalg.norm(visual_context) + 1e-8)
            visual_scores = keys @ visual_norm
        else:
            visual_scores = np.zeros_like(semantic_scores)
        
        # Reliability scores
        reliability_scores = np.array([s.reliability_score for s in slot_memory.slots])
        
        # Confidence scores
        confidence_scores = np.array([s.confidence for s in slot_memory.slots])
        
        # Combined utility score (Equation 10 from paper)
        # s_p(t) = q_t @ k_p + w_v @ v_t + phi(f_p) + beta * logit(c_p) + b_g
        logit_reliability = np.log(reliability_scores / (1 - reliability_scores + 1e-8) + 1e-8)
        
        scores = (
            semantic_scores +
            0.3 * visual_scores +
            0.2 * confidence_scores +
            self.reliability_bias * logit_reliability +
            self.gate_bias
        )
        
        return scores
    
    def route(
        self,
        query: np.ndarray,
        slot_memory: HieraSlotMemory,
        visual_context: Optional[np.ndarray] = None,
    ) -> Tuple[List[int], np.ndarray]:
        """
        Route query to top-k slots.
        
        Args:
            query: Query embedding
            slot_memory: HieraSlot memory
            visual_context: Optional visual context embedding
        
        Returns:
            Tuple of (active_slot_indices, gate_probabilities)
        """
        if len(slot_memory) == 0:
            return [], np.array([])
        
        # Compute scores
        scores = self.compute_slot_scores(query, slot_memory, visual_context)
        
        # Get top-k indices
        top_k_indices = np.argsort(scores)[::-1][:self.top_k]
        
        # Compute gate probabilities (sigmoid)
        gate_probs = 1 / (1 + np.exp(-scores))
        
        # Filter by activation threshold
        active_indices = [
            idx for idx in top_k_indices
            if gate_probs[idx] >= self.activation_threshold
        ]
        
        return active_indices, gate_probs
    
    def get_active_slots(
        self,
        query: np.ndarray,
        slot_memory: HieraSlotMemory,
        visual_context: Optional[np.ndarray] = None,
    ) -> List[SlotMemory]:
        """
        Get active slot memories for a query.
        
        Args:
            query: Query embedding
            slot_memory: HieraSlot memory
            visual_context: Optional visual context embedding
        
        Returns:
            List of active SlotMemory objects
        """
        active_indices, _ = self.route(query, slot_memory, visual_context)
        return [slot_memory.slots[i] for i in active_indices]
    
    def get_active_evidence(
        self,
        query: np.ndarray,
        slot_memory: HieraSlotMemory,
        triples: List["Triple"],
        visual_context: Optional[np.ndarray] = None,
    ) -> List["Triple"]:
        """
        Get triples corresponding to active slots.
        
        Args:
            query: Query embedding
            slot_memory: HieraSlot memory
            triples: Original triples
            visual_context: Optional visual context embedding
        
        Returns:
            List of active Triple objects
        """
        active_slots = self.get_active_slots(query, slot_memory, visual_context)
        
        # Get unique triple indices
        triple_indices = set()
        for slot in active_slots:
            triple_indices.add(slot.triple_idx)
        
        return [triples[i] for i in sorted(triple_indices) if i < len(triples)]


class EvidenceMemoryFusion:
    """
    Fuses multimodal prefix memories with routed slot memories.
    
    Creates the combined memory context for the decoder.
    """
    
    def __init__(self, prefix_length: int = 32):
        """
        Initialize the fusion module.
        
        Args:
            prefix_length: Length of the prefix memory
        """
        self.prefix_length = prefix_length
    
    def create_prefix_memory(
        self,
        question_embedding: np.ndarray,
        facts_embedding: np.ndarray,
        visual_embedding: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Create fused prefix memory.
        
        Args:
            question_embedding: Question embedding
            facts_embedding: Facts/evidence embedding
            visual_embedding: Optional visual embedding
        
        Returns:
            Fused prefix memory
        """
        # Simple fusion: concatenate and project
        if visual_embedding is not None:
            fused = np.concatenate([
                question_embedding,
                facts_embedding,
                visual_embedding,
            ])
        else:
            fused = np.concatenate([
                question_embedding,
                facts_embedding,
            ])
        
        # Normalize
        fused = fused / (np.linalg.norm(fused) + 1e-8)
        
        return fused
    
    def build_memory_context(
        self,
        prefix_memory: np.ndarray,
        active_slots: List[SlotMemory],
    ) -> Dict:
        """
        Build complete memory context for decoding.
        
        Args:
            prefix_memory: Fused prefix memory
            active_slots: List of active slot memories
        
        Returns:
            Dictionary with memory context
        """
        context = {
            "prefix": prefix_memory,
            "slot_keys": [],
            "slot_values": [],
            "slot_info": [],
        }
        
        for slot in active_slots:
            context["slot_keys"].append(slot.key)
            context["slot_values"].append(slot.value)
            context["slot_info"].append(slot.to_dict())
        
        if context["slot_keys"]:
            context["slot_keys"] = np.vstack(context["slot_keys"])
            context["slot_values"] = np.vstack(context["slot_values"])
        
        return context
