"""
GraphLoom Verification and Revision Module
==========================================
Evidence verification and answer revision using NLI.
"""

import os
import numpy as np
from typing import List, Dict, Tuple, Optional
import re


class ClaimExtractor:
    """
    Extracts atomic claims from generated answers.
    """
    
    def extract_claims(self, text: str) -> List[str]:
        """
        Extract atomic claims from text.
        
        Args:
            text: Generated answer text
        
        Returns:
            List of atomic claims
        """
        # Split into sentences
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        claims = []
        for sentence in sentences:
            # Split compound sentences
            parts = re.split(r'\s*(?:,\s*(?:and|but|or)\s*|\s+and\s+|\s+but\s+|\s+or\s+)', sentence)
            for part in parts:
                part = part.strip()
                if len(part) > 10:  # Filter very short fragments
                    claims.append(part)
        
        return claims


class NLIVerifier:
    """
    Natural Language Inference verifier using DeBERTa-MNLI.
    
    Checks if claims are entailed by evidence.
    """
    
    def __init__(
        self,
        model_name: str = "facebook/bart-large-mnli",
        device: Optional[str] = None,
    ):
        """
        Initialize the NLI verifier.
        
        Args:
            model_name: NLI model name (default: facebook/bart-large-mnli, public model)
            device: Device to use
        """
        self.model_name = model_name
        self.device = device
        self.model = None
        self.tokenizer = None
        self._loaded = False
    
    def load(self):
        """Load the NLI model."""
        if self._loaded:
            return
        
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        
        print(f"Loading NLI model: {self.model_name}...")
        
        # Load model (facebook/bart-large-mnli is public, no token needed)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
        
        if self.device:
            self.model = self.model.to(self.device)
        elif torch.cuda.is_available():
            self.model = self.model.to("cuda")
        
        self.model.eval()
        self._loaded = True
        print("NLI model loaded!")
    
    def unload(self):
        """Unload the NLI model to free GPU memory."""
        if not self._loaded:
            return
        
        import gc
        import torch
        
        print("Unloading NLI model to free GPU memory...")
        
        if self.model is not None:
            del self.model
            self.model = None
        
        if self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None
        
        # Force garbage collection and clear CUDA cache
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        
        self._loaded = False
        print("NLI model unloaded, GPU memory freed.")
    
    def verify_claim(
        self,
        claim: str,
        evidence: str,
    ) -> Tuple[float, str]:
        """
        Verify if a claim is entailed by evidence.
        
        Args:
            claim: Claim to verify
            evidence: Evidence text
        
        Returns:
            Tuple of (entailment_score, label)
        """
        self.load()
        
        import torch
        
        # Tokenize
        inputs = self.tokenizer(
            evidence,
            claim,
            return_tensors="pt",
            max_length=512,
            truncation=True,
            padding=True,
        )
        
        if self.device:
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
        elif torch.cuda.is_available():
            inputs = {k: v.to("cuda") for k, v in inputs.items()}
        
        # Get predictions
        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1)
        
        probs = probs.cpu().numpy()[0]
        
        # DeBERTa MNLI labels: 0=contradiction, 1=neutral, 2=entailment
        labels = ["contradiction", "neutral", "entailment"]
        pred_idx = np.argmax(probs)
        
        entailment_score = float(probs[2])  # Entailment probability
        label = labels[pred_idx]
        
        return entailment_score, label
    
    def verify_claims(
        self,
        claims: List[str],
        evidence: str,
        threshold: float = 0.7,
    ) -> List[Dict]:
        """
        Verify multiple claims against evidence.
        
        Args:
            claims: List of claims
            evidence: Evidence text
            threshold: Entailment threshold
        
        Returns:
            List of verification results
        """
        results = []
        
        for claim in claims:
            score, label = self.verify_claim(claim, evidence)
            results.append({
                "claim": claim,
                "entailment_score": score,
                "label": label,
                "supported": score >= threshold,
            })
        
        return results


class RetrievalQualityEvaluator:
    """
    Evaluates the quality of retrieved evidence.
    
    Triggers corrective actions when evidence is insufficient.
    """
    
    def __init__(self, adequacy_threshold: float = 0.6):
        """
        Initialize the evaluator.
        
        Args:
            adequacy_threshold: Threshold for adequate evidence
        """
        self.adequacy_threshold = adequacy_threshold
    
    def evaluate(
        self,
        query: str,
        evidence_triples: List["Triple"],
        query_embedding: Optional[np.ndarray] = None,
        evidence_embeddings: Optional[np.ndarray] = None,
    ) -> Dict:
        """
        Evaluate retrieval quality.
        
        Args:
            query: Query text
            evidence_triples: Retrieved triples
            query_embedding: Optional query embedding
            evidence_embeddings: Optional evidence embeddings
        
        Returns:
            Evaluation result with adequacy score and recommendations
        """
        if not evidence_triples:
            return {
                "adequacy_score": 0.0,
                "is_adequate": False,
                "recommendation": "expansion_boost",
                "details": "No evidence retrieved",
            }
        
        # Compute features
        features = {}
        
        # Coverage: number of triples
        features["coverage"] = min(len(evidence_triples) / 15, 1.0)
        
        # Confidence: average triple confidence
        confidences = [t.confidence for t in evidence_triples]
        features["avg_confidence"] = np.mean(confidences)
        
        # Diversity: unique entities
        entities = set()
        for t in evidence_triples:
            entities.add(t.subject.lower())
            entities.add(t.object.lower())
        features["entity_diversity"] = min(len(entities) / 20, 1.0)
        
        # Relevance: embedding similarity (if available)
        if query_embedding is not None and evidence_embeddings is not None:
            query_norm = query_embedding / (np.linalg.norm(query_embedding) + 1e-8)
            similarities = evidence_embeddings @ query_norm
            features["avg_relevance"] = float(np.mean(similarities))
        else:
            features["avg_relevance"] = 0.5
        
        # Compute adequacy score
        adequacy_score = (
            0.3 * features["coverage"] +
            0.2 * features["avg_confidence"] +
            0.2 * features["entity_diversity"] +
            0.3 * features["avg_relevance"]
        )
        
        is_adequate = adequacy_score >= self.adequacy_threshold
        
        # Determine recommendation
        if is_adequate:
            recommendation = None
        elif features["coverage"] < 0.3:
            recommendation = "expansion_boost"
        elif features["avg_relevance"] < 0.4:
            recommendation = "query_reformulation"
        else:
            recommendation = "verbalization_repair"
        
        return {
            "adequacy_score": adequacy_score,
            "is_adequate": is_adequate,
            "recommendation": recommendation,
            "features": features,
        }


class AnswerVerifierReviser:
    """
    Verifies and revises generated answers.
    
    Implements the verification-and-revision pipeline from GraphLoom.
    """
    
    def __init__(
        self,
        nli_model: str = "facebook/bart-large-mnli",
        entailment_threshold: float = 0.7,
        device: Optional[str] = None,
    ):
        """
        Initialize the verifier-reviser.
        
        Args:
            nli_model: NLI model name (default: facebook/bart-large-mnli, public model)
            entailment_threshold: Threshold for entailment
            device: Device to use
        """
        self.claim_extractor = ClaimExtractor()
        self.nli_verifier = NLIVerifier(nli_model, device)
        self.entailment_threshold = entailment_threshold
    
    def verify_answer(
        self,
        answer: str,
        evidence: str,
    ) -> Dict:
        """
        Verify an answer against evidence.
        
        Args:
            answer: Generated answer
            evidence: Evidence text
        
        Returns:
            Verification result
        """
        # Extract claims
        claims = self.claim_extractor.extract_claims(answer)
        
        if not claims:
            return {
                "claims": [],
                "supported_claims": [],
                "unsupported_claims": [],
                "hallucination_rate": 0.0,
                "is_faithful": True,
            }
        
        # Verify each claim
        verification_results = self.nli_verifier.verify_claims(
            claims,
            evidence,
            self.entailment_threshold,
        )
        
        supported = [r for r in verification_results if r["supported"]]
        unsupported = [r for r in verification_results if not r["supported"]]
        
        hallucination_rate = len(unsupported) / len(claims) if claims else 0.0
        
        return {
            "claims": verification_results,
            "supported_claims": [r["claim"] for r in supported],
            "unsupported_claims": [r["claim"] for r in unsupported],
            "hallucination_rate": hallucination_rate,
            "is_faithful": hallucination_rate < 0.3,  # Allow some tolerance
        }
    
    def revise_answer(
        self,
        answer: str,
        verification_result: Dict,
        evidence_triples: List["Triple"],
        answer_generator,
        image=None,
    ) -> str:
        """
        Revise answer to correct unsupported claims.
        
        Args:
            answer: Original answer
            verification_result: Result from verify_answer
            evidence_triples: Evidence triples for revision
            answer_generator: AnswerGenerator instance
            image: Optional image
        
        Returns:
            Revised answer
        """
        unsupported = verification_result.get("unsupported_claims", [])
        
        if not unsupported:
            return answer
        
        # Get supporting evidence
        supporting_evidence = [t.to_text() for t in evidence_triples[:10]]
        
        # Use answer generator to revise
        revised = answer_generator.revise_answer(
            original_answer=answer,
            unsupported_claims=unsupported,
            supporting_evidence=supporting_evidence,
            image=image,
        )
        
        return revised
    
    def verify_and_revise(
        self,
        answer: str,
        evidence_triples: List["Triple"],
        answer_generator,
        image=None,
        max_revisions: int = 2,
    ) -> Tuple[str, Dict]:
        """
        Verify and revise answer iteratively.
        
        Args:
            answer: Generated answer
            evidence_triples: Evidence triples
            answer_generator: AnswerGenerator instance
            image: Optional image
            max_revisions: Maximum revision iterations
        
        Returns:
            Tuple of (final_answer, verification_history)
        """
        evidence_text = "\n".join(t.to_text() for t in evidence_triples[:20])
        
        history = []
        current_answer = answer
        
        for i in range(max_revisions):
            # Verify current answer
            verification = self.verify_answer(current_answer, evidence_text)
            history.append({
                "iteration": i,
                "answer": current_answer,
                "verification": verification,
            })
            
            # Check if faithful enough
            if verification["is_faithful"]:
                break
            
            # Revise
            current_answer = self.revise_answer(
                current_answer,
                verification,
                evidence_triples,
                answer_generator,
                image,
            )
        
        return current_answer, {"history": history}
