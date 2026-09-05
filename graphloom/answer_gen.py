"""
GraphLoom Answer Generation Module
==================================
Answer generation using Groq API (Llama-4-Scout).
"""

import os
import base64
from typing import Optional, Union, List
from PIL import Image
from io import BytesIO


class AnswerGenerator:
    """
    Answer generation client using Groq API.
    
    Uses Llama-4-Scout-17B for multimodal answer generation.
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "meta-llama/llama-4-scout-17b-16e-instruct",
        max_tokens: int = 1024,
        temperature: float = 0.7,
        top_p: float = 0.9,
    ):
        """
        Initialize the answer generator.
        
        Args:
            api_key: Groq API key (or set GROQ_API_KEY env var)
            model: Model identifier
            max_tokens: Maximum tokens in response
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
        """
        self.api_key = api_key or os.getenv("GROQ_API_KEY")
        if not self.api_key:
            raise ValueError("Groq API key required. Set GROQ_API_KEY or pass api_key.")
        
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        
        # Initialize Groq client
        from groq import Groq
        self.client = Groq(api_key=self.api_key)
    
    def _encode_image(self, image_source: Union[str, Image.Image]) -> str:
        """Encode image to base64 data URL or return URL."""
        if isinstance(image_source, str):
            if image_source.startswith(("http://", "https://")):
                return image_source
            else:
                with open(image_source, "rb") as f:
                    image_data = f.read()
        else:
            buffer = BytesIO()
            image_source.save(buffer, format="JPEG")
            image_data = buffer.getvalue()
        
        base64_data = base64.b64encode(image_data).decode("utf-8")
        return f"data:image/jpeg;base64,{base64_data}"
    
    def _build_content(
        self,
        text: str,
        images: Optional[List[Union[str, Image.Image]]] = None,
    ) -> List[dict]:
        """Build content array for API request."""
        content = []
        
        if text:
            content.append({"type": "text", "text": text})
        
        if images:
            for img in images:
                image_url = self._encode_image(img)
                content.append({
                    "type": "image_url",
                    "image_url": {"url": image_url}
                })
        
        return content
    
    def generate(
        self,
        prompt: str,
        images: Optional[List[Union[str, Image.Image]]] = None,
        system_prompt: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> str:
        """
        Generate text response.
        
        Args:
            prompt: Text prompt
            images: Optional list of images
            system_prompt: Optional system prompt
            max_tokens: Override default max_tokens
            temperature: Override default temperature
        
        Returns:
            Generated text response
        """
        messages = []
        
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        
        content = self._build_content(prompt, images)
        messages.append({"role": "user", "content": content})
        
        completion = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_completion_tokens=max_tokens or self.max_tokens,
            temperature=temperature or self.temperature,
            top_p=self.top_p,
            stream=False,
        )
        
        return completion.choices[0].message.content
    
    def answer_with_evidence(
        self,
        question: str,
        evidence: List[str],
        image: Optional[Union[str, Image.Image]] = None,
        answer_type: str = "open",
        options: Optional[List[str]] = None,
    ) -> str:
        """
        Generate an answer using retrieved evidence.
        
        Args:
            question: Question to answer
            evidence: List of evidence triples/facts
            image: Optional image for multimodal QA
            answer_type: "open" for open-ended, "mcq" for multiple choice
            options: Answer options for MCQ
        
        Returns:
            Generated answer
        """
        evidence_text = "\n".join(f"- {e}" for e in evidence)
        
        if answer_type == "mcq" and options:
            options_text = "\n".join(f"{chr(65+i)}. {opt}" for i, opt in enumerate(options))
            prompt = f"""Based on the image and the following evidence, answer the multiple-choice question.

Evidence:
{evidence_text}

Question: {question}

Options:
{options_text}

Select the best answer and explain your reasoning briefly.

Answer:"""
        else:
            prompt = f"""Based on the image and the following evidence, answer the question accurately and concisely.

Evidence:
{evidence_text}

Question: {question}

Provide a clear, evidence-grounded answer. If the evidence is insufficient, indicate what information is missing.

Answer:"""
        
        images = [image] if image else None
        return self.generate(prompt, images=images, temperature=0.3)
    
    def generate_hypothesis(
        self,
        question: str,
        current_evidence: List[str],
        image: Optional[Union[str, Image.Image]] = None,
    ) -> str:
        """
        Generate a bridging hypothesis for multi-hop retrieval.
        
        Args:
            question: Original question
            current_evidence: Currently retrieved evidence
            image: Optional image
        
        Returns:
            Intermediate question/hypothesis for additional retrieval
        """
        evidence_text = "\n".join(f"- {e}" for e in current_evidence[:5])
        
        prompt = f"""Given the question and current evidence, generate a concise intermediate question that would help find missing information.

Original Question: {question}

Current Evidence:
{evidence_text}

What additional information is needed? Generate a short, focused follow-up question (max 15 words):"""
        
        images = [image] if image else None
        response = self.generate(prompt, images=images, temperature=0.5, max_tokens=50)
        
        # Truncate to ~15 tokens
        words = response.split()[:15]
        return " ".join(words)
    
    def revise_answer(
        self,
        original_answer: str,
        unsupported_claims: List[str],
        supporting_evidence: List[str],
        image: Optional[Union[str, Image.Image]] = None,
    ) -> str:
        """
        Revise an answer to correct unsupported claims.
        
        Args:
            original_answer: The original generated answer
            unsupported_claims: Claims that lack evidence support
            supporting_evidence: Evidence to use for revision
            image: Optional image
        
        Returns:
            Revised answer
        """
        claims_text = "\n".join(f"- {c}" for c in unsupported_claims)
        evidence_text = "\n".join(f"- {e}" for e in supporting_evidence)
        
        prompt = f"""Revise the following answer to correct unsupported claims while preserving accurate information.

Original Answer:
{original_answer}

Unsupported Claims (need correction):
{claims_text}

Supporting Evidence:
{evidence_text}

Provide a revised answer that:
1. Removes or corrects unsupported claims
2. Uses the supporting evidence accurately
3. Maintains the original structure where possible

Revised Answer:"""
        
        images = [image] if image else None
        return self.generate(prompt, images=images, temperature=0.2)


def create_answer_generator(config) -> AnswerGenerator:
    """Create an answer generator from config."""
    return AnswerGenerator(
        api_key=config.answer_gen.api_key,
        model=config.answer_gen.model,
        max_tokens=config.answer_gen.max_tokens,
        temperature=config.answer_gen.temperature,
        top_p=config.answer_gen.top_p,
    )
