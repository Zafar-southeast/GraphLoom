"""
GraphLoom VLM Module
====================
Vision-Language Model integration using OpenRouter API (Qwen-2.5-VL).
"""

import os
import json
import re
import base64
from typing import Optional, Union, List, Dict
from PIL import Image
import requests
from io import BytesIO


class VLMClient:
    """
    Vision-Language Model client using OpenRouter API.
    
    Uses Qwen-2.5-VL-7B-Instruct for scene description and visual understanding.
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = "https://openrouter.ai/api/v1",
        model: str = "qwen/qwen-2.5-vl-7b-instruct",
        max_tokens: int = 2000,
        temperature: float = 0.7,
        site_url: str = "",
        site_name: str = "GraphLoom",
    ):
        """
        Initialize the VLM client.
        
        Args:
            api_key: OpenRouter API key (or set OPENROUTER_API_KEY env var)
            base_url: OpenRouter API base URL
            model: Model identifier
            max_tokens: Maximum tokens in response
            temperature: Sampling temperature
            site_url: Optional site URL for OpenRouter rankings
            site_name: Optional site name for OpenRouter rankings
        """
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        if not self.api_key:
            raise ValueError("OpenRouter API key required. Set OPENROUTER_API_KEY or pass api_key.")
        
        self.base_url = base_url
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.site_url = site_url
        self.site_name = site_name
        
        # Initialize OpenAI client
        from openai import OpenAI
        self.client = OpenAI(
            base_url=self.base_url,
            api_key=self.api_key,
        )
    
    def _encode_image(self, image_source: Union[str, Image.Image]) -> str:
        """Encode image to base64 data URL."""
        if isinstance(image_source, str):
            if image_source.startswith(("http://", "https://")):
                # Return URL directly for remote images
                return image_source
            else:
                # Load local image
                with open(image_source, "rb") as f:
                    image_data = f.read()
        else:
            # PIL Image
            buffer = BytesIO()
            image_source.save(buffer, format="JPEG")
            image_data = buffer.getvalue()
        
        # Encode to base64
        base64_data = base64.b64encode(image_data).decode("utf-8")
        return f"data:image/jpeg;base64,{base64_data}"
    
    def _build_content(
        self,
        text: str,
        images: Optional[List[Union[str, Image.Image]]] = None,
    ) -> List[dict]:
        """Build content array for API request."""
        content = []
        
        # Add text
        if text:
            content.append({"type": "text", "text": text})
        
        # Add images
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
        Generate text response from the VLM.
        
        Args:
            prompt: Text prompt
            images: Optional list of images (paths, URLs, or PIL Images)
            system_prompt: Optional system prompt
            max_tokens: Override default max_tokens
            temperature: Override default temperature
        
        Returns:
            Generated text response
        """
        messages = []
        
        # Add system message if provided
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        
        # Build user message content
        content = self._build_content(prompt, images)
        messages.append({"role": "user", "content": content})
        
        # Build extra headers
        extra_headers = {}
        if self.site_url:
            extra_headers["HTTP-Referer"] = self.site_url
        if self.site_name:
            extra_headers["X-Title"] = self.site_name
        
        # Make API call
        completion = self.client.chat.completions.create(
            extra_headers=extra_headers if extra_headers else None,
            model=self.model,
            messages=messages,
            max_tokens=max_tokens or self.max_tokens,
            temperature=temperature or self.temperature,
        )
        
        return completion.choices[0].message.content
    
    def describe_scene(
        self,
        image: Union[str, Image.Image],
        question: Optional[str] = None,
    ) -> str:
        """
        Generate a scene description for an image.
        
        Args:
            image: Image path, URL, or PIL Image
            question: Optional question to contextualize the description
        
        Returns:
            Scene description text
        """
        system_prompt = (
            "You are an expert visual analyst creating text for downstream KG extraction. "
            "Write factual, explicit, and specific descriptions."
        )
        if question:
            prompt = f"""Analyze this image and provide a detailed scene description that would help answer the following question:

Question: {question}

Describe the image in high detail using short factual sentences (one fact per line).
Include as many concrete facts as possible, including:
1. Main entities and objects
2. Exact visible text (titles, names, years, numbers, labels)
3. Relationships between entities
4. Attributes (colors, clothing, uniforms, equipment, setting)
5. Domain facts if visible (movie title/year/genre, sport type, cast names)

If a sport is visually indicated, add an explicit sentence:
"The sport shown is <sport_name>."
If unknown, say: "The sport shown is unknown."

Avoid speculation. Prefer direct visual evidence.

Scene Description:"""
        else:
            prompt = """Describe this image in high detail using short factual sentences (one fact per line).
Include:
1. Main entities and objects
2. Exact visible text (titles, names, years, numbers, labels)
3. Relationships between entities
4. Attributes (colors, clothing, uniforms, equipment, setting)
5. Domain facts if visible (movie title/year/genre, sport type, cast names)

If a sport is visually indicated, add an explicit sentence:
"The sport shown is <sport_name>."
If unknown, say: "The sport shown is unknown."

Avoid speculation. Prefer direct visual evidence.

Scene Description:"""
        
        return self.generate(
            prompt,
            images=[image],
            system_prompt=system_prompt,
            temperature=0.1,
            max_tokens=max(self.max_tokens, 2000),
        )

    def _parse_triples_json(self, response_text: str) -> List[Dict]:
        """Parse a JSON array of triples from model text."""
        text = response_text.strip()
        if not text:
            return []

        # Handle fenced JSON blocks if model adds markdown formatting.
        fence_match = re.search(r"```(?:json)?\s*(.*?)\s*```", text, flags=re.DOTALL | re.IGNORECASE)
        if fence_match:
            text = fence_match.group(1).strip()

        data = None
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            # Fallback: try to isolate the first JSON array.
            start = text.find("[")
            end = text.rfind("]")
            if start != -1 and end != -1 and end > start:
                try:
                    data = json.loads(text[start : end + 1])
                except json.JSONDecodeError:
                    return []
            else:
                return []

        if not isinstance(data, list):
            return []

        triples = []
        for item in data:
            if not isinstance(item, dict):
                continue
            subject = str(item.get("subject", "")).strip()
            relation = str(item.get("relation", "")).strip()
            obj = str(item.get("object", "")).strip()
            if not (subject and relation and obj):
                continue
            confidence = item.get("confidence", 0.9)
            try:
                confidence = float(confidence)
            except (TypeError, ValueError):
                confidence = 0.9
            triples.append(
                {
                    "subject": subject,
                    "relation": relation,
                    "object": obj,
                    "confidence": max(0.0, min(1.0, confidence)),
                }
            )
        return triples

    def extract_structured_triples(
        self,
        image: Union[str, Image.Image],
        question: Optional[str] = None,
        max_triples: int = 20,
    ) -> List[Dict]:
        """
        Extract KG-ready triples directly from image using the VLM.

        Returns:
            List[{"subject": str, "relation": str, "object": str, "confidence": float}]
        """
        context_line = f"Question context: {question}" if question else "Question context: none"
        prompt = f"""You are building a knowledge graph from an image.
{context_line}

Extract up to {max_triples} factual triples from the image.
Focus on explicit entities and visual/textual facts.
If this looks like a sports or movie poster, include title, year, sport, cast, and genre when visible.

Return ONLY a JSON array (no markdown, no explanation) using this schema:
[
  {{"subject": "string", "relation": "string", "object": "string", "confidence": 0.0}}
]
"""
        response = self.generate(prompt, images=[image], temperature=0.1, max_tokens=900)
        return self._parse_triples_json(response)
    
    def extract_visual_relations(
        self,
        image: Union[str, Image.Image],
    ) -> str:
        """
        Extract visual relations from an image as structured triples.
        
        Args:
            image: Image path, URL, or PIL Image
        
        Returns:
            Text containing extracted relations
        """
        prompt = """Analyze this image and extract all visual relationships as structured triples.

For each relationship, output in the format: (subject, relation, object)

Examples:
- (dog, sitting_on, grass)
- (person, holding, umbrella)
- (car, parked_near, building)

Extract all visible relationships:"""
        
        return self.generate(prompt, images=[image], temperature=0.2)
    
    def answer_question(
        self,
        question: str,
        image: Union[str, Image.Image],
        context: Optional[str] = None,
    ) -> str:
        """
        Answer a question about an image with optional context.
        
        Args:
            question: Question to answer
            image: Image path, URL, or PIL Image
            context: Optional additional context (e.g., retrieved evidence)
        
        Returns:
            Answer text
        """
        if context:
            prompt = f"""Based on the image and the following context, answer the question.

Context:
{context}

Question: {question}

Answer:"""
        else:
            prompt = f"""Based on the image, answer the following question.

Question: {question}

Answer:"""
        
        return self.generate(prompt, images=[image])


def create_vlm_client(config) -> VLMClient:
    """Create a VLM client from config."""
    return VLMClient(
        api_key=config.vlm.api_key,
        base_url=config.vlm.base_url,
        model=config.vlm.model,
        max_tokens=config.vlm.max_tokens,
        temperature=config.vlm.temperature,
        site_url=config.vlm.site_url,
        site_name=config.vlm.site_name,
    )
