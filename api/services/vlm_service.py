"""
GLM-4.6V Vision Language Model service for breed verification.

This module provides VLM-based verification of CNN predictions. When your CNN
classifier returns top-3 predictions for a cat image, the VLM analyzes the
image and either agrees with the top prediction or suggests a correction.

Why VLM verification?
- CNNs can be confident but wrong (especially for similar breeds)
- VLMs can reason about visual features like coat patterns, face shapes
- Combined approach catches more edge cases than CNN alone
"""

import base64
import logging
import os
import threading
from pathlib import Path
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)

# Thread lock for singleton initialization (thread-safe for FastAPI)
_instance_lock = threading.Lock()


class VLMService:
    """
    Service for GLM-4.6V breed verification using base64 image encoding.

    This uses the singleton pattern because:
    - API client initialization has overhead (loading configs, auth)
    - We want to reuse the same connection across requests
    - Thread-safe for FastAPI's async handling
    """

    _instance: Optional["VLMService"] = None

    def __init__(self):
        """
        Initialize VLM service with Z.ai API client.

        Raises:
            ValueError: If ZAI_API_KEY environment variable is not set
            ImportError: If zai-sdk is not installed
        """
        # Get API key from environment (never hardcode secrets!)
        self.api_key = os.getenv("ZAI_API_KEY")
        if not self.api_key:
            raise ValueError(
                "ZAI_API_KEY environment variable not set. "
                "Get your key from https://docs.z.ai and add to .env"
            )

        # Import SDK here to provide clear error if not installed
        try:
            from zai import ZaiClient
            self.client = ZaiClient(api_key=self.api_key)
        except ImportError:
            raise ImportError(
                "zai-sdk not installed. Run: pip install zai-sdk>=0.0.4"
            )

        # Model to use for vision tasks
        self.model = "glm-4.6v"

    @classmethod
    def get_instance(cls) -> "VLMService":
        """
        Get singleton instance of VLMService (thread-safe).

        Why singleton? We want to reuse the API client across requests
        rather than creating a new one each time (which has overhead).

        Why thread-safe? FastAPI handles concurrent requests, so multiple
        threads could try to create the instance simultaneously.

        Returns:
            The shared VLMService instance
        """
        # Double-check locking pattern for thread safety
        if cls._instance is None:
            with _instance_lock:
                # Check again inside lock (another thread may have created it)
                if cls._instance is None:
                    cls._instance = VLMService()
        return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        """Reset singleton instance. Useful for testing."""
        cls._instance = None

    def _encode_image_to_base64(self, image_path: str) -> str:
        """
        Encode image file to base64 string.

        Why base64? The Z.ai API accepts images as base64 data URIs,
        which is simpler than file upload (no extra API call).

        Args:
            image_path: Path to the image file (JPG, PNG, etc.)

        Returns:
            Base64 encoded string of the image

        Raises:
            FileNotFoundError: If image file doesn't exist
        """
        path = Path(image_path)
        if not path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")

    def _get_image_mime_type(self, image_path: str) -> str:
        """
        Determine MIME type from file extension.

        The API needs to know the image format (JPEG vs PNG etc.)
        to properly decode the base64 data.

        Args:
            image_path: Path to image file

        Returns:
            MIME type string like "image/jpeg" or "image/png"
        """
        extension = Path(image_path).suffix.lower()
        mime_types = {
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".png": "image/png",
            ".gif": "image/gif",
            ".webp": "image/webp",
        }
        # Default to JPEG if unknown (most common for photos)
        return mime_types.get(extension, "image/jpeg")

    def _build_prompt(self, cnn_top_3: List[Tuple[str, float]]) -> str:
        """
        Build verification prompt with CNN top-3 candidates.

        Why structured prompt?
        - Guides the VLM to focus on specific task (breed ID)
        - Lists candidates so VLM doesn't need to know all 67 breeds
        - Enforces consistent response format for easier parsing

        Args:
            cnn_top_3: List of (breed_name, confidence) tuples from CNN

        Returns:
            Formatted prompt string
        """
        # Format candidates with ranking and confidence percentage
        candidates = "\n".join([
            f"  {i+1}. {breed} ({conf:.1%})"
            for i, (breed, conf) in enumerate(cnn_top_3)
        ])

        return f"""Analyze this cat image and identify its breed.

The CNN classifier's top-3 predictions are:
{candidates}

Instructions:
1. Look at the cat's features: coat pattern, face shape, eye color, body type, ear shape
2. Compare with the 3 candidates above
3. Choose the most likely breed from the candidates, OR suggest a different breed if none match

Response format:
BREED: [your prediction]
MATCHES_CNN: [YES if your choice is in top-3, NO if different]
REASON: [1-2 sentence explanation of key visual features]"""

    def verify_prediction(
        self,
        image_path: str,
        cnn_top_3: List[Tuple[str, float]]
    ) -> Tuple[str, str, Optional[str]]:
        """
        Verify CNN prediction using GLM-4.6V.

        This is the main entry point for VLM verification. Given an image
        and CNN's top-3 predictions, it returns whether VLM agrees.

        Args:
            image_path: Path to cat image file
            cnn_top_3: List of (breed_name, confidence) tuples from CNN

        Returns:
            Tuple of (status, vlm_prediction, reasoning)
            - status: "agree" (VLM picks same as CNN #1),
                     "disagree" (VLM picks different breed),
                     "error" (API call failed)
            - vlm_prediction: The breed VLM thinks it is
            - reasoning: VLM's explanation of visual features

        Common pitfalls:
            - Forgetting to handle API errors (network can fail!)
            - Not normalizing breed names (case sensitivity)
        """
        try:
            # Encode image to base64 (no file upload needed)
            base64_image = self._encode_image_to_base64(image_path)
            mime_type = self._get_image_mime_type(image_path)

            # Build data URI format: data:image/jpeg;base64,<data>
            image_uri = f"data:{mime_type};base64,{base64_image}"

            # Build focused prompt with top-3 candidates
            prompt = self._build_prompt(cnn_top_3)

            # Call GLM-4.6V with image and prompt
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{
                    "role": "user",
                    "content": [
                        # Image comes first, then the text prompt
                        {"type": "image_url", "image_url": {"url": image_uri}},
                        {"type": "text", "text": prompt}
                    ]
                }],
                temperature=0.3,  # Lower = more deterministic
                max_tokens=200    # Breed + reason doesn't need many tokens
            )

            # Extract response text
            content = response.choices[0].message.content
            logger.debug(f"VLM response: {content}")

            # Parse structured response
            return self._parse_response(content, cnn_top_3)

        except FileNotFoundError as e:
            logger.error(f"Image file not found: {e}")
            return ("error", cnn_top_3[0][0], str(e))
        except Exception as e:
            # Log error but don't crash - return CNN prediction as fallback
            logger.error(f"VLM verification failed: {e}")
            return ("error", cnn_top_3[0][0], str(e))

    def _parse_response(
        self,
        content: str,
        cnn_top_3: List[Tuple[str, float]]
    ) -> Tuple[str, str, Optional[str]]:
        """
        Parse VLM response into structured output.

        The response format we expect:
            BREED: Persian
            MATCHES_CNN: YES
            REASON: The flat face and long fur are characteristic of Persian cats.

        Args:
            content: Raw response text from VLM
            cnn_top_3: Original CNN predictions for comparison

        Returns:
            Tuple of (status, vlm_prediction, reasoning)
        """
        cnn_breeds = [breed for breed, _ in cnn_top_3]
        cnn_top_1 = cnn_breeds[0]

        # Extract breed from response
        vlm_prediction = None
        if "BREED:" in content:
            # Split on BREED: and take first line after it
            breed_line = content.split("BREED:")[-1].split("\n")[0].strip()
            vlm_prediction = breed_line

        # Extract reasoning
        reasoning = None
        if "REASON:" in content:
            reasoning = content.split("REASON:")[-1].strip()

        # If we couldn't parse a breed, return unclear status
        if not vlm_prediction:
            logger.warning(f"Could not parse breed from VLM response: {content}")
            return ("unclear", cnn_top_1, reasoning)

        # Check if VLM prediction matches any CNN top-3
        # Normalize both strings for comparison (lowercase, strip whitespace)
        vlm_normalized = vlm_prediction.lower().strip()

        for i, cnn_breed in enumerate(cnn_breeds):
            cnn_normalized = cnn_breed.lower().strip()

            # Exact match after normalization (preferred)
            if vlm_normalized == cnn_normalized:
                status = "agree" if i == 0 else "disagree"
                return (status, cnn_breed, reasoning)

            # Partial match: VLM might say "Persian Cat" when CNN has "Persian"
            # Only match if the VLM response starts with or contains the breed name
            # as a complete word (avoid "Persian" matching "Persian Longhair")
            vlm_words = vlm_normalized.split()
            cnn_words = cnn_normalized.split()

            # Check if all CNN words appear in VLM response in order
            if all(word in vlm_words for word in cnn_words):
                status = "agree" if i == 0 else "disagree"
                return (status, cnn_breed, reasoning)

        # VLM picked something not in CNN top-3
        return ("disagree", vlm_prediction, reasoning)
