"""
Response Formatter Module
Formats JSON responses with classification, confidence, and explanation
"""
from typing import Dict, Any


def format_detection_response(
    classification: str,
    confidence: float,
    language: str = "Unknown",
    ai_probability: float = None,
    human_probability: float = None,
) -> Dict[str, Any]:
    """
    Format detection response as JSON for legacy endpoints.

    Args:
        classification: \"AI\" or \"Human\"
        confidence: Confidence score (0.0 to 1.0)
        language: Detected language (default: \"Unknown\")
        ai_probability: AI probability (optional, for explanation)
        human_probability: Human probability (optional, for explanation)

    Returns:
        Formatted response dictionary
    """
    explanation = generate_explanation(
        classification,
        confidence,
        ai_probability,
        human_probability,
    )

    return {
        "classification": classification,
        "confidence": round(confidence, 4),
        "explanation": explanation,
        "language": language,
    }


def format_detection_response_v1(
    external_classification: str,
    internal_classification: str,
    confidence: float,
    language: str = "Unknown",
    ai_probability: float = None,
    human_probability: float = None,
) -> Dict[str, Any]:
    """
    Format detection response for the v1 voice endpoint.

    external_classification must be one of:
    - \"AI_GENERATED\"
    - \"HUMAN\"
    - \"UNCERTAIN\"

    internal_classification should be \"AI\" or \"Human\" and is used
    purely for generating a detailed explanation.
    """
    explanation = generate_explanation(
        internal_classification,
        confidence,
        ai_probability,
        human_probability,
    )

    return {
        "classification": external_classification,
        "confidence": round(confidence, 4),
        "explanation": explanation,
        "language": language,
    }


def generate_explanation(
    classification: str,
    confidence: float,
    ai_probability: float = None,
    human_probability: float = None
) -> str:
    """
    Generate explanation text based on classification and confidence.
    
    Args:
        classification: "AI" or "Human"
        confidence: Confidence score (0.0 to 1.0)
        ai_probability: AI probability (optional)
        human_probability: Human probability (optional)
        
    Returns:
        Explanation string
    """
    confidence_percent = confidence * 100
    
    if classification == "AI":
        if confidence >= 0.9:
            explanation = (
                f"The audio sample is classified as AI-generated with very high confidence "
                f"({confidence_percent:.1f}%). The model detected strong indicators of "
                f"synthetic voice generation."
            )
        elif confidence >= 0.7:
            explanation = (
                f"The audio sample is classified as AI-generated with high confidence "
                f"({confidence_percent:.1f}%). The model detected characteristics consistent "
                f"with synthetic voice generation."
            )
        elif confidence >= 0.5:
            explanation = (
                f"The audio sample is classified as AI-generated with moderate confidence "
                f"({confidence_percent:.1f}%). Some characteristics suggest synthetic generation, "
                f"but the result should be interpreted with caution."
            )
        else:
            explanation = (
                f"The audio sample is classified as AI-generated with low confidence "
                f"({confidence_percent:.1f}%). The result is uncertain and may require "
                f"additional verification."
            )
    else:  # Human
        if confidence >= 0.9:
            explanation = (
                f"The audio sample is classified as human-generated with very high confidence "
                f"({confidence_percent:.1f}%). The model detected natural voice characteristics "
                f"consistent with human speech."
            )
        elif confidence >= 0.7:
            explanation = (
                f"The audio sample is classified as human-generated with high confidence "
                f"({confidence_percent:.1f}%). The model detected characteristics consistent "
                f"with natural human voice."
            )
        elif confidence >= 0.5:
            explanation = (
                f"The audio sample is classified as human-generated with moderate confidence "
                f"({confidence_percent:.1f}%). The audio appears to be natural, but the result "
                f"should be interpreted with caution."
            )
        else:
            explanation = (
                f"The audio sample is classified as human-generated with low confidence "
                f"({confidence_percent:.1f}%). The result is uncertain and may require "
                f"additional verification."
            )
    
    # Add probability details if available
    if ai_probability is not None and human_probability is not None:
        explanation += (
            f" AI probability: {ai_probability*100:.1f}%, "
            f"Human probability: {human_probability*100:.1f}%."
        )
    
    return explanation
