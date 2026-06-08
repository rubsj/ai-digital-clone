# A4 — Data Models

> Pydantic models from `src/schemas.py`. Arrows show composition / ownership.

```mermaid
classDiagram
    direction TB

    class EmailMessage {
        +str sender
        +list~str~ recipients
        +str subject
        +str body
        +datetime timestamp
        +str message_id
        +bool is_patch
        +float quote_ratio
    }

    class StyleFeatures {
        +float avg_message_length
        +dict greeting_patterns
        +dict punctuation_patterns
        +float capitalization_ratio
        +float question_frequency
        +float vocabulary_richness
        +list~str~ common_phrases
        +dict reasoning_patterns
        +dict sentiment_distribution
        +float formality_level
        +float technical_terminology
        +float code_snippet_freq
        +float quote_reply_ratio
        +dict patch_language
        +float technical_depth
        +to_vector() ndarray
    }

    class StyleProfile {
        +str leader_name
        +StyleFeatures features
        +ndarray style_vector
        +int email_count
        +datetime last_updated
        +float alpha
        +list~str~ sample_emails
    }

    class KnowledgeChunk {
        +str content
        +str source_topic
        +str source_field
        +int chunk_index
        +ndarray embedding
    }

    class RetrievalResult {
        +KnowledgeChunk chunk
        +float score
        +int rank
    }

    class Citation {
        +str chunk_id
        +str source_topic
        +str text_snippet
        +float relevance_score
    }

    class CloneResponse {
        +str response_text
        +list~Citation~ citations
    }

    class EvaluationResult {
        +float style_score
        +float groundedness_score
        +float confidence_score
        +str explanation
        +list~str~ flags
    }

    class RoutingDecision {
        +Literal decision
        +str reasoning
        +str trigger_reason
        +str trigger_category
        +list~str~ quality_flags
    }

    class FallbackResponse {
        +str acknowledgment
        +list~str~ suggested_redirections
        +str calendar_link
        +list~str~ available_slots
        +str unstyled_response
        +str trigger_category
    }

    class StyledResponse {
        +str query
        +str leader
        +str response
        +EvaluationResult evaluation
        +list~Citation~ citations
        +FallbackResponse fallback
    }

    class LeaderComparison {
        +str query
        +StyledResponse torvalds
        +StyledResponse kroah_hartman
    }

    class CloneState {
        +str query
        +str leader
        +list~RetrievalResult~ chunks
        +StyleProfile style_profile
        +str response_text
        +list~Citation~ citations
        +EvaluationResult evaluation
        +RoutingDecision routing_decision
        +StyledResponse styled_response
        +FallbackResponse fallback_response
    }

    StyleProfile *-- StyleFeatures : features
    RetrievalResult *-- KnowledgeChunk : chunk
    CloneResponse *-- Citation : citations
    StyledResponse *-- EvaluationResult : evaluation
    StyledResponse *-- Citation : citations
    StyledResponse o-- FallbackResponse : fallback (optional)
    LeaderComparison o-- StyledResponse : torvalds / kroah_hartman

    CloneState o-- StyleProfile : style_profile
    CloneState *-- RetrievalResult : chunks
    CloneState *-- Citation : citations
    CloneState o-- EvaluationResult : evaluation
    CloneState o-- RoutingDecision : routing_decision
    CloneState o-- StyledResponse : styled_response
    CloneState o-- FallbackResponse : fallback_response
```

## Notes

- **CloneState** is the typed Pydantic `Flow[CloneState]` state: all fields have defaults; the flow populates them incrementally as each step completes.
- **EvaluationResult** has no `final_score` field — routing is done by Gatekeeper's arithmetic, not a weighted formula (ADR-018). `extra="forbid"` enforces this.
- **StyleFeatures** holds 15 normalized [0, 1] features: 11 base + 4 LKML-specific. `to_vector()` concatenates them to a length-15 `ndarray` for cosine comparison.
- **FallbackResponse** carries `unstyled_response` — the hardcoded failsafe string FallbackAgent writes if its LLM call fails.
