"""
RAG Pipeline with LLM integration - Optimized for CPU
"""

from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch
from typing import List, Dict, Tuple, Optional
import re


class RAGPipeline:
    """RAG system with LLM and context checking - CPU Optimized"""

    def __init__(
        self,
        model_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        device: str = "cpu",
    ):
        """
        Initialize RAG pipeline with LLM - CPU Optimized

        Args:
            model_name: HuggingFace model name (default: TinyLlama for CPU)
            device: Device to run on (forced to 'cpu' for stability)
        """
        print(f"Loading LLM: {model_name}")
        print("⚙️  CPU Mode - Optimized for Ryzen 5 5600G")

        # Force CPU for stability
        self.device = "cpu"
        print(f"Using device: {self.device}")

        # Load model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Ensure pad token is set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load model in CPU mode with optimizations
        print("Loading model... This may take a few minutes...")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float32,  # Use float32 for CPU
            low_cpu_mem_usage=True,
            device_map=None,  # No device map for CPU
        )
        self.model.to(self.device)
        self.model.eval()  # Set to evaluation mode

        # Create pipeline with CPU-friendly settings
        self.pipe = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            max_new_tokens=256,  # Reduced for faster CPU inference
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.1,
            device=self.device,
        )

        print("✅ LLM loaded successfully (CPU mode)")

        # Conversation history
        self.conversation_history = []

    def check_context_relevance(
        self,
        query: str,
        retrieved_chunks: List[Tuple[Dict, float]],
        threshold: float = 0.2,
        use_llm_check: bool = False,
    ) -> Tuple[bool, str]:
        """
        Check if retrieved context is relevant to the query

        Args:
            query: User query
            retrieved_chunks: Retrieved document chunks with scores
            threshold: Minimum similarity threshold (lowered for better recall)
            use_llm_check: Use LLM for verification (slower but more accurate)

        Returns:
            (is_relevant, reason)
        """
        # Check similarity scores first
        if not retrieved_chunks:
            return False, "No relevant documents found"

        max_score = max([score for _, score in retrieved_chunks])
        if max_score < threshold:
            return (
                False,
                f"Retrieved documents have low relevance (max score: {max_score:.2f})",
            )

        # If we have good similarity score, consider it relevant
        # This avoids false negatives from LLM verification
        if not use_llm_check:
            return True, f"Found relevant content (score: {max_score:.2f})"

        # Optional: Use LLM for verification (more accurate but slower)
        context = "\n\n".join(
            [chunk["text"][:300] for chunk, _ in retrieved_chunks[:2]]
        )

        verification_prompt = f"""Context from documents: {context}

Question: {query}

Does the context contain information relevant to the question? Answer RELEVANT if the context has any related information, even partial. Answer NOT_RELEVANT only if completely unrelated."""

        try:
            response = self.pipe(
                verification_prompt,
                max_new_tokens=20,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )[0]["generated_text"]

            # Extract the answer after the prompt
            answer = response.split(verification_prompt)[-1].strip()

            # More lenient check - if "NOT_RELEVANT" is explicitly stated, then it's not relevant
            # Otherwise, assume it's relevant
            is_relevant = (
                "NOT_RELEVANT" not in answer.upper() or "RELEVANT" in answer.upper()
            )

            # Extract reason
            reason_match = re.search(r":\s*(.+)", answer)
            reason = (
                reason_match.group(1) if reason_match else "Context check completed"
            )

            return is_relevant, reason
        except Exception as e:
            print(f"Context verification error: {e}")
            # Default to similarity threshold - be more permissive
            return (
                max_score >= threshold,
                f"Using similarity threshold (score: {max_score:.2f})",
            )

    def generate_answer(
        self,
        query: str,
        retrieved_chunks: List[Tuple[Dict, float]],
        use_history: bool = True,
        use_llm_verification: bool = False,
    ) -> Dict:
        """
        Generate answer using RAG - CPU Optimized

        Args:
            query: User query
            retrieved_chunks: Retrieved document chunks with scores
            use_history: Whether to use conversation history
            use_llm_verification: Use LLM to verify context (slower, more false negatives)

        Returns:
            Dictionary with answer and sources
        """
        # Check context relevance (by default, trust similarity scores)
        is_relevant, reason = self.check_context_relevance(
            query, retrieved_chunks, use_llm_check=use_llm_verification
        )

        if not is_relevant:
            return {
                "answer": f"I cannot answer this question based on the provided documents. {reason}",
                "sources": [],
                "out_of_context": True,
            }

        # Prepare context (reduced for CPU)
        context_parts = []
        sources = []

        for i, (chunk, score) in enumerate(
            retrieved_chunks[:3], 1
        ):  # Reduced to 3 chunks
            # Limit chunk size for faster processing
            chunk_text = chunk["text"][:500]
            context_parts.append(f"[Doc {i}] {chunk_text}")

            source_info = {
                "source": chunk["source"],
                "page": chunk.get("page"),
                "score": score,
                "doc_type": chunk.get("doc_type"),
            }

            if "row" in chunk:
                source_info["row"] = chunk["row"]
            if "chapter" in chunk:
                source_info["chapter"] = chunk["chapter"]

            sources.append(source_info)

        context = "\n\n".join(context_parts)

        # Build conversation context (limited for CPU)
        conv_context = ""
        if use_history and self.conversation_history:
            recent_history = self.conversation_history[-2:]  # Last 2 exchanges only
            conv_context = "\n".join(
                [f"Q: {h['query']}\nA: {h['answer'][:150]}" for h in recent_history]
            )
            conv_context = f"\nHistory:\n{conv_context}\n"

        # Create RAG prompt (shorter for CPU)
        prompt = f"""You are a helpful assistant. Answer based on the context.

{conv_context}Context:
{context}

Question: {query}

Answer concisely:"""

        # Generate response
        try:
            response = self.pipe(
                prompt,
                max_new_tokens=256,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
            )[0]["generated_text"]

            # Extract answer (remove the prompt)
            answer = response.split("Answer concisely:")[-1].strip()

            # Clean up the answer
            if answer:
                # Take first complete paragraph
                answer = answer.split("\n\n")[0]
                # Remove any trailing incomplete sentences
                sentences = answer.split(". ")
                if len(sentences) > 1 and not sentences[-1].endswith((".", "!", "?")):
                    answer = ". ".join(sentences[:-1]) + "."
            else:
                answer = "I found relevant information but couldn't generate a complete answer. Please rephrase your question."

            # Update conversation history
            self.conversation_history.append({"query": query, "answer": answer})

            # Keep history manageable
            if len(self.conversation_history) > 5:
                self.conversation_history = self.conversation_history[-5:]

            return {"answer": answer, "sources": sources, "out_of_context": False}

        except Exception as e:
            print(f"Generation error: {e}")
            return {
                "answer": f"I found relevant information in the documents but encountered an error generating the answer. Please try rephrasing your question.",
                "sources": sources,
                "out_of_context": False,
            }

    def reset_conversation(self):
        """Clear conversation history"""
        self.conversation_history = []
        print("Conversation history cleared")
