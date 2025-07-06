import chromadb
from chromadb.config import Settings
import google.generativeai as genai
import os

class FinancialSituationMemory:
    def __init__(self, name, config):
        self.llm_provider = config.get("llm_provider", "openai").lower()
        if self.llm_provider == "google":
            if not os.getenv("GOOGLE_API_KEY"):
                try:
                    from dotenv import load_dotenv
                    load_dotenv()
                    if not os.getenv("GOOGLE_API_KEY"):
                        raise ValueError("GOOGLE_API_KEY not found for Google provider in Memory.")
                except ImportError:
                    raise ValueError("dotenv package not installed. Please install it or set GOOGLE_API_KEY for Google provider in Memory.")
            genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
            self.embedding_model_name = "models/embedding-001"
        elif config.get("backend_url") == "http://localhost:11434/v1": # ollama likely
            self.embedding_model_name = "nomic-embed-text"
            # Assuming OpenAI client is still used for local/OpenRouter embeddings if not Google
            from openai import OpenAI
            self.client = OpenAI(base_url=config["backend_url"])
        else: # Default to OpenAI
            self.embedding_model_name = "text-embedding-3-small"
            from openai import OpenAI
            self.client = OpenAI(base_url=config.get("backend_url", "https://api.openai.com/v1"))

        self.chroma_client = chromadb.Client(Settings(allow_reset=True))
        # Ensure collection name is valid for ChromaDB (e.g., no spaces, certain characters)
        safe_collection_name = name.replace(" ", "_").replace("-", "_")
        self.situation_collection = self.chroma_client.create_collection(name=safe_collection_name)

    def get_embedding(self, text):
        """Get embedding for a text based on the configured LLM provider."""
        if self.llm_provider == "google":
            result = genai.embed_content(model=self.embedding_model_name, content=text)
            return result['embedding']
        else: # OpenAI or compatible
            response = self.client.embeddings.create(
                model=self.embedding_model_name, input=text
            )
            return response.data[0].embedding

    def add_situations(self, situations_and_advice):
        """Add financial situations and their corresponding advice. Parameter is a list of tuples (situation, rec)"""

        situations = []
        advice = []
        ids = []
        embeddings = []

        offset = self.situation_collection.count()

        for i, (situation, recommendation) in enumerate(situations_and_advice):
            situations.append(situation)
            advice.append(recommendation)
            ids.append(str(offset + i))
            embeddings.append(self.get_embedding(situation))

        self.situation_collection.add(
            documents=situations,
            metadatas=[{"recommendation": rec} for rec in advice],
            embeddings=embeddings,
            ids=ids,
        )

    def get_memories(self, current_situation, n_matches=1):
        """Find matching recommendations using OpenAI embeddings"""
        query_embedding = self.get_embedding(current_situation)

        results = self.situation_collection.query(
            query_embeddings=[query_embedding],
            n_results=n_matches,
            include=["metadatas", "documents", "distances"],
        )

        matched_results = []
        for i in range(len(results["documents"][0])):
            matched_results.append(
                {
                    "matched_situation": results["documents"][0][i],
                    "recommendation": results["metadatas"][0][i]["recommendation"],
                    "similarity_score": 1 - results["distances"][0][i],
                }
            )

        return matched_results


if __name__ == "__main__":
    # Example usage
    matcher = FinancialSituationMemory()

    # Example data
    example_data = [
        (
            "High inflation rate with rising interest rates and declining consumer spending",
            "Consider defensive sectors like consumer staples and utilities. Review fixed-income portfolio duration.",
        ),
        (
            "Tech sector showing high volatility with increasing institutional selling pressure",
            "Reduce exposure to high-growth tech stocks. Look for value opportunities in established tech companies with strong cash flows.",
        ),
        (
            "Strong dollar affecting emerging markets with increasing forex volatility",
            "Hedge currency exposure in international positions. Consider reducing allocation to emerging market debt.",
        ),
        (
            "Market showing signs of sector rotation with rising yields",
            "Rebalance portfolio to maintain target allocations. Consider increasing exposure to sectors benefiting from higher rates.",
        ),
    ]

    # Add the example situations and recommendations
    matcher.add_situations(example_data)

    # Example query
    current_situation = """
    Market showing increased volatility in tech sector, with institutional investors 
    reducing positions and rising interest rates affecting growth stock valuations
    """

    try:
        recommendations = matcher.get_memories(current_situation, n_matches=2)

        for i, rec in enumerate(recommendations, 1):
            print(f"\nMatch {i}:")
            print(f"Similarity Score: {rec['similarity_score']:.2f}")
            print(f"Matched Situation: {rec['matched_situation']}")
            print(f"Recommendation: {rec['recommendation']}")

    except Exception as e:
        print(f"Error during recommendation: {str(e)}")
