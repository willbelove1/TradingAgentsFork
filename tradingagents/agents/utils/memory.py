import chromadb
from chromadb.config import Settings
# No longer need direct genai or OpenAI SDK imports here for embedding
# import google.generativeai as genai
# import os
# from openai import OpenAI
from typing import Dict, Optional
from tradingagents.llm_clients import BaseLLMClient, get_llm_client # Import BaseLLMClient and factory
from tradingagents.llm_clients.base_client import logger # For logging

class FinancialSituationMemory:
    def __init__(self, name: str, config: Dict, llm_client: Optional[BaseLLMClient] = None):
        """
        Initializes the FinancialSituationMemory.
        Args:
            name (str): Name for the memory collection.
            config (Dict): Configuration dictionary. Expected to have llm_provider,
                           embedding_model (for fallback if llm_client not provided).
            llm_client (BaseLLMClient, optional): An instance of BaseLLMClient.
                                                 If not provided, one will be created using get_llm_client.
        """
        self.config = config
        if llm_client:
            self.llm_client = llm_client
        else:
            logger.info(f"FinancialSituationMemory '{name}': llm_client not provided, creating one.")
            # Ensure the config for the client has the necessary details
            client_config = self.config.copy()
            if 'model' not in client_config: # default_model from main config might be text model
                 client_config['model'] = self.config.get('default_model', 'gemini-pro' if self.config.get('llm_provider') == 'google' else 'gpt-3.5-turbo')
            # Ensure embedding_model is present in client_config for the client to pick up
            if 'embedding_model' not in client_config:
                client_config['embedding_model'] = self.config.get('embedding_model',
                                                                    'models/embedding-001' if self.config.get('llm_provider') == 'google'
                                                                    else 'text-embedding-ada-002')
            self.llm_client = get_llm_client(config=client_config)

        self.chroma_client = chromadb.Client(Settings(allow_reset=True))
        safe_collection_name = name.replace(" ", "_").replace("-", "_")
        self.situation_collection = self.chroma_client.get_or_create_collection(name=safe_collection_name) # Use get_or_create
        logger.info(f"FinancialSituationMemory '{name}' initialized using LLM provider: {self.llm_client.__class__.__name__}")


    def get_embedding(self, text: str) -> list[float]:
        """
        Get embedding for a text using the configured llm_client.
        The underlying client (GeminiClient, OpenAIClient) will use its default embedding model
        or one specified in its config.
        """
        # The llm_client's get_embedding method already has logging and error handling.
        # No need to specify model here, client uses its default embedding model.
        return self.llm_client.get_embedding(text=text)

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
