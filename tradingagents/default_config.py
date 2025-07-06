import os

DEFAULT_CONFIG = {
    "project_dir": os.path.abspath(os.path.join(os.path.dirname(__file__), ".")),
    "results_dir": os.getenv("TRADINGAGENTS_RESULTS_DIR", "./results"),
    "data_dir": "/Users/yluo/Documents/Code/ScAI/FR1-data",
    "data_cache_dir": os.path.join(
        os.path.abspath(os.path.join(os.path.dirname(__file__), ".")),
        "dataflows/data_cache",
    ),
    # LLM settings
    "llm_provider": "google",  # Can be "google", "openai", etc.
    "default_model": "gemini-pro", # General purpose model for the selected provider
    "embedding_model": "models/embedding-001", # Specific to the provider
    # "backend_url": "https://generativelanguage.googleapis.com/v1beta", # GeminiClient doesn't use this directly, OpenAIClient uses 'base_url'
    # For OpenAI compatible APIs (like Ollama, OpenRouter):
    # "openai_base_url": "http://localhost:11434/v1", # Example for Ollama
    # "openai_api_key": "ollama", # Example for Ollama

    # Specific model names for different agent roles or tasks can be added here if needed,
    # or configured at a more granular level later.
    # For now, the clients will use 'default_model' and 'embedding_model' from this config.
    "deep_think_llm": "gemini-pro", # Retained for Langchain agents that might still use this key
    "quick_think_llm": "gemini-pro",# Retained for Langchain agents
    # Debate and discussion settings
    "max_debate_rounds": 1,
    "max_risk_discuss_rounds": 1,
    "max_recur_limit": 100,
    # Tool settings
    "online_tools": True,
}
