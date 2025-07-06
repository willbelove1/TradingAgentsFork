import yaml
import os
import logging
from typing import Dict, Optional, List, Any

logger = logging.getLogger(__name__)

PROMPTS_DIR = "prompts" # Relative to project root or where settings are expected
GEMINI_PROMPTS_FILENAME = "gemini_prompts.yaml"
# Can add filenames for other providers later, e.g., OPENAI_PROMPTS_FILENAME

def _format_few_shot_examples(examples: Optional[List[Dict[str, str]]], user_template: str) -> str:
    """
    Formats few-shot examples into a string.
    A simple approach: concatenates input and output.
    More complex formatting might be needed depending on how the LLM best uses few-shot examples.
    """
    if not examples:
        return ""

    formatted_examples = ["\n--- Examples ---"]
    for ex in examples:
        example_input_str = ex.get("input", "")
        # If the few-shot input itself is a template, we might need to indicate that
        # For now, assume 'input' is a string representation of the input part of the example.
        # A common way is to show how the user_template would be filled.
        # This part can be made more sophisticated.
        # Example: if user_template is "Analyze: {text}", and example input is "{text}: 'Good stock'"
        # it implies the actual input was 'Good stock'.

        # Simple formatting:
        formatted_examples.append(f"Example Input:\n{example_input_str}")
        formatted_examples.append(f"Example Output:\n{ex.get('output', '')}")
        formatted_examples.append("---")
    return "\n".join(formatted_examples) + "\n"


def load_prompts_from_file(filepath: str) -> Dict[str, Any]:
    """Loads and preprocesses prompts from a given YAML filepath."""
    if not os.path.exists(filepath):
        logger.warning(f"Prompt file not found at: {filepath}. Returning empty prompts.")
        return {}

    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            raw_prompts = yaml.safe_load(f)
        if not raw_prompts:
            logger.warning(f"Prompt file '{filepath}' is empty. Returning empty prompts.")
            return {}

        processed_prompts: Dict[str, Any] = {"raw_yaml_content": raw_prompts} # Store raw for debugging

        default_system_prompt = raw_prompts.get('default_system_prompt', '')
        processed_prompts['default_system_prompt'] = default_system_prompt

        for key, config in raw_prompts.items():
            if key in ['default_system_prompt', 'raw_yaml_content']: # Skip special keys
                continue

            if isinstance(config, dict):
                prompt_data: Dict[str, Any] = {
                    "type": config.get("type", "llm_client"), # Default to llm_client type
                    "description": config.get("description", ""),
                    "system_prompt": config.get("system_prompt", default_system_prompt), # Use specific or default
                    "user_prompt_template": config.get("user_prompt_template", ""),
                }

                few_shot_examples_raw = config.get("few_shot_examples")
                if few_shot_examples_raw and isinstance(few_shot_examples_raw, list):
                    # Store both raw and a simple formatted version for llm_client type prompts
                    prompt_data["few_shot_examples_raw"] = few_shot_examples_raw
                    prompt_data["few_shot_examples_string"] = _format_few_shot_examples(
                        few_shot_examples_raw,
                        prompt_data["user_prompt_template"]
                    )
                else:
                    prompt_data["few_shot_examples_raw"] = []
                    prompt_data["few_shot_examples_string"] = ""

                # For Langchain type, we might just store the system_message directly if that's all
                if prompt_data["type"] == "langchain" and "system_message" in config:
                    prompt_data["system_message"] = config.get("system_message", default_system_prompt)
                    # Langchain templates often handle user input and few-shots differently,
                    # so we might not pre-format user_prompt_template or few_shot_examples_string for them here.

                processed_prompts[key] = prompt_data
            else:
                logger.warning(f"Skipping invalid prompt configuration for key '{key}' in '{filepath}'. Expected a dictionary.")

        logger.info(f"Successfully loaded and processed prompts from '{filepath}'.")
        return processed_prompts

    except yaml.YAMLError as e:
        logger.error(f"Error parsing YAML from '{filepath}': {e}")
        return {}
    except Exception as e:
        logger.error(f"An unexpected error occurred while loading prompts from '{filepath}': {e}")
        return {}


def get_prompt_config(prompt_key: str, provider_name: Optional[str] = None, prompts_config: Optional[Dict] = None) -> Optional[Dict[str, Any]]:
    """
    Retrieves a specific prompt configuration by its key.
    If prompts_config is not provided, it attempts to load it.
    Args:
        prompt_key (str): The key of the prompt to retrieve (e.g., "SentimentAnalysis").
        provider_name (str, optional): Name of the LLM provider (e.g., "gemini", "openai").
                                       Used to determine which prompt file to load. Defaults to "gemini".
        prompts_config (Dict, optional): Pre-loaded prompts dictionary. If None, will load from file.
    Returns:
        Optional[Dict[str, Any]]: The prompt configuration dictionary or None if not found.
    """
    if prompts_config is None:
        # For now, hardcoding to Gemini. This can be made more dynamic.
        filename_to_load = GEMINI_PROMPTS_FILENAME # Default to Gemini
        if provider_name and provider_name.lower() == "openai": # Example for future
            # filename_to_load = OPENAI_PROMPTS_FILENAME
            pass

        # Determine path to prompts directory (assuming it's relative to project root)
        # This assumes prompt_loader.py is in tradingagents/config/
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        filepath = os.path.join(project_root, PROMPTS_DIR, filename_to_load)

        prompts_config = load_prompts_from_file(filepath)

    if not prompts_config:
        return None

    prompt_data = prompts_config.get(prompt_key)
    if prompt_data is None:
        logger.warning(f"Prompt key '{prompt_key}' not found in loaded prompts.")
        return None
    return prompt_data


def format_prompt(prompt_config: Dict[str, Any], context_vars: Optional[Dict[str, Any]] = None) -> str:
    """
    Formats a complete prompt string using the prompt_config and context variables.
    For 'llm_client' type prompts, it combines system_prompt, few_shot_examples_string,
    and the user_prompt_template filled with context_vars.
    For 'langchain' type, it might just return the system_message or a combination,
    as Langchain handles its own templating.
    """
    if not prompt_config:
        return ""

    final_parts = []

    system_prompt = prompt_config.get("system_prompt", "")
    if system_prompt: # Add system prompt if available
        final_parts.append(system_prompt)

    # For Langchain, system_message might be the primary content if not using complex templates here
    if prompt_config.get("type") == "langchain" and "system_message" in prompt_config:
        # If only system_message is used by Langchain setup, other parts might be skipped or handled differently.
        # For now, let's assume if system_message is there, it's the main part for Langchain from this loader.
        # Langchain's ChatPromptTemplate will handle the user input part.
        if not system_prompt: # If system_prompt was not already added from prompt_config["system_prompt"]
             final_parts.append(prompt_config["system_message"])
        # Depending on how Langchain templates are built, few-shots might be formatted differently or injected by Langchain itself.
        # For now, we won't add few_shot_examples_string for 'langchain' type here.

    # For llm_client type, assemble more parts
    if prompt_config.get("type") == "llm_client":
        few_shot_string = prompt_config.get("few_shot_examples_string", "")
        if few_shot_string:
            final_parts.append(few_shot_string)

        user_template = prompt_config.get("user_prompt_template", "")
        if user_template:
            try:
                filled_user_prompt = user_template.format(**(context_vars or {}))
                final_parts.append(filled_user_prompt)
            except KeyError as e:
                logger.error(f"Missing key {e} in context_vars for user_prompt_template: {user_template}")
                final_parts.append(f"[ERROR: Missing data for template: {user_template}]")
            except Exception as e:
                logger.error(f"Error formatting user_prompt_template: {e}")
                final_parts.append(f"[ERROR: Could not format template: {user_template}]")

    return "\n\n".join(part for part in final_parts if part.strip())


# Example Usage (for testing the loader and formatter)
if __name__ == '__main__':
    # Ensure prompts/gemini_prompts.yaml exists in the project root for this test
    # This assumes you run: python -m tradingagents.config.prompt_loader

    print("--- Testing load_prompts_from_file ---")
    project_r = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    gemini_prompts_path = os.path.join(project_r, PROMPTS_DIR, GEMINI_PROMPTS_FILENAME)

    if not os.path.exists(gemini_prompts_path):
        print(f"WARNING: Test requires '{GEMINI_PROMPTS_FILENAME}' in '{os.path.join(project_r, PROMPTS_DIR)}'. Skipping some tests.")
    else:
        all_prompts = load_prompts_from_file(gemini_prompts_path)
        if all_prompts:
            print(f"Default System Prompt: {all_prompts.get('default_system_prompt')[:100]}...")

            print("\n--- Testing get_prompt_config ---")
            sentiment_config = get_prompt_config("SentimentAnalysis", prompts_config=all_prompts)
            if sentiment_config:
                print(f"SentimentAnalysis Config: {sentiment_config.get('description')}")
                # print(sentiment_config)

                print("\n--- Testing format_prompt for SentimentAnalysis ---")
                formatted_sentiment_prompt = format_prompt(
                    sentiment_config,
                    context_vars={"text_input": "Đây là một ngày tuyệt vời để giao dịch!"}
                )
                print(formatted_sentiment_prompt)

            trade_decision_config = get_prompt_config("TradeDecision", prompts_config=all_prompts)
            if trade_decision_config:
                print("\n--- Testing format_prompt for TradeDecision ---")
                formatted_trade_prompt = format_prompt(
                    trade_decision_config,
                    context_vars={
                        "market_data": "Thị trường đang có xu hướng tăng.",
                        "technical_analysis": "Các chỉ báo đều cho tín hiệu mua.",
                        "news_summary": "Không có tin tức tiêu cực nào đáng kể."
                    }
                )
                print(formatted_trade_prompt)

            market_analyst_lc_config = get_prompt_config("MarketAnalystLC", prompts_config=all_prompts)
            if market_analyst_lc_config:
                 print("\n--- Testing format_prompt for MarketAnalystLC (Langchain type) ---")
                 # For Langchain, context_vars might not be used by this basic formatter
                 # as ChatPromptTemplate handles it. We'd primarily get the system_message.
                 formatted_lc_prompt = format_prompt(market_analyst_lc_config, context_vars={"ticker": "XYZ", "current_date": "2023-01-01"})
                 print(formatted_lc_prompt) # Should mainly be the system_message
        else:
            print("Failed to load any prompts.")
