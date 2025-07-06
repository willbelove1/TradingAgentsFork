import questionary
from typing import List, Optional, Tuple, Dict

from cli.models import AnalystType

ANALYST_ORDER = [
    ("Market Analyst", AnalystType.MARKET),
    ("Social Media Analyst", AnalystType.SOCIAL),
    ("News Analyst", AnalystType.NEWS),
    ("Fundamentals Analyst", AnalystType.FUNDAMENTALS),
]


def get_ticker() -> str:
    """Prompt the user to enter a ticker symbol."""
    ticker = questionary.text(
        "Enter the ticker symbol to analyze:",
        validate=lambda x: len(x.strip()) > 0 or "Please enter a valid ticker symbol.",
        style=questionary.Style(
            [
                ("text", "fg:green"),
                ("highlighted", "noinherit"),
            ]
        ),
    ).ask()

    if not ticker:
        console.print("\n[red]No ticker symbol provided. Exiting...[/red]")
        exit(1)

    return ticker.strip().upper()


def get_analysis_date() -> str:
    """Prompt the user to enter a date in YYYY-MM-DD format."""
    import re
    from datetime import datetime

    def validate_date(date_str: str) -> bool:
        if not re.match(r"^\d{4}-\d{2}-\d{2}$", date_str):
            return False
        try:
            datetime.strptime(date_str, "%Y-%m-%d")
            return True
        except ValueError:
            return False

    date = questionary.text(
        "Enter the analysis date (YYYY-MM-DD):",
        validate=lambda x: validate_date(x.strip())
        or "Please enter a valid date in YYYY-MM-DD format.",
        style=questionary.Style(
            [
                ("text", "fg:green"),
                ("highlighted", "noinherit"),
            ]
        ),
    ).ask()

    if not date:
        console.print("\n[red]No date provided. Exiting...[/red]")
        exit(1)

    return date.strip()


def select_analysts() -> List[AnalystType]:
    """Select analysts using an interactive checkbox."""
    choices = questionary.checkbox(
        "Select Your [Analysts Team]:",
        choices=[
            questionary.Choice(display, value=value) for display, value in ANALYST_ORDER
        ],
        instruction="\n- Press Space to select/unselect analysts\n- Press 'a' to select/unselect all\n- Press Enter when done",
        validate=lambda x: len(x) > 0 or "You must select at least one analyst.",
        style=questionary.Style(
            [
                ("checkbox-selected", "fg:green"),
                ("selected", "fg:green noinherit"),
                ("highlighted", "noinherit"),
                ("pointer", "noinherit"),
            ]
        ),
    ).ask()

    if not choices:
        console.print("\n[red]No analysts selected. Exiting...[/red]")
        exit(1)

    return choices


def select_research_depth() -> int:
    """Select research depth using an interactive selection."""

    # Define research depth options with their corresponding values
    DEPTH_OPTIONS = [
        ("Shallow - Quick research, few debate and strategy discussion rounds", 1),
        ("Medium - Middle ground, moderate debate rounds and strategy discussion", 3),
        ("Deep - Comprehensive research, in depth debate and strategy discussion", 5),
    ]

    choice = questionary.select(
        "Select Your [Research Depth]:",
        choices=[
            questionary.Choice(display, value=value) for display, value in DEPTH_OPTIONS
        ],
        instruction="\n- Use arrow keys to navigate\n- Press Enter to select",
        style=questionary.Style(
            [
                ("selected", "fg:yellow noinherit"),
                ("highlighted", "fg:yellow noinherit"),
                ("pointer", "fg:yellow noinherit"),
            ]
        ),
    ).ask()

    if choice is None:
        console.print("\n[red]No research depth selected. Exiting...[/red]")
        exit(1)

    return choice


def select_shallow_thinking_agent(provider) -> str:
    """Select shallow thinking llm engine using an interactive selection."""

    # Define shallow thinking llm engine options with their corresponding model names
    SHALLOW_AGENT_OPTIONS = {
        "google": [
            ("Gemini 1.5 Flash - Fast and versatile", "gemini-1.5-flash-latest"), # Updated to more common names
            ("Gemini 1.0 Pro - Solid performance (can be quick for some tasks)", "gemini-1.0-pro"),
            # ("Gemini 2.0 Flash-Lite - Cost efficiency and low latency", "gemini-2.0-flash-lite"), # Kept old ones for reference if needed
            # ("Gemini 2.0 Flash - Next generation features, speed, and thinking", "gemini-2.0-flash"),
            # ("Gemini 2.5 Flash - Adaptive thinking, cost efficiency", "gemini-2.5-flash-preview-05-20"),
        ],
        # "openai": [], # Removed
        # "anthropic": [], # Removed
        # "openrouter": [], # Removed
        # "ollama": [] # Removed
    }

    # Ensure provider is in options, default to 'google' if not (e.g. if provider was 'openai' before)
    provider_key = provider.lower()
    if provider_key not in SHALLOW_AGENT_OPTIONS:
        print(f"[yellow]Warning: Provider '{provider}' not found in SHALLOW_AGENT_OPTIONS, defaulting to 'google'.[/yellow]")
        provider_key = "google"

    choice = questionary.select(
        "Select Your [Quick-Thinking LLM Engine]:",
        choices=[
            questionary.Choice(display, value=value)
            for display, value in SHALLOW_AGENT_OPTIONS[provider.lower()]
        ],
        instruction="\n- Use arrow keys to navigate\n- Press Enter to select",
        style=questionary.Style(
            [
                ("selected", "fg:magenta noinherit"),
                ("highlighted", "fg:magenta noinherit"),
                ("pointer", "fg:magenta noinherit"),
            ]
        ),
    ).ask()

    if choice is None:
        console.print(
            "\n[red]No shallow thinking llm engine selected. Exiting...[/red]"
        )
        exit(1)

    return choice


def select_deep_thinking_agent(provider) -> str:
    """Select deep thinking llm engine using an interactive selection."""

    # Define deep thinking llm engine options with their corresponding model names
    DEEP_AGENT_OPTIONS = {
        "google": [
            ("Gemini 1.0 Pro - Solid performance for general tasks", "gemini-1.0-pro"),
            ("Gemini 1.5 Pro - Advanced reasoning and long context", "gemini-1.5-pro-latest"),
            # ("Gemini 2.0 Flash-Lite - Cost efficiency and low latency", "gemini-2.0-flash-lite"), # Kept for ref
            # ("Gemini 2.0 Flash - Next generation features, speed, and thinking", "gemini-2.0-flash"),
            # ("Gemini 2.5 Flash - Adaptive thinking, cost efficiency", "gemini-2.5-flash-preview-05-20"),
            # ("Gemini 2.5 Pro", "gemini-2.5-pro-preview-06-05"),
        ],
        # "openai": [], # Removed
        # "anthropic": [], # Removed
        # "openrouter": [], # Removed
        # "ollama": [] # Removed
    }
    
    provider_key = provider.lower()
    if provider_key not in DEEP_AGENT_OPTIONS:
        print(f"[yellow]Warning: Provider '{provider}' not found in DEEP_AGENT_OPTIONS, defaulting to 'google'.[/yellow]")
        provider_key = "google"

    choice = questionary.select(
        "Select Your [Deep-Thinking LLM Engine]:",
        choices=[
            questionary.Choice(display, value=value)
            for display, value in DEEP_AGENT_OPTIONS[provider_key]
        ],
        instruction="\n- Use arrow keys to navigate\n- Press Enter to select",
        style=questionary.Style(
            [
                ("selected", "fg:magenta noinherit"),
                ("highlighted", "fg:magenta noinherit"),
                ("pointer", "fg:magenta noinherit"),
            ]
        ),
    ).ask()

    if choice is None:
        console.print("\n[red]No deep thinking llm engine selected. Exiting...[/red]")
        exit(1)

    return choice

def select_llm_provider() -> tuple[str, str]:
    """Select the LLM Provider. Now defaults to Google and its URL is illustrative as not directly used by GeminiClient."""
    # Define LLM Provider options
    # Since this is a Gemini-only fork, we primarily offer Google.
    # The URL for Google is illustrative as GeminiClient uses genai.configure() not a base_url.
    # The `backend_url` in config is not used by GeminiClient.
    PROVIDER_CHOICES = [
        ("Google (Gemini)", ("Google", "https://generativelanguage.googleapis.com/v1")), # URL is mostly for info
        # ("OpenAI", ("OpenAI", "https://api.openai.com/v1")), # Removed
        # ("Anthropic", ("Anthropic", "https://api.anthropic.com/")), # Removed
        # ("Ollama (via OpenAI compatible API)", ("Ollama", "http://localhost:11434/v1")), # Removed
        # ("OpenRouter", ("OpenRouter", "https://openrouter.ai/api/v1")), # Removed
    ]
    
    if len(PROVIDER_CHOICES) == 1:
        print(f"[cyan]Defaulting to LLM Provider: {PROVIDER_CHOICES[0][0]}[/cyan]")
        selected_display_name, selected_url = PROVIDER_CHOICES[0][1]
        return selected_display_name, selected_url # Return tuple (display_name, url)

    choice_obj = questionary.select( # Renamed variable to avoid conflict
        "Select your LLM Provider:",
        choices=[
            questionary.Choice(display, value=value_tuple) # value is now the tuple
            for display, value_tuple in PROVIDER_CHOICES
        ],
        instruction="\n- Use arrow keys to navigate\n- Press Enter to select",
        style=questionary.Style(
            [
                ("selected", "fg:magenta noinherit"),
                ("highlighted", "fg:magenta noinherit"),
                ("pointer", "fg:magenta noinherit"),
            ]
        ),
    ).ask()
    
    if choice_obj is None: # Use renamed variable
        print("\n[red]No LLM Provider selected. Exiting...[/red]") # Changed message
        exit(1)
    
    # choice_obj is now the tuple, e.g. ("Google", "https://...")
    selected_display_name, selected_url = choice_obj
    print(f"You selected LLM Provider: {selected_display_name}\t(URL: {selected_url})") # Adjusted print
    
    return selected_display_name, selected_url
