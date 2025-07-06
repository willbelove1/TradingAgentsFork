import streamlit as st
import datetime
import os
import logging
from collections import deque
from typing import Dict, List, Optional, Any # Added Any, Optional, List

# Import necessary components from tradingagents
try:
    from tradingagents.dataflows.config import get_config, initialize_config
    from tradingagents.default_config import DEFAULT_CONFIG
    from tradingagents.graph.trading_graph import TradingAgentsGraph
    from tradingagents.agents.utils.agent_states import AgentState, InvestDebateState, RiskDebateState # For type hinting if needed
    from tradingagents.llm_clients import SUPPORTED_PROVIDERS # To get provider list
except ImportError as e:
    st.error(f"Failed to import TradingAgents modules. Ensure PYTHONPATH is set correctly or run Streamlit from project root. Error: {e}")
    def get_config(): return DEFAULT_CONFIG if 'DEFAULT_CONFIG' in globals() else {}
    def initialize_config(custom_model_settings_path=None): pass
    class TradingAgentsGraph:
        def __init__(self, *args, **kwargs): pass
        def propagate(self, *args, **kwargs):
            yield {"messages": ["TradingAgentsGraph not loaded due to import error."]}
            return {}, "ERROR"
    SUPPORTED_PROVIDERS = {"google": None} # Placeholder, ensure only google related

# --- App Configuration ---
st.set_page_config(layout="wide", page_title="TradingAgents Dashboard (Gemini Edition)")

# --- Logging & State ---
if 'app_logs' not in st.session_state:
    st.session_state.app_logs = deque(maxlen=200)
if 'analysis_running' not in st.session_state:
    st.session_state.analysis_running = False
if 'current_status_text' not in st.session_state:
    st.session_state.current_status_text = "Ready to start."
if 'agent_statuses' not in st.session_state: # Stores dicts: {"status": "pending/...", "details": "..."}
    st.session_state.agent_statuses = {}
if 'current_report_markdown' not in st.session_state:
    st.session_state.current_report_markdown = ""
if 'final_report_markdown' not in st.session_state:
    st.session_state.final_report_markdown = ""
if 'final_decision' not in st.session_state:
    st.session_state.final_decision = ""
if 'last_active_node_ui_key' not in st.session_state: # Track last active agent for completion logic
    st.session_state.last_active_node_ui_key = None


def add_log_message(message, level="INFO"):
    timestamp = datetime.datetime.now().strftime("%H:%M:%S")
    st.session_state.app_logs.append(f"[{timestamp}][{level}] {message}")

# --- Agent Status & Progress Display ---
AGENT_DISPLAY_ORDER_AND_NAMES: List[Tuple[str, str]] = [
    ("Market Analyst", "Market Analyst"), ("Social Analyst", "Social Media Analyst"),
    ("News Analyst", "News Analyst"), ("Fundamentals Analyst", "Fundamentals Analyst"),
    ("Bull Researcher", "Bull Researcher"), ("Bear Researcher", "Bear Researcher"),
    ("Research Manager", "Research Manager"), ("Trader", "Trader"),
    ("RiskyDebator", "Risky Debator"), ("NeutralDebator", "Neutral Debator"),
    ("SafeDebator", "Safe Debator"), ("RiskManager", "Risk Manager (Portfolio Mgr)")
]
NODE_NAME_TO_UI_KEY_MAP: Dict[str, str] = { # Maps LangGraph node names to UI keys from AGENT_DISPLAY_ORDER_AND_NAMES
    "Market Analyst": "Market Analyst", "Social Analyst": "Social Analyst", # Assuming direct match for analysts
    "News Analyst": "News Analyst", "Fundamentals Analyst": "Fundamentals Analyst",
    "Bull Researcher": "Bull Researcher", "Bear Researcher": "Bear Researcher",
    "Research Manager": "Research Manager", "Trader": "Trader",
    "Risky Analyst": "RiskyDebator", # Graph node name for create_risky_debator might be "Risky Analyst"
    "Neutral Analyst": "NeutralDebator", "Safe Analyst": "SafeDebator",
    "Risk Judge": "RiskManager", # Graph node name for create_risk_manager might be "Risk Judge"
    # Add actual node names from your compiled LangGraph if they differ.
    # Special keys from stream like '__start__', '__end__' should be handled or ignored.
}

def get_initial_agent_statuses(selected_analyst_names_from_ui: List[str]) -> Dict[str, Dict[str, str]]:
    statuses = {}
    selected_analyst_ui_keys = {f"{name_ui} Analyst" for name_ui in selected_analyst_names_from_ui}

    for key, _ in AGENT_DISPLAY_ORDER_AND_NAMES:
        is_selected_analyst = any(name_ui_part in key for name_ui_part in ["Market", "Social", "News", "Fundamentals"])
        if is_selected_analyst and key not in selected_analyst_ui_keys:
            statuses[key] = {"status": "skipped", "details": "Not selected"}
        else:
            statuses[key] = {"status": "pending", "details": ""}
    return statuses

def map_stream_key_to_ui_key(stream_key_from_graph: Optional[str]) -> Optional[str]:
    if not stream_key_from_graph: return None

    # Check for exact match in mapping first (more reliable)
    if stream_key_from_graph in NODE_NAME_TO_UI_KEY_MAP:
        return NODE_NAME_TO_UI_KEY_MAP[stream_key_from_graph]

    # If not found, try matching against the UI keys directly (first element of tuples in AGENT_DISPLAY_ORDER_AND_NAMES)
    # This is useful if node names in the graph are sometimes the same as the desired UI keys.
    normalized_stream_key = stream_key_from_graph.lower().replace(" ", "").replace("_", "")
    for ui_key, _ in AGENT_DISPLAY_ORDER_AND_NAMES:
        if normalized_stream_key == ui_key.lower().replace(" ", "").replace("_", ""):
            return ui_key # Return the UI key (which is also the key for st.session_state.agent_statuses)

    # Fallback: if stream_key contains a UI key (e.g., stream_key "Market Analyst Node" contains "Market Analyst")
    for ui_key, _ in AGENT_DISPLAY_ORDER_AND_NAMES:
        if ui_key.lower() in stream_key_from_graph.lower():
            return ui_key

    # add_log_message(f"Warning: No UI mapping for graph stream key '{stream_key_from_graph}'", "WARN")
    return None


def draw_agent_progress(container: st.container, statuses: Dict[str, Dict[str, str]]):
    """Renders the agent progress statuses using st.status."""
    with container:
        # container.empty() # Clearing and redrawing everything can cause flicker.
        # Instead, we will use st.status which can be updated.
        # However, st.status itself doesn't have a direct "update" method if the label changes.
        # We might need to manage individual st.empty() for each agent if fine-grained update is needed
        # or accept that this part redraws. For simplicity, let's redraw this container part.
        # A more advanced way is to have a placeholder for each agent.

        st.subheader("Agent Progress")
        num_columns = 3

        # Use st.columns within the container if it's not already a column itself.
        # If 'container' is already a column from st.columns(), then don't nest st.columns().
        # For this function, assume 'container' is a general container like st.container() or a main area.

        cols = st.columns(num_columns)
        col_idx = 0

        for ui_key, display_name in AGENT_DISPLAY_ORDER_AND_NAMES:
            status_info = statuses.get(ui_key, {"status": "pending", "details": ""}) # Default if key somehow missing
            status_val = status_info.get('status', 'pending')
            details = status_info.get('details', '')

            icon = "⚙️" if status_val == "in_progress" else \
                   ("✅" if status_val == "completed" else \
                   ("❌" if status_val == "error" else \
                   ("⏩" if status_val == "skipped" else "⏳"))) # Added skipped icon

            current_col = cols[col_idx % num_columns]
            with current_col:
                # Use st.status for a collapsible, stateful display
                # The label of st.status should be relatively static if we want to update its content.
                # If the label (icon + name + status_val) changes, it might recreate the widget.
                # Let's try making the label more static and putting dynamic parts inside.
                # Key for st.status must be unique if we want to manage them.
                # For now, let's make label dynamic.
                status_widget_label = f"{icon} **{display_name}**: {status_val}"

                # If we want to update content *inside* st.status without recreating it:
                # Create a placeholder for each agent if not already done:
                # if f"status_ph_{ui_key}" not in st.session_state:
                #    st.session_state[f"status_ph_{ui_key}"] = st.empty()
                # st.session_state[f"status_ph_{ui_key}"].status(...)
                # This is more complex. Let's stick to redrawing this section for now.

                with st.status(label=status_widget_label,
                               state="running" if status_val == "in_progress" else ("complete" if status_val == "completed" else ("error" if status_val == "error" else "complete")), # 'complete' for pending/skipped too
                               expanded=(status_val == "in_progress" or bool(details))):
                    if details:
                        st.markdown(f"_{details}_")
                    elif status_val not in ["pending", "skipped"]: # Don't show "no details" for pending/skipped
                        st.markdown("_No details yet._")
            col_idx += 1

# --- Helper Functions (from CLI or new ones for dashboard) ---
def get_analyst_choices():
    return ["Market", "Social", "News", "Fundamentals"] # Corresponds to UI display

def get_llm_provider_choices_dashboard(app_config: Dict) -> List[str]:
    providers = set()
    if app_config.get("llm_provider"): providers.add(app_config.get("llm_provider").lower())
    for key in app_config:
        if key.endswith("_config") and isinstance(app_config[key], dict):
            provider_name_from_key = key.replace("_config", "")
            if provider_name_from_key in SUPPORTED_PROVIDERS: providers.add(provider_name_from_key)
    lc_models_cfg = app_config.get('langchain_chat_models', {})
    for lc_model_def in lc_models_cfg.values():
        if isinstance(lc_model_def, dict) and 'provider' in lc_model_def: providers.add(lc_model_def['provider'].lower())

    lc_models_cfg = app_config.get('langchain_chat_models', {})
    for lc_model_def in lc_models_cfg.values():
        if isinstance(lc_model_def, dict) and 'provider' in lc_model_def:
            # Only add if it's a Gemini provider
            if lc_model_def['provider'].lower() in ["google", "gemini"]:
                providers.add(lc_model_def['provider'].lower())

    # Ensure only "google" or its aliases (like "gemini") are returned.
    # Since SUPPORTED_PROVIDERS in llm_clients should now only contain Gemini related entries.
    from tradingagents.llm_clients import SUPPORTED_PROVIDERS as ACTUAL_SUPPORTED_PROVIDERS

    # Filter by what's actually supported in the backend llm_clients
    # This ensures UI doesn't offer providers for which no client exists.
    final_provider_options = [p for p in providers if p in ACTUAL_SUPPORTED_PROVIDERS]

    # If, after filtering, list is empty, or if original 'providers' set was empty, default to "google".
    if not final_provider_options:
        # Check if 'google' is even in supported providers (it should be for Gemini-only fork)
        if "google" in ACTUAL_SUPPORTED_PROVIDERS or "gemini" in ACTUAL_SUPPORTED_PROVIDERS:
            final_provider_options = ["google"] # Default to google if it's supported
        else: # Should not happen in a correctly configured Gemini-only fork
            final_provider_options = list(ACTUAL_SUPPORTED_PROVIDERS.keys()) # Or just empty / error
            if not final_provider_options: final_provider_options = ["google"] # Absolute fallback

    return sorted(list(set(final_provider_options))) # Use set to ensure uniqueness


def get_model_choices_for_provider_dashboard(provider_name: str, app_config: Dict, role: str = "any") -> List[str]:
    models = set()
    # This fork is Gemini-only, so provider_name should always be 'google' or 'gemini'.
    # If not, we default to providing Gemini model list.
    provider_name_lower = "google" if provider_name.lower() not in ["google", "gemini"] else provider_name.lower()


    provider_specific_cfg = app_config.get(f"{provider_name}_config", {}) # Should be google_config
    default_model_for_provider = provider_specific_cfg.get("default_text_model")
    if not default_model_for_provider and provider_name == app_config.get("llm_provider","").lower():
        default_model_for_provider = app_config.get("default_text_model")
    if default_model_for_provider: models.add(default_model_for_provider)

    lc_models_cfg = app_config.get('langchain_chat_models', {})
    for lc_model_def in lc_models_cfg.values():
        if isinstance(lc_model_def, dict) and lc_model_def.get('provider', '').lower() == provider_name and lc_model_def.get('model_name'):
            models.add(lc_model_def['model_name'])

    # Add well-known Gemini models
    models.update(["gemini-1.5-flash", "gemini-1.0-pro", "gemini-1.5-pro"])
    # elif provider_name == "openai": models.update(["gpt-3.5-turbo", "gpt-4", "gpt-4-turbo", "gpt-4o"]) # REMOVED

    return sorted(list(models)) if models else ["gemini-1.5-flash"] # Fallback for Gemini


# --- Sidebar for Inputs (largely same as before) ---
with st.sidebar:
    # ... (sidebar code as before) ...
    st.header("Analysis Configuration")
    model_settings_path_input = st.text_input("Path to model_settings.yaml (optional, uses default if empty)",
                                              value=os.path.join(os.getcwd(), "model_settings.yaml"))
    if model_settings_path_input and os.path.exists(model_settings_path_input):
        initialize_config(custom_model_settings_path=model_settings_path_input)
        current_loaded_config = get_config()
        st.success(f"Loaded model settings from: {model_settings_path_input}")
    elif model_settings_path_input:
        st.warning(f"Custom model_settings.yaml not found at: {model_settings_path_input}. Using defaults.")
        initialize_config(); current_loaded_config = get_config()
    else:
        initialize_config(); current_loaded_config = get_config()

    ticker = st.text_input("Company Ticker", value=current_loaded_config.get("default_ticker", "SPY"))
    analysis_date = st.date_input("Analysis Date", value=datetime.date.today())
    selected_analysts_names_ui = st.multiselect( # Renamed to avoid clash with internal var
        "Select Analysts", options=get_analyst_choices(),
        default=current_loaded_config.get("default_analysts", get_analyst_choices()[:2])
    )
    research_depth = st.slider(
        "Research Depth (Debate Rounds)", min_value=1, max_value=5,
        value=current_loaded_config.get("max_debate_rounds", 1)
    )
    st.subheader("LLM Configuration")
    provider_options = get_llm_provider_choices_dashboard(current_loaded_config)
    default_provider_from_config = current_loaded_config.get("llm_provider", "google").lower()
    try: provider_default_idx = provider_options.index(default_provider_from_config)
    except ValueError: provider_default_idx = 0
    llm_provider_ui = st.selectbox( # Renamed to avoid clash
        "LLM Provider", options=provider_options, index=provider_default_idx
    )

    # Quick Thinker Model
    quick_instance_key_ui = next((k for k,v in current_loaded_config.get('langchain_chat_models', {}).items() if isinstance(v,dict) and v.get('provider','').lower()==llm_provider_ui and ("quick" in k.lower() or "flash" in v.get('model_name','').lower())), current_loaded_config.get('langchain_quick_thinker_key', 'quick_google_lc'))
    default_quick_model_name = current_loaded_config.get('langchain_chat_models', {}).get(quick_instance_key_ui, {}).get('model_name', current_loaded_config.get(f"{llm_provider_ui}_config", {}).get("default_text_model", current_loaded_config.get("default_text_model")))
    model_options_quick = get_model_choices_for_provider_dashboard(llm_provider_ui, current_loaded_config, "quick")
    quick_model_default_idx = model_options_quick.index(default_quick_model_name) if default_quick_model_name in model_options_quick else 0
    quick_thinker_model_ui = st.selectbox( # Renamed
        "Quick Thinker Model (Langchain Agents)", options=model_options_quick, index=quick_model_default_idx,
        help=f"Default for this role & provider: {default_quick_model_name or 'Not specified'}"
    )

    # Deep Thinker Model
    deep_instance_key_ui = next((k for k,v in current_loaded_config.get('langchain_chat_models', {}).items() if isinstance(v,dict) and v.get('provider','').lower()==llm_provider_ui and ("deep" in k.lower() or "pro" in v.get('model_name','').lower() or "gpt-4" in v.get('model_name','').lower())), current_loaded_config.get('langchain_deep_thinker_key', 'deep_google_lc'))
    default_deep_model_name = current_loaded_config.get('langchain_chat_models', {}).get(deep_instance_key_ui, {}).get('model_name', current_loaded_config.get(f"{llm_provider_ui}_config", {}).get("default_text_model", current_loaded_config.get("default_text_model")))
    model_options_deep = get_model_choices_for_provider_dashboard(llm_provider_ui, current_loaded_config, "deep")
    deep_model_default_idx = model_options_deep.index(default_deep_model_name) if default_deep_model_name in model_options_deep else 0
    deep_thinker_model_ui = st.selectbox( # Renamed
        "Deep Thinker Model (Langchain Managers/Judges)", options=model_options_deep, index=deep_model_default_idx,
        help=f"Default for this role & provider: {default_deep_model_name or 'Not specified'}"
    )
    run_button = st.button("🚀 Run Analysis", type="primary", disabled=st.session_state.analysis_running)

# --- Main Content Area (Tabs defined as before) ---
# status_placeholder and progress_display_container will be moved inside the "Run Analysis" tab

main_display_area = st.container()

with main_display_area:
    tab_run, tab_logs, tab_reports, tab_analytics = st.tabs([
        "📊 Run Analysis", "📜 Live Logs", "📑 Current/Final Report", "📈 Usage Analytics"
    ])
    with tab_run:
        status_placeholder = st.empty() # Moved here
        progress_display_container = st.container() # Moved here, for agent statuses
        st.markdown("---") # Divider
        st.caption("Live output from the currently running analysis step will appear below.")
        current_step_display_area = st.empty()

    with tab_logs:
        log_display_area = st.empty()
    with tab_reports:
        report_display_area = st.empty()
    with tab_analytics:
        # ... (analytics tab content as before) ...
        st.subheader("LLM Usage Analytics")
        st.markdown("#### Filter Usage Data")
        analytics_cols = st.columns(4)
        analytics_provider_options = ["All"] + get_llm_provider_choices_dashboard(current_loaded_config) # Use loaded config for options
        analytics_provider = analytics_cols[0].selectbox("Provider", options=analytics_provider_options, key="analytics_provider")
        analytics_model = analytics_cols[1].text_input("Model (contains)", key="analytics_model")
        analytics_agent = analytics_cols[2].text_input("Agent Name (exact)", key="analytics_agent")
        analytics_key_id = analytics_cols[3].text_input("API Key ID (contains)", key="analytics_key_id")
        analytics_date_cols = st.columns(2)
        analytics_start_date = analytics_date_cols[0].date_input("Start Date", value=None, key="analytics_start_date")
        analytics_end_date = analytics_date_cols[1].date_input("End Date", value=None, key="analytics_end_date")
        if st.button("Load Usage Data", key="load_analytics"):
            # ... (analytics data loading logic as before)
            from tradingagents.utils.cost_tracker import get_usage_summary # Moved import here
            summary_df = get_usage_summary(
                provider_filter=None if analytics_provider == "All" else analytics_provider,
                model_filter=analytics_model if analytics_model else None,
                agent_filter=analytics_agent if analytics_agent else None,
                key_filter=analytics_key_id if analytics_key_id else None,
                date_range=(
                    analytics_start_date.strftime('%Y-%m-%d') if analytics_start_date else None,
                    analytics_end_date.strftime('%Y-%m-%d') if analytics_end_date else None,
                )
            )
            if summary_df is not None and not summary_df.empty:
                st.write("### Usage Summary"); st.dataframe(summary_df)
                total_cost = summary_df['total_estimated_cost_usd'].sum()
                st.metric("Total Estimated Cost (Filtered)", f"${total_cost:,.4f}")
            elif summary_df is not None: st.info("No usage data found matching the filters.")
            else: st.error("Could not load usage data. Check logs/llm_usage.csv.")


# --- Analysis Logic ---
if run_button:
    current_step_display_area.empty() # Clear previous run's display in "Run" tab
    st.session_state.analysis_running = True
    st.session_state.current_status_text = "Starting analysis..."
    st.session_state.final_report_markdown = ""
    st.session_state.current_report_markdown = "" # Will be updated into current_step_display_area
    st.session_state.final_decision = ""
    st.session_state.app_logs.clear()

    # Initialize agent_statuses based on selected analysts and fixed agents
    st.session_state.agent_statuses = get_initial_agent_statuses(selected_analysts_names_ui)
    st.session_state.last_active_node_ui_key = None # Reset last active node

    add_log_message("Run Analysis button clicked.")
    status_placeholder.info(f"🚀 Analysis for {ticker} on {analysis_date.strftime('%Y-%m-%d')} starting...")

    # Draw initial progress (all pending/skipped)
    draw_agent_progress(progress_display_container, st.session_state.agent_statuses)

    run_config = DEFAULT_CONFIG.copy()
    initialize_config(custom_model_settings_path=model_settings_path_input if model_settings_path_input and os.path.exists(model_settings_path_input) else None)
    loaded_run_config = get_config()
    run_config.update(loaded_run_config)

    run_config["llm_provider"] = llm_provider_ui.lower() # Use UI selected provider
    run_config["max_debate_rounds"] = research_depth
    run_config["max_risk_discuss_rounds"] = research_depth

    if 'langchain_chat_models' not in run_config: run_config['langchain_chat_models'] = {}
    # Use the instance keys determined by UI logic
    # The UI selectbox for quick_thinker_model_ui now directly gives the model_name.
    # We need to find the key in langchain_chat_models that corresponds to this model_name and provider,
    # or update the run_config structure that TradingAgentsGraph expects.
    # For now, let's assume TradingAgentsGraph will use quick_thinker_model_ui and deep_thinker_model_ui
    # to select/override models for its internal Langchain quick/deep thinkers.
    # This means model_settings.yaml's langchain_quick_thinker_key / langchain_deep_thinker_key
    # are used to *find the default model names for the UI*, and then the UI's selection
    # is what's passed to the graph run.

    # We need to ensure that the run_config reflects the UI choices for quick/deep thinkers.
    # The TradingAgentsGraph __init__ looks for model names under specific keys in langchain_chat_models.
    # So, we update those specific entries.

    # Get the keys for quick/deep thinkers from config (e.g., "quick_google_lc")
    # These keys should point to definitions that include the provider selected in the UI.
    # If not, the graph init might fail or pick a wrong provider's model.
    # The UI logic for selecting quick_thinker_model_ui and deep_thinker_model_ui should ensure
    # that the model names are compatible with the selected llm_provider_ui.

    # Update the specific entries in run_config that TradingAgentGraph's Langchain setup will use.
    # Find the correct key in langchain_chat_models that corresponds to the UI's role (quick/deep) + provider.

    # Quick Thinker assignment
    cfg_quick_key_to_update = None
    for key, definition in run_config.get('langchain_chat_models', {}).items():
        if isinstance(definition, dict) and definition.get('provider','').lower() == llm_provider_ui.lower() and \
           ("quick" in key.lower() or definition.get('model_name') == quick_thinker_model_ui): # Match by current UI selection or key name
            cfg_quick_key_to_update = key
            break
    if not cfg_quick_key_to_update: # If no specific key found, use the default hint key
        cfg_quick_key_to_update = run_config.get('langchain_quick_thinker_key', 'quick_llm_lc_placeholder') # a new key if needed
        if cfg_quick_key_to_update not in run_config['langchain_chat_models']:
             run_config['langchain_chat_models'][cfg_quick_key_to_update] = {}
    run_config['langchain_chat_models'][cfg_quick_key_to_update]['model_name'] = quick_thinker_model_ui
    run_config['langchain_chat_models'][cfg_quick_key_to_update]['provider'] = llm_provider_ui.lower()


    # Deep Thinker assignment
    cfg_deep_key_to_update = None
    for key, definition in run_config.get('langchain_chat_models', {}).items():
        if isinstance(definition, dict) and definition.get('provider','').lower() == llm_provider_ui.lower() and \
            ("deep" in key.lower() or definition.get('model_name') == deep_thinker_model_ui):
            cfg_deep_key_to_update = key
            break
    if not cfg_deep_key_to_update:
        cfg_deep_key_to_update = run_config.get('langchain_deep_thinker_key', 'deep_llm_lc_placeholder')
        if cfg_deep_key_to_update not in run_config['langchain_chat_models']:
             run_config['langchain_chat_models'][cfg_deep_key_to_update] = {}
    run_config['langchain_chat_models'][cfg_deep_key_to_update]['model_name'] = deep_thinker_model_ui
    run_config['langchain_chat_models'][cfg_deep_key_to_update]['provider'] = llm_provider_ui.lower()

    add_log_message(f"Run Config - LLM Provider: {run_config['llm_provider']}")
    add_log_message(f"Run Config - Quick Thinker Model (for LC key {cfg_quick_key_to_update}): {quick_thinker_model_ui}")
    add_log_message(f"Run Config - Deep Thinker Model (for LC key {cfg_deep_key_to_update}): {deep_thinker_model_ui}")

    analyst_type_values = [name.lower() for name in selected_analysts_names_ui]
    graph = TradingAgentsGraph(selected_analysts=analyst_type_values, config=run_config, debug=True)

    with st.spinner("Analysis in progress..."):
        try:
            final_graph_state_dict = {}
            for stream_chunk_key, chunk_output_value in graph.graph.stream( # Iterate over items in the chunk
                {"company_of_interest": ticker, "trade_date": analysis_date.strftime('%Y-%m-%d'), "messages": []},
                graph.propagator.get_graph_args()
            ).items(): # .items() gives key-value pairs from the stream dict output

                # The stream_chunk_key is typically the name of the node that just produced output.
                # chunk_output_value is the output of that node.
                # The last item in the stream will be the final state under the key "__end__".

                if stream_chunk_key == "__end__":
                    final_graph_state_dict = chunk_output_value
                    # Mark all remaining "in_progress" as "completed"
                    for agent_k, data in st.session_state.agent_statuses.items():
                        if data["status"] == "in_progress":
                            st.session_state.agent_statuses[agent_k]["status"] = "completed"
                            st.session_state.agent_statuses[agent_k]["details"] = "Finished."
                    break # End of stream

                final_graph_state_dict.update({stream_chunk_key: chunk_output_value}) # Accumulate state

                # --- Log messages from the stream ---
                # This assumes node output is a dict with a "messages" key containing AIMessage etc.
                node_messages = chunk_output_value.get("messages") if isinstance(chunk_output_value, dict) else None
                if node_messages and isinstance(node_messages, list) and node_messages:
                    last_message_obj = node_messages[-1]
                    content = getattr(last_message_obj, 'content', str(last_message_obj))
                    if isinstance(content, list):
                        content = " ".join(str(p.get("text","")) for p in content if isinstance(p,dict) and p.get("type")=="text")
                    if content.strip():
                        add_log_message(f"Node '{stream_chunk_key}': {content[:200]}{'...' if len(content)>200 else ''}")
                    if hasattr(last_message_obj, 'tool_calls') and last_message_obj.tool_calls:
                        for tc in last_message_obj.tool_calls:
                            tool_name = tc.get('name') if isinstance(tc, dict) else tc.name; tool_args = tc.get('args') if isinstance(tc, dict) else tc.args
                            add_log_message(f"Node '{stream_chunk_key}' Tool Call: {tool_name}({str(tool_args)[:100]})")

                # --- Update Agent Statuses ---
                ui_key_to_update = map_stream_key_to_ui_key(stream_chunk_key)
                if ui_key_to_update and ui_key_to_update in st.session_state.agent_statuses:
                    if st.session_state.last_active_node_ui_key and st.session_state.last_active_node_ui_key != ui_key_to_update:
                        if st.session_state.agent_statuses[st.session_state.last_active_node_ui_key]["status"] == "in_progress":
                            st.session_state.agent_statuses[st.session_state.last_active_node_ui_key]["status"] = "completed"
                            st.session_state.agent_statuses[st.session_state.last_active_node_ui_key]["details"] = "Done."

                    st.session_state.agent_statuses[ui_key_to_update]["status"] = "in_progress"
                    st.session_state.agent_statuses[ui_key_to_update]["details"] = "Processing..."
                    st.session_state.last_active_node_ui_key = ui_key_to_update

                # --- Update Current Report Snippet ---
                latest_report_key, latest_report_content = None, None
                # ... (logic for finding latest_report_key & content from final_graph_state_dict - as before) ...
                investment_judge_decision = final_graph_state_dict.get("investment_debate_state", {}).get("judge_decision")
                risk_judge_decision = final_graph_state_dict.get("risk_debate_state", {}).get("judge_decision")
                report_keys_to_check = ["market_report","sentiment_report","news_report","fundamentals_report","investment_plan","trader_investment_plan","final_trade_decision"]
                if risk_judge_decision: latest_report_key, latest_report_content = "Final Decision (Portfolio Mgr)", risk_judge_decision
                elif final_graph_state_dict.get("final_trade_decision"): latest_report_key, latest_report_content = "Final Trade Decision", final_graph_state_dict["final_trade_decision"]
                elif final_graph_state_dict.get("trader_investment_plan"): latest_report_key, latest_report_content = "Trading Team Plan", final_graph_state_dict["trader_investment_plan"]
                elif investment_judge_decision: latest_report_key, latest_report_content = "Research Team Decision", investment_judge_decision
                elif final_graph_state_dict.get("investment_plan"): latest_report_key, latest_report_content = "Research Team Plan", final_graph_state_dict["investment_plan"]
                else:
                    for r_key in reversed(report_keys_to_check[:4]):
                        if final_graph_state_dict.get(r_key): latest_report_key, latest_report_content = r_key.replace('_',' ').title(), final_graph_state_dict[r_key]; break

                if latest_report_key and latest_report_content:
                    st.session_state.current_report_markdown = f"### Latest Output from: {latest_report_key}\n\n{latest_report_content}"
                    current_step_display_area.markdown(st.session_state.current_report_markdown)

                # --- Rerender UI parts ---
                # Use a consistent key for the log text_area for smoother updates
                log_display_area.text_area("Logs", "\n".join(list(st.session_state.app_logs)), height=400, key="live_log_feed_area", disabled=True)
                draw_agent_progress(progress_display_container, st.session_state.agent_statuses)
                status_placeholder.info(f"🏃 Running: {ui_key_to_update or stream_chunk_key}...")

            # --- After stream finishes (now handled by __end__ key) ---
            if final_graph_state_dict: # Ensure we have the final state
                # Mark the very last active node as completed if it was in_progress
                if st.session_state.last_active_node_ui_key and \
                   st.session_state.agent_statuses.get(st.session_state.last_active_node_ui_key, {}).get("status") == "in_progress":
                    st.session_state.agent_statuses[st.session_state.last_active_node_ui_key]["status"] = "completed"
                    st.session_state.agent_statuses[st.session_state.last_active_node_ui_key]["details"] = "Finished."
                draw_agent_progress(progress_display_container, st.session_state.agent_statuses) # Final progress draw

                st.session_state.current_status_text = "Processing final report..."
                status_placeholder.success("✅ Analysis complete! Final report generated.")

                final_report_parts = [] # ... (construct final_report_parts as before) ...
                if final_graph_state_dict.get("market_report"): final_report_parts.append(f"## Market Analysis\n{final_graph_state_dict['market_report']}")
                if final_graph_state_dict.get("sentiment_report"): final_report_parts.append(f"## Social Sentiment\n{final_graph_state_dict['sentiment_report']}")
                if final_graph_state_dict.get("news_report"): final_report_parts.append(f"## News Analysis\n{final_graph_state_dict['news_report']}")
                if final_graph_state_dict.get("fundamentals_report"): final_report_parts.append(f"## Fundamentals Analysis\n{final_graph_state_dict['fundamentals_report']}")
                investment_judge_decision = final_graph_state_dict.get("investment_debate_state", {}).get("judge_decision")
                if investment_judge_decision: final_report_parts.append(f"## Research Team Decision\n{investment_judge_decision}")
                elif final_graph_state_dict.get("investment_plan"): final_report_parts.append(f"## Research Team Plan\n{final_graph_state_dict['investment_plan']}")
                if final_graph_state_dict.get("trader_investment_plan"): final_report_parts.append(f"## Trading Team Plan\n{final_graph_state_dict['trader_investment_plan']}")
                risk_judge_decision = final_graph_state_dict.get("risk_debate_state", {}).get("judge_decision")
                if risk_judge_decision:
                     final_report_parts.append(f"## Final Trade Decision (Portfolio Manager)\n{risk_judge_decision}")
                     st.session_state.final_decision = graph.signal_processor.process_signal(risk_judge_decision)
                elif final_graph_state_dict.get("final_trade_decision"):
                     final_report_parts.append(f"## Final Trade Decision\n{final_graph_state_dict['final_trade_decision']}")
                     st.session_state.final_decision = graph.signal_processor.process_signal(final_graph_state_dict['final_trade_decision'])

                st.session_state.final_report_markdown = "\n\n---\n\n".join(final_report_parts)
                report_display_area.markdown(st.session_state.final_report_markdown) # Display in "Reports" tab
                current_step_display_area.empty() # Clear the "Run Analysis" tab's current step area

                if st.session_state.final_decision:
                    st.sidebar.subheader("Final Decision:")
                    st.sidebar.success(f"**{st.session_state.final_decision}** for {ticker} on {analysis_date.strftime('%Y-%m-%d')}")

                # Add download button for the final report in the Reports Tab
                with tab_reports: # Ensure this is drawn within the correct tab context
                    if st.session_state.final_report_markdown: # Check again in case of race condition or clearing
                        st.download_button(
                            label="📥 Download Full Report",
                            data=st.session_state.final_report_markdown,
                            file_name=f"trading_analysis_report_{ticker}_{analysis_date.strftime('%Y%m%d')}.md",
                            mime="text/markdown"
                        )
            else:
                status_placeholder.warning("Analysis finished, but no final state was captured from the stream.")

        except Exception as e:
            st.session_state.current_status_text = f"Error during analysis: {e}"
            status_placeholder.error(f"❌ Error during analysis: {e}")
            add_log_message(f"ERROR: {e}", level="ERROR")
            st.exception(e)
        finally:
            st.session_state.analysis_running = False
            st.experimental_rerun() # Changed to experimental_rerun


# Final status update (if not rerun by button logic)
if not st.session_state.analysis_running and 'current_status_text' in st.session_state: # check key exists
    status_placeholder.info(st.session_state.current_status_text)

# Initial drawing of agent progress if statuses exist in session state
# This will be cleared and redrawn when analysis starts by draw_agent_progress call.
if 'agent_statuses' in st.session_state and st.session_state.agent_statuses:
    draw_agent_progress(progress_display_container, st.session_state.agent_statuses)
else: # Ensure progress_display_container is cleared or has a default message if no statuses yet
    with progress_display_container:
        progress_display_container.empty() # Clear it initially
        # st.info("Agent progress will appear here once analysis starts.")


# To run: streamlit run dashboard_app.py

[end of dashboard_app.py]
