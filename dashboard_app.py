import streamlit as st
import datetime
import os
import logging
from collections import deque

# Import necessary components from tradingagents
# It's assumed these modules and their functions are structured to be importable.
# We might need to adjust PYTHONPATH or how these are imported if running Streamlit
# from a different context than the main CLI.
try:
    from tradingagents.dataflows.config import get_config, initialize_config, set_config as set_global_config
    from tradingagents.default_config import DEFAULT_CONFIG
    from tradingagents.graph.trading_graph import TradingAgentsGraph
    # For enums or lists of choices if defined elsewhere
    # from cli.models import AnalystType
    # from cli.utils import get_llm_provider_choices, get_model_choices_for_provider
except ImportError as e:
    st.error(f"Failed to import TradingAgents modules. Ensure PYTHONPATH is set correctly or run Streamlit from project root. Error: {e}")
    # Add placeholder functions if imports fail, so app can still partially load for UI dev
    def get_config(): return DEFAULT_CONFIG if 'DEFAULT_CONFIG' in globals() else {}
    def initialize_config(custom_model_settings_path=None): pass
    def set_global_config(new_config_values): pass
    class TradingAgentsGraph:
        def __init__(self, *args, **kwargs): pass
        def propagate(self, *args, **kwargs):
            yield {"messages": ["TradingAgentsGraph not loaded due to import error."]}
            return {}, "ERROR"


# --- App Configuration ---
st.set_page_config(layout="wide", page_title="TradingAgents Dashboard")

# --- Logging Configuration for Streamlit ---
# Option 1: Custom Handler to display logs in Streamlit
class StreamlitLogHandler(logging.Handler):
    def __init__(self, container, max_logs=100):
        super().__init__()
        self.container = container
        self.log_queue = deque(maxlen=max_logs)

    def emit(self, record):
        log_entry = self.format(record)
        self.log_queue.append(log_entry)
        # Update the container - needs to be thread-safe if graph runs in thread
        # For simplicity, we'll update directly. If using threads, use st.experimental_rerun or a queue.
        log_text = "\n".join(list(self.log_queue))
        self.container.text_area("Live Logs", value=log_text, height=300, key="live_logs_textarea", disabled=True)

# Option 2: Using st.session_state to store logs (simpler for single-threaded updates)
if 'app_logs' not in st.session_state:
    st.session_state.app_logs = deque(maxlen=200) # Store last 200 log messages

def add_log_message(message, level="INFO"):
    timestamp = datetime.datetime.now().strftime("%H:%M:%S")
    st.session_state.app_logs.append(f"[{timestamp}][{level}] {message}")

# Configure root logger or specific loggers if needed
# For now, we'll primarily use add_log_message and print for debugging.
# If TradingAgents modules use standard logging, we can add a StreamlitLogHandler to them.
# For example:
# base_client_logger = logging.getLogger('tradingagents.llm_clients.base_client')
# if not any(isinstance(h, StreamlitLogHandler) for h in base_client_logger.handlers):
#    base_client_logger.addHandler(StreamlitLogHandler(log_placeholder_container)) # Define container later


# --- Helper Functions (from CLI or new ones for dashboard) ---
def get_analyst_choices():
    # Placeholder - ideally, this comes from an enum or shared config
    return ["Market", "Social", "News", "Fundamentals"]

def get_llm_provider_choices_dashboard():
    # Placeholder
    return ["google", "openai", "anthropic", "ollama", "openrouter"]

def get_model_choices_for_provider_dashboard(provider_name, config, role="quick"):
    # Placeholder - this needs to be more dynamic based on model_settings.yaml
    # For now, let's assume it uses the defaults or what's in config
    if provider_name == "google":
        return [config.get("default_text_model", "gemini-1.5-flash"), "gemini-1.0-pro", "gemini-1.5-pro"]
    elif provider_name == "openai":
        # This should also list models available via openai_base_url if configured
        return [config.get("default_text_model", "gpt-3.5-turbo"), "gpt-4", "gpt-4-turbo"]
    return [config.get("default_text_model", "default-model")]


# --- Session State Initialization ---
if 'analysis_running' not in st.session_state:
    st.session_state.analysis_running = False
if 'current_status_text' not in st.session_state:
    st.session_state.current_status_text = "Ready to start."
if 'agent_statuses' not in st.session_state:
    st.session_state.agent_statuses = {} # Example: {"Market Analyst": "pending", ...}
if 'current_report_markdown' not in st.session_state:
    st.session_state.current_report_markdown = ""
if 'final_report_markdown' not in st.session_state:
    st.session_state.final_report_markdown = ""
if 'final_decision' not in st.session_state:
    st.session_state.final_decision = ""


# --- Main App Layout ---
st.title("📈 TradingAgents Dashboard")

# Load initial base config (without model_settings.yaml initially, or let it load by default)
# This ensures that some config is available for UI elements even before full initialization.
# The full initialization with model_settings happens when "Run Analysis" is clicked.
current_loaded_config = get_config()


# --- Sidebar for Inputs ---
with st.sidebar:
    st.header("Analysis Configuration")

    # Attempt to load model_settings.yaml to populate choices if available
    # This is a bit of a chicken-and-egg, but useful for defaults in UI
    # The actual config for the run will be re-initialized
    model_settings_path_input = st.text_input("Path to model_settings.yaml (optional, uses default if empty)",
                                              value=os.path.join(os.getcwd(), "model_settings.yaml"))

    if model_settings_path_input and os.path.exists(model_settings_path_input):
        initialize_config(custom_model_settings_path=model_settings_path_input)
        current_loaded_config = get_config() # Reload with model settings for UI defaults
        st.success(f"Loaded model settings from: {model_settings_path_input}")
    elif model_settings_path_input: # Path given but not found
        st.warning(f"Custom model_settings.yaml not found at: {model_settings_path_input}. Using defaults.")
        initialize_config() # Initialize with default search for model_settings.yaml
        current_loaded_config = get_config()
    else: # No path given, initialize with default search
        initialize_config()
        current_loaded_config = get_config()


    ticker = st.text_input("Company Ticker", value=current_loaded_config.get("default_ticker", "SPY"))
    analysis_date = st.date_input("Analysis Date", value=datetime.date.today())

    selected_analysts_names = st.multiselect(
        "Select Analysts",
        options=get_analyst_choices(),
        default=current_loaded_config.get("default_analysts", get_analyst_choices()[:2]) # Default to first two
    )

    research_depth = st.slider(
        "Research Depth (Debate Rounds)",
        min_value=1, max_value=5,
        value=current_loaded_config.get("max_debate_rounds", 1)
    )

    st.subheader("LLM Configuration")
    llm_provider = st.selectbox(
        "LLM Provider",
        options=get_llm_provider_choices_dashboard(),
        index=get_llm_provider_choices_dashboard().index(current_loaded_config.get("llm_provider", "google"))
    )

    # Quick Thinker (Langchain)
    lc_quick_key = current_loaded_config.get('langchain_quick_thinker_key', 'quick_llm_lc')
    default_quick_model = current_loaded_config.get('langchain_chat_models', {}).get(lc_quick_key, {}).get('model_name', current_loaded_config.get("default_text_model"))
    quick_thinker_model = st.selectbox(
        "Quick Thinker Model (for Langchain Agents)",
        options=get_model_choices_for_provider_dashboard(llm_provider, current_loaded_config, "quick"),
        index=0, # Default to first option, or try to find default_quick_model in list
        help=f"Model for most Langchain agents. Current default from settings: {default_quick_model}"
    )

    # Deep Thinker (Langchain)
    lc_deep_key = current_loaded_config.get('langchain_deep_thinker_key', 'deep_llm_lc')
    default_deep_model = current_loaded_config.get('langchain_chat_models', {}).get(lc_deep_key, {}).get('model_name', current_loaded_config.get("default_text_model"))
    deep_thinker_model = st.selectbox(
        "Deep Thinker Model (for Langchain Managers/Judges)",
        options=get_model_choices_for_provider_dashboard(llm_provider, current_loaded_config, "deep"),
        index=0, # Default to first option, or try to find default_deep_model in list
        help=f"Model for manager/judge Langchain agents. Current default from settings: {default_deep_model}"
    )

    # Models for BaseLLMClient components (Reflector, SignalProcessor, InterfaceFunctions)
    # These can be further customized if an "Advanced Settings" expander is added.
    # For now, they'll use what's in model_settings.yaml under 'agent_model_configs' for those components,
    # or the 'default_text_model' of the provider.

    run_button = st.button("🚀 Run Analysis", type="primary", disabled=st.session_state.analysis_running)


from tradingagents.utils.cost_tracker import get_usage_summary # Import cost_tracker

# --- Main Content Area ---
status_placeholder = st.empty()
progress_area = st.container() # For agent statuses
main_display_area = st.container() # For tabs

with main_display_area:
    tab_run, tab_logs, tab_reports, tab_analytics = st.tabs([
        "📊 Run Analysis", "📜 Live Logs", "📑 Current/Final Report", "📈 Usage Analytics"
    ])

    # Tab: Run Analysis (to contain current report parts during run) - or integrate into Reports tab better
    with tab_run:
        # This tab could show the "Current Report Snippet" while analysis is running
        # and then switch to "Final Report" in the other tab.
        # For now, current_report_markdown is shown in tab_reports.
        # This tab can be a placeholder or used for more dynamic "current step" info.
        st.caption("Analysis progress and current outputs will be shown here or in the 'Current/Final Report' tab.")
        current_step_display_area = st.empty()


    with tab_logs:
        log_display_area = st.empty()
        log_display_area.text_area("Logs", "".join(st.session_state.app_logs), height=400, key="log_text_area_main", disabled=True)

    with tab_reports:
        report_display_area = st.empty()
        if st.session_state.final_report_markdown:
            report_display_area.markdown(st.session_state.final_report_markdown)
        elif st.session_state.current_report_markdown:
            # Display current report in the "Run Analysis" tab for live updates during run
            with tab_run: # This ensures it updates the correct tab
                 current_step_display_area.markdown(st.session_state.current_report_markdown)
            report_display_area.info("Analysis in progress... Current updates in 'Run Analysis' tab. Final report will appear here.")
        else:
            report_display_area.info("Reports will appear here once the analysis starts.")

    with tab_analytics:
        st.subheader("LLM Usage Analytics")

        # Add filters for analytics
        st.markdown("#### Filter Usage Data")
        analytics_cols = st.columns(4)
        analytics_provider = analytics_cols[0].selectbox("Provider", options=["All"] + get_llm_provider_choices_dashboard(), key="analytics_provider")
        # TODO: Populate model choices based on selected provider and available models in CSV
        analytics_model = analytics_cols[1].text_input("Model (contains)", key="analytics_model")
        analytics_agent = analytics_cols[2].text_input("Agent Name (exact)", key="analytics_agent")
        analytics_key_id = analytics_cols[3].text_input("API Key ID (contains)", key="analytics_key_id")

        # Date range for analytics
        analytics_date_cols = st.columns(2)
        analytics_start_date = analytics_date_cols[0].date_input("Start Date", value=None, key="analytics_start_date")
        analytics_end_date = analytics_date_cols[1].date_input("End Date", value=None, key="analytics_end_date")

        if st.button("Load Usage Data", key="load_analytics"):
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
                st.write("### Usage Summary")
                st.dataframe(summary_df)

                # Simple total cost display
                total_cost = summary_df['total_estimated_cost_usd'].sum()
                st.metric("Total Estimated Cost (Filtered)", f"${total_cost:,.4f}")

            elif summary_df is not None and summary_df.empty:
                st.info("No usage data found matching the filters.")
            else:
                st.error("Could not load usage data. Check logs/llm_usage.csv.")


# --- Analysis Logic ---
if run_button:
    # Clear previous run's "current" display in the run tab
    with tab_run:
        current_step_display_area.empty()

    st.session_state.analysis_running = True
    st.session_state.current_status_text = "Starting analysis..."
    st.session_state.final_report_markdown = ""
    st.session_state.current_report_markdown = ""
    st.session_state.final_decision = ""
    st.session_state.app_logs.clear()
    st.session_state.agent_statuses = {name + " Analyst": "pending" for name in selected_analysts_names}
    # Add other agent types to status
    st.session_state.agent_statuses.update({
        "Bull Researcher": "pending", "Bear Researcher": "pending", "Research Manager": "pending",
        "Trader": "pending",
        "RiskyDebator": "pending", "NeutralDebator": "pending", "SafeDebator": "pending", "RiskManager": "pending",
        # Reflector, SignalProcessor, InterfaceFunctions don't have explicit "agent" status in this UI version
    })

    add_log_message("Run Analysis button clicked.")
    status_placeholder.info(f"🚀 Analysis for {ticker} on {analysis_date.strftime('%Y-%m-%d')} starting...")

    # Prepare config for TradingAgentsGraph
    run_config = DEFAULT_CONFIG.copy() # Start with application defaults

    # Merge model_settings.yaml specified by user (or default loaded one)
    # Re-initialize config to ensure the correct model_settings.yaml is used for this run
    initialize_config(custom_model_settings_path=model_settings_path_input if model_settings_path_input and os.path.exists(model_settings_path_input) else None)
    loaded_run_config = get_config() # This now has model_settings.yaml merged
    run_config.update(loaded_run_config) # Ensure all keys from model_settings are present

    # Override with UI selections
    run_config["llm_provider"] = llm_provider.lower()
    run_config["max_debate_rounds"] = research_depth
    run_config["max_risk_discuss_rounds"] = research_depth

    # Update Langchain model names in the run_config based on UI selection
    # These keys ('langchain_quick_thinker_key', 'langchain_deep_thinker_key') are used by TradingAgentsGraph
    # to find the actual model names within run_config['langchain_chat_models']
    if 'langchain_chat_models' not in run_config: run_config['langchain_chat_models'] = {}

    quick_lc_key = run_config.get('langchain_quick_thinker_key', 'quick_llm_lc')
    if quick_lc_key not in run_config['langchain_chat_models']: run_config['langchain_chat_models'][quick_lc_key] = {}
    run_config['langchain_chat_models'][quick_lc_key]['model_name'] = quick_thinker_model

    deep_lc_key = run_config.get('langchain_deep_thinker_key', 'deep_llm_lc')
    if deep_lc_key not in run_config['langchain_chat_models']: run_config['langchain_chat_models'][deep_lc_key] = {}
    run_config['langchain_chat_models'][deep_lc_key]['model_name'] = deep_thinker_model

    # The BaseLLMClient will pick up 'default_text_model' from run_config (which should be from model_settings.yaml)
    # If specific components (Reflector, SignalProcessor, InterfaceFunctions) need to use a model different
    # from 'default_text_model', that should be set in model_settings.yaml under their 'agent_model_configs'
    # and TradingAgentsGraph will pass that down.

    add_log_message(f"Effective LLM Provider for run: {run_config['llm_provider']}")
    add_log_message(f"Effective Quick Thinker LC Model: {quick_thinker_model}")
    add_log_message(f"Effective Deep Thinker LC Model: {deep_thinker_model}")
    # Add more logs for other effective model settings if needed (e.g., for Reflector)

    # Initialize Graph
    # The list of selected analysts needs to be just the type string, e.g., "market", "news"
    analyst_type_values = [name.lower() for name in selected_analysts_names]

    graph = TradingAgentsGraph(
        selected_analysts=analyst_type_values,
        config=run_config,
        debug=True # Enable debug to get more verbose output if necessary
    )

    # This is a simplified way to update UI from graph stream.
    # For more complex updates or if graph runs in a thread, a queue + st.experimental_rerun is better.
    temp_current_report_parts = []

    with st.spinner("Analysis in progress... Please wait."):
        try:
            # Stream processing
            final_graph_state_dict = {}
            for chunk_dict in graph.graph.stream(
                {"company_of_interest": ticker, "trade_date": analysis_date.strftime("%Y-%m-%d"), "messages": []},
                graph.propagator.get_graph_args() # Get args like recursion_limit
            ):
                final_graph_state_dict = chunk_dict # Keep the latest state

                # --- Log messages from the stream ---
                # The structure of 'chunk_dict' and its 'messages' needs to be understood from LangGraph's output.
                # It's usually a dict where keys are node names and values are their outputs.
                # We need to find the actual human-readable messages or tool calls.
                for node_name, node_output in chunk_dict.items():
                    if isinstance(node_output, dict) and "messages" in node_output and node_output["messages"]:
                        last_message_obj = node_output["messages"][-1]
                        # Assuming last_message_obj has a 'content' attribute (like AIMessage, HumanMessage)
                        if hasattr(last_message_obj, 'content'):
                            content = str(last_message_obj.content)
                            if content.strip(): # Add if not empty
                                add_log_message(f"Node '{node_name}': {content[:200]}{'...' if len(content)>200 else ''}")
                        # Handle tool calls if present
                        if hasattr(last_message_obj, 'tool_calls') and last_message_obj.tool_calls:
                            for tc in last_message_obj.tool_calls:
                                add_log_message(f"Node '{node_name}' Tool Call: {tc.get('name')}({str(tc.get('args', {}))[:100]})")

                    # --- Update Agent Statuses (Simplified) ---
                    # This requires mapping node names from the graph stream to your UI agent names.
                    # This is a very basic example; a more robust mapping would be needed.
                    agent_ui_name = node_name # Or a mapped name
                    if agent_ui_name in st.session_state.agent_statuses:
                        st.session_state.agent_statuses[agent_ui_name] = "in_progress"
                        # Mark previous ones as completed if logic allows. This is tricky without knowing graph flow.

                # --- Update Current Report Snippet (Simplified) ---
                # Look for known report keys in the chunk_dict (the latest state)
                report_keys_to_check = [
                    "market_report", "sentiment_report", "news_report", "fundamentals_report",
                    "investment_plan", # from research manager
                    "trader_investment_plan",
                    "final_trade_decision" # from risk manager
                ]
                new_report_part_found = False
                for key in report_keys_to_check:
                    if key in final_graph_state_dict and final_graph_state_dict[key] and isinstance(final_graph_state_dict[key], str):
                        # Display only the latest significant report part to avoid too much text
                        st.session_state.current_report_markdown = f"### Latest Update: {key.replace('_', ' ').title()}\n\n{final_graph_state_dict[key]}"
                        new_report_part_found = True
                        break # Show one at a time for "current"

                # --- Rerender UI parts that show dynamic data ---
                log_display_area.text_area("Logs", "\n".join(st.session_state.app_logs), height=400, key="log_text_area_main_update", disabled=True)
                if new_report_part_found:
                    report_display_area.markdown(st.session_state.current_report_markdown)

                # Update progress_area (this needs a function to draw agent statuses)
                # draw_agent_progress(progress_area, st.session_state.agent_statuses) # Placeholder for drawing function

            # After stream finishes, process the final state
            st.session_state.current_status_text = "Processing final report..."
            status_placeholder.success("✅ Analysis complete! Processing final report...")

            # Construct final report from final_graph_state_dict
            # This is a simplified version; refer to cli/main.py display_complete_report for a more detailed structure
            final_report_parts = []
            if final_graph_state_dict.get("market_report"): final_report_parts.append(f"## Market Analysis\n{final_graph_state_dict['market_report']}")
            if final_graph_state_dict.get("sentiment_report"): final_report_parts.append(f"## Social Sentiment\n{final_graph_state_dict['sentiment_report']}")
            if final_graph_state_dict.get("news_report"): final_report_parts.append(f"## News Analysis\n{final_graph_state_dict['news_report']}")
            if final_graph_state_dict.get("fundamentals_report"): final_report_parts.append(f"## Fundamentals Analysis\n{final_graph_state_dict['fundamentals_report']}")

            if final_graph_state_dict.get("investment_debate_state", {}).get("judge_decision"):
                final_report_parts.append(f"## Research Team Decision\n{final_graph_state_dict['investment_debate_state']['judge_decision']}")
            elif final_graph_state_dict.get("investment_plan"): # Fallback if judge_decision not directly there
                 final_report_parts.append(f"## Research Team Plan\n{final_graph_state_dict['investment_plan']}")

            if final_graph_state_dict.get("trader_investment_plan"): final_report_parts.append(f"## Trading Team Plan\n{final_graph_state_dict['trader_investment_plan']}")

            if final_graph_state_dict.get("risk_debate_state", {}).get("judge_decision"):
                 final_report_parts.append(f"## Final Trade Decision (Portfolio Manager)\n{final_graph_state_dict['risk_debate_state']['judge_decision']}")
                 st.session_state.final_decision = graph.signal_processor.process_signal(final_graph_state_dict['risk_debate_state']['judge_decision'])
            elif final_graph_state_dict.get("final_trade_decision"): # Fallback
                 final_report_parts.append(f"## Final Trade Decision\n{final_graph_state_dict['final_trade_decision']}")
                 st.session_state.final_decision = graph.signal_processor.process_signal(final_graph_state_dict['final_trade_decision'])


            st.session_state.final_report_markdown = "\n\n---\n\n".join(final_report_parts)
            report_display_area.markdown(st.session_state.final_report_markdown)

            if st.session_state.final_decision:
                st.sidebar.subheader("Final Decision:")
                st.sidebar.success(f"**{st.session_state.final_decision}** for {ticker} on {analysis_date.strftime('%Y-%m-%d')}")


        except Exception as e:
            st.session_state.current_status_text = f"Error during analysis: {e}"
            status_placeholder.error(f"❌ Error during analysis: {e}")
            add_log_message(f"ERROR: {e}", level="ERROR")
            st.exception(e)
        finally:
            st.session_state.analysis_running = False
            # Rerun to update button state etc.
            st.experimental_rerun()


# Final status update (if not rerun by button logic)
if not st.session_state.analysis_running:
    status_placeholder.info(st.session_state.current_status_text)

# Placeholder for drawing agent progress dynamically
# def draw_agent_progress(container, statuses):
#    with container:
#        for agent_name, status in statuses.items():
#            icon = "⚙️" if status == "in_progress" else ("✅" if status == "completed" else "⏳")
#            st.markdown(f"{icon} **{agent_name}**: {status}")

# Initial drawing of agent progress if statuses exist
if st.session_state.agent_statuses:
    # draw_agent_progress(progress_area, st.session_state.agent_statuses)
    with progress_area:
        st.subheader("Agent Progress")
        cols = st.columns(3)
        col_idx = 0
        for agent_name, status_val in st.session_state.agent_statuses.items():
            icon = "⚙️" if status_val == "in_progress" else ("✅" if status_val == "completed" else ("❌" if status_val == "error" else "⏳"))
            cols[col_idx % 3].markdown(f"{icon} **{agent_name}**: {status_val}")
            col_idx +=1

# To run: streamlit run dashboard_app.py
