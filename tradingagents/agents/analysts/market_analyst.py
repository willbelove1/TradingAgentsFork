from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
import time
import json
from typing import Dict, Any # For type hinting

def create_market_analyst(llm, toolkit, prompt_config: Dict[str, Any]): # Added prompt_config

    def market_analyst_node(state):
        current_date = state["trade_date"]
        ticker = state["company_of_interest"]
        company_name = state["company_of_interest"]

        if toolkit.config["online_tools"]:
            tools = [
                toolkit.get_YFin_data_online,
                toolkit.get_stockstats_indicators_report_online,
            ]
        else:
            tools = [
                toolkit.get_YFin_data,
                toolkit.get_stockstats_indicators_report,
            ]

        # Get system message from prompt_config, or use a default if not found
        # The prompt_config should have a 'system_message' key for Langchain type prompts
        system_message_from_config = prompt_config.get(
            "system_message",
            "You are a Market Analyst. Analyze the market for {ticker} on {current_date} using provided tools. Provide a detailed report." # Basic fallback
        )
        # The system_message_from_config might already contain placeholders like {ticker}, {current_date}
        # Langchain's ChatPromptTemplate will fill these if they are part of the main system message string.

        # The generic part of the system prompt for Langchain agents (tool usage, collaboration)
        # can be kept, and the specific agent's role/task (from prompt_config) appended or prepended.
        langchain_system_wrapper = (
            "You are a helpful AI assistant, collaborating with other assistants."
            " Use the provided tools to progress towards answering the question."
            " If you are unable to fully answer, that's OK; another assistant with different tools"
            " will help where you left off. Execute what you can to make progress."
            " If you or any other assistant has the FINAL TRANSACTION PROPOSAL: **BUY/HOLD/SELL** or deliverable,"
            " prefix your response with FINAL TRANSACTION PROPOSAL: **BUY/HOLD/SELL** so the team knows to stop."
            " You have access to the following tools: {tool_names}.\n"
            # Specific instructions for this agent, loaded from prompt_config:
            "--- AGENT TASK START ---\n"
            "{agent_specific_system_message}\n"
            "--- AGENT TASK END ---\n"
            "For your reference, the current date is {current_date}. The company we want to look at is {ticker}."
        )

        # The 'system_message' from prompt_config is the 'agent_specific_system_message'
        final_system_prompt_template = langchain_system_wrapper.replace(
            "{agent_specific_system_message}", system_message_from_config
        )

        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", final_system_prompt_template),
                MessagesPlaceholder(variable_name="messages"),
            ]
        )

        # No need to partial system_message here if it's fully part of the template string above.
        # prompt = prompt.partial(system_message=system_message_from_config)
        prompt = prompt.partial(tool_names=", ".join([tool.name for tool in tools]))
        prompt = prompt.partial(current_date=current_date)
        prompt = prompt.partial(ticker=ticker)

        chain = prompt | llm.bind_tools(tools)

        result = chain.invoke(state["messages"])

        report = ""

        if len(result.tool_calls) == 0:
            report = result.content
       
        return {
            "messages": [result],
            "market_report": report,
        }

    return market_analyst_node
