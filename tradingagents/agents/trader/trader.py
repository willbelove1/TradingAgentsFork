import functools
import time
import json
from typing import Dict, Any, Optional

from tradingagents.llm_clients import BaseLLMClient # For direct LLM calls
from tradingagents.config.prompt_loader import format_prompt
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage


import logging
logger = logging.getLogger(__name__)

def create_trader(
    llm_lc, # Langchain ChatModel for the final trading decision based on full context
    memory,
    main_prompt_config: Optional[Dict[str, Any]],
    # For direct LLM calls for evaluation/clarification steps:
    llm_client_for_direct_calls: Optional[BaseLLMClient] = None,
    evaluation_prompt_config: Optional[Dict[str, Any]] = None,
    clarification_request_prompt_config: Optional[Dict[str, Any]] = None,
    max_consultation_rounds: int = 1,
    direct_call_component_config: Optional[Dict[str, Any]] = None # For model, temp of direct calls
):
    """
    Creates the trader agent node with consultation capabilities.
    If llm_client_for_direct_calls and its associated prompts are not provided,
    the trader will skip the evaluation/clarification step and proceed directly.
    """

    def trader_node(state: Dict[str, Any], name: str = "Trader") -> Dict[str, Any]:
        current_date = state["trade_date"]
        ticker = state["company_of_interest"]
        investment_plan_from_rm = state["investment_plan"] # From ResearchManager

        # --- Consultation Logic ---
        needs_clarification = False
        clarification_request_content = None
        consultation_round = state.get('tr_consultation_round', 0)

        # Store the initial plan if this is the first time Trader sees it or no consultation is active
        if not state.get('tr_consultation_active') or not state.get('initial_investment_plan_for_consultation'):
            state['initial_investment_plan_for_consultation'] = investment_plan_from_rm
            # Reset consultation round for a new plan
            if state.get('initial_investment_plan_for_consultation') != investment_plan_from_rm:
                 consultation_round = 0
                 state['tr_clarification_response'] = None # Clear old response if plan changed


        # Only evaluate and ask for clarification if:
        # 1. We have the necessary LLM client and prompts for evaluation.
        # 2. We haven't received a clarification response yet for the current round OR it's the first look.
        # 3. We are within the allowed consultation rounds.
        if llm_client_for_direct_calls and evaluation_prompt_config and clarification_request_prompt_config and \
           (not state.get('tr_clarification_response') or consultation_round == 0) and \
           consultation_round < max_consultation_rounds:

            logger.info(f"Trader: Round {consultation_round + 1}. Evaluating RM plan for {ticker} on {current_date}.")

            # Step 1: Trader evaluates the plan from ResearchManager
            eval_context_vars = {"investment_plan_from_rm": investment_plan_from_rm}
            eval_prompt_str = format_prompt(evaluation_prompt_config, eval_context_vars)

            eval_model = direct_call_component_config.get('model', llm_client_for_direct_calls.model_name)
            eval_temp = direct_call_component_config.get('temperature', 0.3) # Lower temp for structured output

            if not eval_prompt_str.strip():
                logger.error("Trader: Evaluation prompt is empty. Skipping evaluation.")
            else:
                try:
                    evaluation_response_str = llm_client_for_direct_calls.generate_text(
                        prompt=eval_prompt_str,
                        model=eval_model,
                        temperature=eval_temp,
                        max_tokens=direct_call_component_config.get('max_tokens', 500),
                        agent_name=f"{name}_Evaluator"
                    )
                    logger.debug(f"Trader evaluation response: {evaluation_response_str}")
                    eval_data = json.loads(evaluation_response_str)

                    if eval_data.get("needs_clarification", False):
                        needs_clarification = True
                        clarification_points = eval_data.get("clarification_points", "General clarification needed.")

                        # Step 2: If clarification needed, Trader forms a question
                        logger.info(f"Trader: Clarification needed for '{clarification_points}'. Forming request.")
                        clar_req_context_vars = {
                            "investment_plan_from_rm": investment_plan_from_rm,
                            "clarification_points": clarification_points
                        }
                        clar_req_prompt_str = format_prompt(clarification_request_prompt_config, clar_req_context_vars)

                        if not clar_req_prompt_str.strip():
                             logger.error("Trader: Clarification request prompt is empty. Cannot ask.")
                             needs_clarification = False # Cannot proceed with this path
                        else:
                            clarification_request_content = llm_client_for_direct_calls.generate_text(
                                prompt=clar_req_prompt_str,
                                model=eval_model, # Can use same model or a different one
                                temperature=direct_call_component_config.get('temperature_clarification', 0.5),
                                max_tokens=direct_call_component_config.get('max_tokens_clarification', 500),
                                agent_name=f"{name}_ClarificationRequester"
                            )
                            logger.info(f"Trader: Generated clarification request: {clarification_request_content}")
                except json.JSONDecodeError:
                    logger.error(f"Trader: Failed to parse JSON from evaluation response: {evaluation_response_str}")
                    needs_clarification = False # Cannot determine if clarification is needed
                except Exception as e:
                    logger.error(f"Trader: Error during plan evaluation or clarification request generation: {e}")
                    needs_clarification = False # Default to not needing clarification on error

        if needs_clarification and clarification_request_content:
            logger.info(f"Trader: Requesting clarification from Research Manager (Round {consultation_round + 1}).")
            return {
                "tr_consultation_active": True,
                "tr_clarification_request": clarification_request_content,
                "tr_clarification_response": None, # Clear previous response
                "tr_consultation_round": consultation_round + 1,
                "investment_plan": investment_plan_from_rm, # Keep passing original plan for RM context
                "initial_investment_plan_for_consultation": state['initial_investment_plan_for_consultation'], # Preserve initial plan
                "messages": state.get("messages", []) + [AIMessage(content=f"Clarification request sent: {clarification_request_content}")], # Log action
                "sender": name,
                "trader_investment_plan": "Awaiting clarification from Research Manager..." # Placeholder
            }

        # --- If no clarification needed, or max rounds reached, or clarification received ---
        # Proceed to make final trading decision using Langchain agent structure
        logger.info(f"Trader: Proceeding to make final trading decision for {ticker}.")
        state['tr_consultation_active'] = False # No longer actively seeking clarification in this step

        # Use the latest available plan: either original or the one from clarification_response
        final_plan_to_consider = state.get('tr_clarification_response') or state.get('initial_investment_plan_for_consultation') or investment_plan_from_rm

        # Past memories (unchanged from original trader)
        curr_situation_for_memory = f"{state.get('market_report','')}\n{state.get('sentiment_report','')}\n{state.get('news_report','')}\n{state.get('fundamentals_report','')}"
        past_memories = memory.get_memories(curr_situation_for_memory, n_matches=2)
        past_memory_str = "\n\n".join([rec["recommendation"] for rec in past_memories]) if past_memories else "No past memories found."

        # Prepare Langchain prompt
        # The lc_prompt template expects {investment_plan} and {clarification_response}
        # We need to ensure these are correctly populated for the template.

        # The main_prompt_config's system_message is used in final_lc_system_prompt_template
        # The user_prompt_template from main_prompt_config can be used for the HumanMessage content

        human_message_content = main_prompt_config.get("user_prompt_template", "Please provide your trading decision and rationale.")
        # If user_prompt_template has placeholders, they need to be filled.
        # For now, assume it's a static string or has placeholders that state itself can fill.
        # Or, we can construct it:
        # human_message_content = f"Considering the plan: '{final_plan_to_consider}', and clarification: '{state.get('tr_clarification_response', 'None')}', what is your trade?"

        # The lc_prompt's system message already includes placeholders for investment_plan and clarification_response
        # So, we pass them in the state for the .partial() call.

        partial_lc_prompt = lc_prompt.partial(
            current_date=current_date,
            ticker=ticker,
            investment_plan=final_plan_to_consider, # This will fill {investment_plan}
            clarification_response=state.get('tr_clarification_response', "N/A") # This will fill {clarification_response}
        )
        # We don't use .format() on the system message directly, Langchain does it.
        # The HumanMessage is for the current turn, if the prompt expects it.
        # Often, for agents, the "messages" in state are just the history, and the system prompt drives the next action.

        # Let's assume the main task of the trader (after any clarification) is driven by its system prompt
        # and the current state (which includes the plan).
        # The "messages" in state would be the history.
        # If the trader LC chain is just `prompt | llm`, then the input to invoke should be `state`
        # and `state['messages']` should be the conversation history.
        # The system prompt will use `investment_plan` and `clarification_response` from the `state` passed to `invoke`.

        # Construct the input for the Langchain agent.
        # The state already contains 'investment_plan' (which might be the clarified one if RM updated it)
        # and 'tr_clarification_response'.
        # We need to make sure the `lc_prompt` can access these from the state.
        # The lc_prompt's system message has placeholders {investment_plan} and {clarification_response}
        # These should be picked up from the input dictionary to `invoke`.

        langchain_agent_input_state = state.copy() # Create a copy to avoid modifying original state in unexpected ways
        langchain_agent_input_state["investment_plan"] = final_plan_to_consider # Ensure this is the one to use
        langchain_agent_input_state["clarification_response"] = state.get('tr_clarification_response', "N/A")
        # Add past memories to context if system prompt expects it (e.g. {past_memory_str})
        langchain_agent_input_state["past_memory_str"] = past_memory_str # Assuming system prompt can use this

        # If the LC prompt is just a system message + MessagesPlaceholder, then the input to invoke is state itself
        # as MessagesPlaceholder will take state['messages']

        # The create_trader function receives llm_lc which is the Langchain model
        chain = partial_lc_prompt | llm_lc
        result = chain.invoke(langchain_agent_input_state) # Pass the potentially modified state

        return {
            "messages": state.get("messages", []) + [result], # Append new AIMessage
            "trader_investment_plan": result.content, # The final decision content
            "sender": name,
            # Clear consultation fields as this round is complete
            "tr_consultation_active": False,
            "tr_clarification_request": None,
            # "tr_clarification_response": None, # Keep response for record if needed, or clear
            # tr_consultation_round might be reset by ResearchManager if it sees a new request after this.
        }

    return functools.partial(trader_node, name="Trader")

[end of tradingagents/agents/trader/trader.py]
