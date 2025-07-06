import functools
import time
import json

# Assuming llm_client and prompt_cfg are available in the scope or passed differently.
# The original create_trader took llm and memory.
# The new trader_node signature is (input_data, state, llm_client, prompt_cfg).
# This suggests that create_trader might need to be updated or trader_node is not used via this factory.
# For now, focusing on trader_node logic as per the task.

def trader_node(input_data, state, llm_client, prompt_cfg):
    # input_data here is assumed to be the research report from ResearchManager

    # 1. If chưa đánh giá báo cáo (first round):
    if state.get("consultation_round", 0) == 0:
        # Use the 'evaluate' prompt for the trader
        # The prompt_cfg should contain the actual prompt string for "evaluate" (trader_evaluate_plan)
        # input_data is the report, which the prompt expects.
        # Assuming llm_client.generate takes the prompt string and the report data.
        # The prompt itself is: "Bạn là một nhà giao dịch thông minh. Hãy đọc báo cáo dưới đây và trả lời: ..."
        # So, we need to format the report into this prompt or pass it as context.
        # For simplicity, let's assume the llm_client or the prompt structure handles this.
        # If the prompt is just a template, it might need `report` as a variable.
        # Let's assume `llm_client.generate` can take the prompt name and a dictionary of variables for the prompt.

        # The prompt "trader_evaluate_plan" is a template that implies the report is part of the input.
        # It's not explicitly shown how `input_data` (the report) is combined with the prompt text.
        # Let's assume `llm_client.generate` handles this: it takes the prompt identifier
        # (e.g., "evaluate" which maps to "trader_evaluate_plan") and the data to fill into the prompt.
        # The prompt "trader_evaluate_plan" does not have a placeholder for the report, it says "Hãy đọc báo cáo dưới đây".
        # This implies the report should be appended to the prompt string.

        evaluate_prompt_template = prompt_cfg["evaluate"] # This is "trader_evaluate_plan" content
        full_evaluate_prompt = f"{evaluate_prompt_template}\n\nBáo cáo:\n{input_data}"

        resp = llm_client.generate(full_evaluate_prompt) # Pass the full prompt including the report

        try:
            parsed = json.loads(resp)
        except json.JSONDecodeError:
            # Handle error: LLM response was not valid JSON
            # This could involve logging, retrying, or returning an error state
            # For now, let's assume it might return a string that's not JSON, try to adapt.
            # Or, more robustly, this path should lead to an error or a request to re-format.
            # Given the spec, we expect JSON. If not, it's an issue.
            # For this exercise, we'll proceed assuming valid JSON or an error will be caught by the caller.
            raise ValueError(f"Failed to parse LLM response as JSON: {resp}")


        if parsed.get("clarification_needed"):
            state["clarification_question"] = parsed["clarification_question"]
            state["consultation_round"] = 1
            state["clarification_requested"] = True
            # The return value "clarification_needed" is a string that the graph will use for routing.
            return {"status": "clarification_needed", "state": state} # Return a dictionary to update state and provide status for routing
        else:
            # If no clarification is needed, proceed to final decision directly in the same step.
            # The prompt for final decision is "trader_final_decision".
            # It expects "thông tin làm rõ (nếu có) và báo cáo ban đầu".
            # In this branch, there is no clarification.

            final_decision_prompt_template = prompt_cfg["clarify"] # This is "trader_final_decision"
            # The prompt needs the original report.
            # Format: "Với thông tin làm rõ (nếu có) và báo cáo ban đầu, hãy ra quyết định cuối cùng."
            # Report is in input_data. Clarification is absent.

            final_prompt_input = f"{final_decision_prompt_template}\n\nBáo cáo ban đầu:\n{input_data}\n\nThông tin làm rõ: Không có."
            final_resp = llm_client.generate(final_prompt_input)
            # The final_resp is expected to be a JSON string: {"action": "...", "reason": "..."}
            # This is the final output of the trader node in this case.
            try:
                final_decision_output = json.loads(final_resp)
            except json.JSONDecodeError:
                raise ValueError(f"Failed to parse final decision LLM response as JSON: {final_resp}")

            state["consultation_round"] = 1 # Mark that one round (evaluation) happened
            state["clarification_requested"] = False
            # Store the final decision details in state if needed, or return directly.
            # The task implies `trader_node` returns the final decision directly if ready.
            return {"status": "decision_made", "decision": final_decision_output, "state": state}

    # 2. Nếu đã có trả lời làm rõ (subsequent round after clarification):
    elif state.get("clarification_response"): # Check if clarification_response exists in state
        clarification_text = state["clarification_response"]

        # The prompt for final decision is "trader_final_decision" (mapped from prompt_cfg["clarify"])
        # It needs the original report (input_data) and the clarification_response.
        final_decision_prompt_template = prompt_cfg["clarify"] # This is "trader_final_decision"

        # Format the input for the final decision prompt
        final_prompt_input = (
            f"{final_decision_prompt_template}\n\n"
            f"Báo cáo ban đầu:\n{input_data}\n\n"
            f"Thông tin làm rõ:\n{clarification_text}"
        )

        final_resp = llm_client.generate(final_prompt_input)
        # final_resp is the JSON string: {"action": "...", "reason": "..."}
        try:
            final_decision_output = json.loads(final_resp)
        except json.JSONDecodeError:
            raise ValueError(f"Failed to parse final decision LLM response (with clarification) as JSON: {final_resp}")

        state["consultation_round"] = state.get("consultation_round", 0) + 1
        state["clarification_requested"] = False # Reset flag
        # The task implies returning the final response directly.
        return {"status": "decision_made", "decision": final_decision_output, "state": state}

    else:
        # This case should ideally not be reached if the graph logic is correct
        # (e.g., if consultation_round > 0 but no clarification_response means something went wrong
        # or it's a state not covered by the two main branches).
        # For robustness, handle this, perhaps by logging an error or re-evaluating.
        # Or, it could be that the graph directs here if no clarification was needed initially
        # and the first block didn't make a final decision (which it now does).
        # Based on the new logic in the first block, this 'else' might be less likely to be hit
        # unless state is manipulated unexpectedly.
        # If first block always either requests clarification or makes a decision, this is an anomaly.
        # Let's assume for now this indicates an issue or an unexpected state.
        # For safety, could re-trigger evaluation if state is unclear.
        # However, the prompt implies a clear two-step (evaluate -> maybe clarify -> final decision).
        # This path suggests the state is not in one of those clear phases.
        # For now, let's return an error or a status indicating an issue.
        return {"status": "error", "message": "Trader node reached an unexpected state.", "state": state}

# The create_trader function might need to be adjusted or may no longer be the way trader_node is instantiated
# if llm_client and prompt_cfg are to be passed directly or via a different mechanism.
# If create_trader is still used, it needs to be adapted to provide these to trader_node.

# Example of how create_trader might be (if it's still the entry point for creating this node):
# def create_trader(llm_client, prompt_cfg, memory): # Added llm_client, prompt_cfg
    # trader_node_configured = functools.partial(trader_node, llm_client=llm_client, prompt_cfg=prompt_cfg)
    # The 'input_data' and 'state' for trader_node would be passed by the graph execution logic.
    # The 'name' parameter from the original trader_node is not in the new signature.
    # If 'name' is still needed (e.g., for sender field), state update within trader_node should handle it.
    # return trader_node_configured

# For now, the task is to refactor trader_node itself. The integration with create_trader
# or the graph setup will determine how llm_client and prompt_cfg are provided.
# The original `create_trader` is left below for reference, but it's not compatible with the new `trader_node` directly.

def create_trader(llm, memory):
    # This function is from the original file.
    # It's not directly compatible with the new trader_node signature without modification.
    # The new trader_node expects (input_data, state, llm_client, prompt_cfg).
    # The original trader_node (within this factory) had (state, name) and used 'llm' and 'memory' from the closure.
    # This part would need to be reconciled with how the graph instantiates and calls nodes.
    # For now, I'm keeping the refactored trader_node separate as requested.
    # If this create_trader is still the intended way to get the trader_node,
    # it would need to be like:
    # def create_trader(llm_as_llm_client, prompt_config_for_trader, memory_if_needed_by_new_node):
    #     # Note: new trader_node doesn't use memory directly in its signature, but state might carry it
    #     configured_node = functools.partial(trader_node, llm_client=llm_as_llm_client, prompt_cfg=prompt_config_for_trader)
    #     return configured_node
    # However, the prompt shows trader_node being called with (input_data, state, llm_client, prompt_cfg)
    # which implies it might be used directly rather than via this factory, or the factory changes.

    # Keeping original content below for reference, but it's superseded by the trader_node above.
    def original_trader_logic_node(state, name): # Renamed to avoid clash
        company_name = state["company_of_interest"]
        # ... (rest of the original logic)
        investment_plan = state["investment_plan"]
        market_research_report = state["market_report"]
        sentiment_report = state["sentiment_report"]
        news_report = state["news_report"]
        fundamentals_report = state["fundamentals_report"]

        curr_situation = f"{market_research_report}\n\n{sentiment_report}\n\n{news_report}\n\n{fundamentals_report}"
        past_memories = memory.get_memories(curr_situation, n_matches=2)

        past_memory_str = ""
        if past_memories:
            for i, rec in enumerate(past_memories, 1):
                past_memory_str += rec["recommendation"] + "\n\n"
        else:
            past_memory_str = "No past memories found."

        context = {
            "role": "user",
            "content": f"Based on a comprehensive analysis by a team of analysts, here is an investment plan tailored for {company_name}. This plan incorporates insights from current technical market trends, macroeconomic indicators, and social media sentiment. Use this plan as a foundation for evaluating your next trading decision.\n\nProposed Investment Plan: {investment_plan}\n\nLeverage these insights to make an informed and strategic decision.",
        }

        messages = [
            {
                "role": "system",
                "content": f"""You are a trading agent analyzing market data to make investment decisions. Based on your analysis, provide a specific recommendation to buy, sell, or hold. End with a firm decision and always conclude your response with 'FINAL TRANSACTION PROPOSAL: **BUY/HOLD/SELL**' to confirm your recommendation. Do not forget to utilize lessons from past decisions to learn from your mistakes. Here is some reflections from similar situatiosn you traded in and the lessons learned: {past_memory_str}""",
            },
            context,
        ]

        result = llm.invoke(messages)

        return {
            "messages": [result],
            "trader_investment_plan": result.content,
            "sender": name, # Original node set a sender
        }

    # return functools.partial(original_trader_logic_node, name="Trader")
    # For the purpose of this refactoring, we assume trader_node is defined as a standalone function
    # and the create_trader factory will be adapted or bypassed later.
    # So, this function `create_trader` as a whole is now problematic if trader_node is global.
    # Let's comment out the return of the old logic.
    # The file will now contain the new trader_node and the old create_trader structure.
    # This needs clarification on how trader_node is actually integrated.
    # For now, the new trader_node is defined, and the old factory is left, but commented out its return.
    pass # Placeholder for the rest of create_trader if it's to be kept and modified.
