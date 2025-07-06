import pandas as pd
import os
import logging
from typing import List, Dict, Optional, Union

logger = logging.getLogger(__name__)

DEFAULT_LOG_DIR = "logs"
DEFAULT_USAGE_CSV = "llm_usage.csv"

def get_usage_summary(
    csv_filepath: Optional[str] = None,
    group_by: Optional[List[str]] = None,
    date_range: Optional[tuple[Optional[str], Optional[str]]] = None, # (start_date_str, end_date_str)
    provider_filter: Optional[str] = None,
    model_filter: Optional[str] = None,
    agent_filter: Optional[str] = None,
    key_filter: Optional[str] = None # Filter by api_key_identifier
) -> Optional[pd.DataFrame]:
    """
    Reads the LLM usage CSV log and returns a summarized DataFrame.

    Args:
        csv_filepath (str, optional): Path to the llm_usage.csv file.
                                      Defaults to 'logs/llm_usage.csv'.
        group_by (List[str], optional): List of columns to group by.
                                         Example: ['date', 'provider_name', 'model_name', 'agent_name']
                                         Defaults to ['date', 'provider_name', 'model_name', 'agent_name', 'api_key_identifier'].
        date_range (tuple[str, str], optional): A tuple (start_date, end_date) for filtering by date.
                                                Dates should be in 'YYYY-MM-DD' format.
                                                None for start or end means no lower/upper bound.
        provider_filter (str, optional): Filter by a specific provider name.
        model_filter (str, optional): Filter by a specific model name.
        agent_filter (str, optional): Filter by a specific agent name.
        key_filter (str, optional): Filter by a specific API key identifier (e.g., last 4 chars).


    Returns:
        Optional[pd.DataFrame]: A DataFrame with summarized usage data (tokens, cost, duration, count)
                                or None if the file is not found or an error occurs.
    """
    if csv_filepath is None:
        # Default path relative to project root (assuming utils is one level down from root)
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        # Fallback to current working directory if that fails for some reason
        if not os.path.exists(os.path.join(project_root, DEFAULT_LOG_DIR)):
             project_root = os.getcwd()
        csv_filepath = os.path.join(project_root, DEFAULT_LOG_DIR, DEFAULT_USAGE_CSV)


    if not os.path.exists(csv_filepath):
        logger.warning(f"LLM usage log file not found at: {csv_filepath}")
        return None

    try:
        df = pd.read_csv(csv_filepath)
        if df.empty:
            logger.info(f"LLM usage log file is empty: {csv_filepath}")
            return pd.DataFrame() # Return empty DataFrame

        # Data type conversions and new columns
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['date'] = df['timestamp'].dt.date # For grouping by date

        # Ensure numeric columns are numeric, fill NaNs with 0 for aggregation
        numeric_cols = ['prompt_tokens', 'completion_tokens', 'total_tokens',
                        'duration_seconds', 'estimated_cost_usd']
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

        df['success'] = df['success'].astype(bool)

        # Apply filters
        if date_range:
            start_date, end_date = date_range
            if start_date:
                df = df[df['timestamp'] >= pd.to_datetime(start_date)]
            if end_date:
                # Add 1 day to end_date to make it inclusive for timestamp
                df = df[df['timestamp'] < pd.to_datetime(end_date) + pd.Timedelta(days=1)]

        if provider_filter:
            df = df[df['provider_name'].str.lower() == provider_filter.lower()]
        if model_filter:
            df = df[df['model_name'].str.contains(model_filter, case=False, na=False)] # Partial match
        if agent_filter:
            df = df[df['agent_name'].str.lower() == agent_filter.lower()]
        if key_filter:
            df = df[df['api_key_identifier'].str.contains(key_filter, case=False, na=False)]


        if df.empty:
            logger.info(f"No data matches the applied filters in: {csv_filepath}")
            return pd.DataFrame()

        # Default grouping if not specified
        if group_by is None:
            group_by = ['date', 'provider_name', 'model_name', 'agent_name', 'api_key_identifier']

        # Ensure all group_by columns exist
        valid_group_by = [col for col in group_by if col in df.columns]
        if not valid_group_by:
            logger.warning(f"None of the group_by columns {group_by} exist in the DataFrame. Returning raw filtered data.")
            return df

        summary_df = df.groupby(valid_group_by).agg(
            total_prompt_tokens=('prompt_tokens', 'sum'),
            total_completion_tokens=('completion_tokens', 'sum'),
            total_tokens_sum=('total_tokens', 'sum'), # Sum of 'total_tokens' column
            total_estimated_cost_usd=('estimated_cost_usd', 'sum'),
            total_duration_seconds=('duration_seconds', 'sum'),
            average_duration_seconds=('duration_seconds', 'mean'),
            request_count=('timestamp', 'count'), # Count of requests
            successful_requests=('success', lambda x: x.sum()), # Count of True values
        ).reset_index()

        summary_df['failure_requests'] = summary_df['request_count'] - summary_df['successful_requests']
        summary_df['success_rate'] = (summary_df['successful_requests'] / summary_df['request_count']).fillna(0) * 100

        # Round numerics for display
        for col in ['total_estimated_cost_usd']:
            summary_df[col] = summary_df[col].round(8)
        for col in ['total_duration_seconds', 'average_duration_seconds', 'success_rate']:
             summary_df[col] = summary_df[col].round(3)

        return summary_df

    except FileNotFoundError:
        logger.error(f"LLM usage log file not found at path: {csv_filepath}")
        return None
    except pd.errors.EmptyDataError:
        logger.warning(f"LLM usage log file is empty or malformed: {csv_filepath}")
        return pd.DataFrame()
    except Exception as e:
        logger.error(f"Error processing LLM usage log '{csv_filepath}': {e}", exc_info=True)
        return None


def check_quota_warnings(
    csv_filepath: Optional[str] = None,
    quota_config: Optional[Dict] = None,
    alert_threshold_pct: float = 0.90
) -> List[str]:
    """
    Checks current usage against defined quotas and returns warning messages.
    (Basic implementation, can be expanded)

    Args:
        csv_filepath (str, optional): Path to the llm_usage.csv file.
        quota_config (Dict, optional): Dict defining quotas, e.g.,
            {
                "api_key_identifiers": { # Key is the api_key_identifier (e.g., ...key_suffix)
                    "...key1": {"max_cost_monthly": 50.0, "max_tokens_daily": 1000000},
                    "...key2": {"max_cost_monthly": 20.0}
                },
                "models": { # Key is provider/model_name
                    "google/gemini-1.5-flash": {"max_requests_per_minute": 60} // Harder to track without precise timing
                }
            }
        alert_threshold_pct (float): Percentage of quota usage to trigger a warning (e.g., 0.9 for 90%).

    Returns:
        List[str]: A list of warning messages.
    """
    warnings = []
    if quota_config is None:
        logger.info("No quota configuration provided for checking warnings.")
        return warnings

    summary_daily = get_usage_summary(csv_filepath, group_by=['date', 'api_key_identifier'])
    summary_monthly_cost = get_usage_summary(csv_filepath, group_by=['api_key_identifier']) # Simplified for monthly cost

    if summary_daily is None or summary_monthly_cost is None:
        logger.warning("Could not generate usage summaries for quota checking.")
        return ["Could not generate usage summaries for quota checking."]

    today_date = pd.to_datetime('today').date()

    # Check daily token limits per key
    key_quotas = quota_config.get("api_key_identifiers", {})
    for key_id, quotas in key_quotas.items():
        if "max_tokens_daily" in quotas:
            key_daily_usage = summary_daily[
                (summary_daily['api_key_identifier'] == key_id) &
                (summary_daily['date'] == today_date)
            ]
            if not key_daily_usage.empty:
                current_tokens = key_daily_usage['total_tokens_sum'].sum()
                if current_tokens >= quotas["max_tokens_daily"] * alert_threshold_pct:
                    warnings.append(
                        f"Warning: API Key '{key_id}' has used {current_tokens} tokens today, "
                        f"exceeding {alert_threshold_pct*100:.0f}% of daily quota ({quotas['max_tokens_daily']})."
                    )

        if "max_cost_monthly" in quotas:
            # This is a simplified monthly check (total cost ever for the key).
            # A more accurate check would sum costs for the current calendar month.
            key_total_cost_usage = summary_monthly_cost[summary_monthly_cost['api_key_identifier'] == key_id]
            if not key_total_cost_usage.empty:
                current_total_cost = key_total_cost_usage['total_estimated_cost_usd'].sum()
                if current_total_cost >= quotas["max_cost_monthly"] * alert_threshold_pct:
                     warnings.append(
                        f"Warning: API Key '{key_id}' has accumulated ${current_total_cost:.2f} total cost, "
                        f"exceeding {alert_threshold_pct*100:.0f}% of illustrative monthly quota (${quotas['max_cost_monthly']:.2f})."
                    )

    # TODO: Add checks for model-specific quotas (e.g., requests per minute) - more complex.

    for warning in warnings:
        logger.warning(warning)
    return warnings


if __name__ == '__main__':
    print("--- Testing LLM Usage Summarizer ---")
    # Create a dummy llm_usage.csv if it doesn't exist for testing
    dummy_log_dir = os.path.join(os.getcwd(), DEFAULT_LOG_DIR)
    dummy_csv_path = os.path.join(dummy_log_dir, DEFAULT_USAGE_CSV)

    if not os.path.exists(dummy_csv_path):
        print(f"Creating dummy '{dummy_csv_path}' for testing...")
        os.makedirs(dummy_log_dir, exist_ok=True)
        header = 'timestamp,provider_name,api_key_identifier,agent_name,model_name,prompt_tokens,completion_tokens,total_tokens,duration_seconds,estimated_cost_usd,success\n'
        # Yesterday's date
        yesterday = (datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(days=1)).isoformat()
        today = datetime.datetime.now(datetime.timezone.utc).isoformat()

        dummy_data = [
            f"{yesterday},google,...key1,MarketAnalyst,gemini-1.5-flash,100,200,300,1.5,0.0005,True\n",
            f"{today},google,...key1,NewsAnalyst,gemini-1.5-flash,150,300,450,2.1,0.0007,True\n",
            f"{today},google,...key1,MarketAnalyst,gemini-1.0-pro,200,500,700,3.0,0.0015,True\n",
            f"{today},openai,...keyX,Trader,gpt-3.5-turbo,50,100,150,0.8,0.0001,True\n",
            f"{today},google,...key2,Reflector,gemini-1.0-pro,300,600,900,4.5,0.0025,False\n", # Failed request
        ]
        with open(dummy_csv_path, 'w', encoding='utf-8') as f:
            f.write(header)
            for line in dummy_data:
                f.write(line)
    else:
        print(f"Using existing '{dummy_csv_path}' for testing.")

    summary = get_usage_summary(csv_filepath=dummy_csv_path)
    if summary is not None:
        print("\nDefault Summary (grouped by date, provider, model, agent, key):")
        print(summary.to_string())

    summary_by_agent = get_usage_summary(csv_filepath=dummy_csv_path, group_by=['date', 'agent_name'])
    if summary_by_agent is not None:
        print("\nSummary by Date and Agent:")
        print(summary_by_agent.to_string())

    summary_today_google = get_usage_summary(
        csv_filepath=dummy_csv_path,
        date_range=(datetime.date.today().strftime('%Y-%m-%d'), datetime.date.today().strftime('%Y-%m-%d')),
        provider_filter="google"
    )
    if summary_today_google is not None:
        print("\nSummary for Today and Google provider:")
        print(summary_today_google.to_string())

    print("\n--- Testing Quota Warnings (Illustrative) ---")
    dummy_quota_config = {
        "api_key_identifiers": {
            "...key1": {"max_tokens_daily": 1000, "max_cost_monthly": 1.0}, # key1 today has 300+450+700 = 1450 tokens
            "...key2": {"max_tokens_daily": 500, "max_cost_monthly": 0.5}  # key2 today has 900 tokens
        }
    }
    warnings = check_quota_warnings(csv_filepath=dummy_csv_path, quota_config=dummy_quota_config, alert_threshold_pct=0.8)
    if warnings:
        print("Quota Warnings Found:")
        for w in warnings:
            print(f"- {w}")
    else:
        print("No quota warnings based on dummy data and config.")

    # Clean up dummy file if you want
    # if os.path.exists(dummy_csv_path) and "dummy_data" in open(dummy_csv_path).read():
    #     os.remove(dummy_csv_path)
    #     print(f"\nRemoved dummy '{dummy_csv_path}'.")
