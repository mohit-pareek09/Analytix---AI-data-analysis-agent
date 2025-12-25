import sys
import os
import json
import re
import pandas as pd
import matplotlib.pyplot as plt
from pydantic import BaseModel, Field
from typing import Optional

# Try to import ChatOllama from common packages
try:
    from langchain_ollama import ChatOllama
except Exception:
    try:
        from langchain_community.chat_models import ChatOllama
    except Exception:
        ChatOllama = None

# ---------------- Tools ----------------
def clean_feature(df: pd.DataFrame, feature: str, persist: bool = False, csv_path: Optional[str] = None) -> str:
    # Try to match column name case-insensitively
    matching_cols = [col for col in df.columns if col.lower() == feature.lower()]
    if matching_cols:
        feature = matching_cols[0]
    elif feature not in df.columns:
        return f"ERROR: Column '{feature}' not found. Available columns: {list(df.columns)}"
    
    col = df[feature]
    if pd.api.types.is_numeric_dtype(col):
        median = col.median()
        df[feature] = col.fillna(median)
        if persist and csv_path:
            df.to_csv(csv_path, index=False)
        return f"Filled missing numeric values in '{feature}' with median = {median:.4f}."
    else:
        mode = col.mode().iloc[0] if not col.mode().empty else ""
        df[feature] = col.fillna(mode)
        if persist and csv_path:
            df.to_csv(csv_path, index=False)
        return f"Filled missing categorical values in '{feature}' with mode = '{mode}'."

def plot_histogram(df: pd.DataFrame, feature: str, bins: int = 10) -> str:
    # Try to match column name case-insensitively
    matching_cols = [col for col in df.columns if col.lower() == feature.lower()]
    if matching_cols:
        feature = matching_cols[0]
    elif feature not in df.columns:
        return f"ERROR: Column '{feature}' not found. Available columns: {list(df.columns)}"
    
    if not pd.api.types.is_numeric_dtype(df[feature]):
        return f"ERROR: Column '{feature}' is not numeric; histogram not available."
    plt.close('all')
    fig, ax = plt.subplots()
    df[feature].dropna().hist(bins=bins, ax=ax)
    ax.set_title(f"Histogram of {feature}")
    ax.set_xlabel(feature)
    ax.set_ylabel("Count")
    out_path = f"hist_{feature}.png"
    fig.savefig(out_path, bbox_inches='tight')
    plt.close(fig)
    return f"Saved histogram to {out_path}"

def calculate_correlation(df: pd.DataFrame, f1: str, f2: str) -> str:
    if f1 not in df.columns or f2 not in df.columns:
        return f"ERROR: One or both columns not found: {f1}, {f2}."
    if not (pd.api.types.is_numeric_dtype(df[f1]) and pd.api.types.is_numeric_dtype(df[f2])):
        return f"ERROR: Both columns must be numeric to compute correlation."
    corr = df[[f1, f2]].dropna().corr().iloc[0, 1]
    return f"Pearson correlation between '{f1}' and '{f2}': {corr:.4f}"

def find_best_correlation(df: pd.DataFrame, feature: str) -> str:
    """Find which feature has the strongest correlation with the given feature."""
    if feature not in df.columns:
        return f"ERROR: Column '{feature}' not found."
    if not pd.api.types.is_numeric_dtype(df[feature]):
        return f"ERROR: Column '{feature}' is not numeric."
    
    numeric_cols = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col]) and col != feature]
    if not numeric_cols:
        return f"ERROR: No other numeric columns to correlate with '{feature}'."
    
    correlations = {}
    for col in numeric_cols:
        try:
            corr = df[[feature, col]].dropna().corr().iloc[0, 1]
            correlations[col] = abs(corr)  # Use absolute value to find strongest relationship
        except:
            continue
    
    if not correlations:
        return f"ERROR: Could not calculate correlations for '{feature}'."
    
    best_feature = max(correlations, key=correlations.get)
    best_corr = df[[feature, best_feature]].dropna().corr().iloc[0, 1]
    
    result = f"Feature most closely related to '{feature}': '{best_feature}' (correlation: {best_corr:.4f})\n"
    result += "\nAll correlations with '" + feature + "':\n"
    sorted_corrs = sorted(correlations.items(), key=lambda x: x[1], reverse=True)
    for col, corr_val in sorted_corrs:
        actual_corr = df[[feature, col]].dropna().corr().iloc[0, 1]
        result += f"  - {col}: {actual_corr:.4f}\n"
    
    return result.strip()

# -------------- Expected JSON schema --------------
class ToolCall(BaseModel):
    tool: str = Field(..., description="Tool to run: 'clean_feature', 'plot_histogram', 'calculate_correlation', 'find_best_correlation', or 'help'")
    feature: Optional[str] = None
    feature2: Optional[str] = None
    bins: Optional[int] = 10

def dispatch_tool(df: pd.DataFrame, call: ToolCall, csv_path: Optional[str] = None) -> str:
    t = (call.tool or "").lower()
    if t == "clean_feature":
        if not call.feature:
            return "ERROR: 'feature' is required for clean_feature."
        return clean_feature(df, call.feature, persist=False, csv_path=csv_path)
    elif t == "plot_histogram":
        if not call.feature:
            return "ERROR: 'feature' is required for plot_histogram."
        return plot_histogram(df, call.feature, bins=call.bins or 10)
    elif t == "calculate_correlation":
        if not (call.feature and call.feature2):
            return "ERROR: 'feature' and 'feature2' are required for calculate_correlation."
        return calculate_correlation(df, call.feature, call.feature2)
    elif t == "find_best_correlation":
        if not call.feature:
            return "ERROR: 'feature' is required for find_best_correlation."
        return find_best_correlation(df, call.feature)
    elif t in ("help", "what_can_you_do"):
        cols_str = ", ".join(df.columns)
        return (f"Available columns: {cols_str}\n\n"
                "I can run these tools:\n"
                "- clean_feature(feature): Fill missing values\n"
                "- plot_histogram(feature, bins): Create histogram\n"
                "- calculate_correlation(feature, feature2): Calculate correlation between two features\n"
                "- find_best_correlation(feature): Find which feature correlates most strongly\n\n"
                "Example prompts:\n"
                "- 'Plot a histogram of age'\n"
                "- 'What's the correlation between weight and score?'\n"
                "- 'Clean the column weight'\n"
                "- 'Which feature is most closely related to age?'")
    else:
        return f"ERROR: Unknown tool '{call.tool}'."

# ---------------- Helper: robust response -> text ----------------
def extract_text_from_response(resp) -> str:
    """
    Robustly convert various model response types to a simple string.
    Handles: plain str, LangChain-like objects with .generations, dicts, or objects with .text/.content.
    """
    # Plain string
    if isinstance(resp, str):
        return resp

    # LangChain-like: resp.generations -> list of lists / list
    if hasattr(resp, "generations"):
        try:
            gens = resp.generations
            first = gens[0]
            if isinstance(first, (list, tuple)):
                choice = first[0]
            else:
                choice = first
            if hasattr(choice, "text"):
                return choice.text
            if hasattr(choice, "content"):
                return choice.content
            return str(choice)
        except Exception:
            return str(resp)

    # dict-like
    if isinstance(resp, dict):
        # common keys
        for key in ("text", "content", "response", "answer"):
            if key in resp:
                return resp[key]
        # fallback to json-dump
        try:
            return json.dumps(resp)
        except Exception:
            return str(resp)

    # generic object: try common attributes
    if hasattr(resp, "text"):
        try:
            return getattr(resp, "text")
        except Exception:
            pass
    if hasattr(resp, "content"):
        try:
            return getattr(resp, "content")
        except Exception:
            pass

    # fallback to string conversion
    return str(resp)

# ---------------- Main ----------------
def main(csv_path: str):
    if ChatOllama is None:
        print("ERROR: ChatOllama import failed. Install 'langchain-ollama' or the appropriate package.")
        return

    if not os.path.exists(csv_path):
        print(f"ERROR: CSV file not found: {csv_path}")
        return

    print("Loading data...")
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} rows and {len(df.columns)} columns: {list(df.columns)}")

    # Try a list of candidate model names (adjust if you have different tags)
    model = None
    last_err = None
    candidates = ("gemma3:latest", "gemma3:1b")
    
    print("Attempting to connect to Ollama models...")
    for cand in candidates:
        try:
            print(f"  Trying {cand}...", end=" ")
            test_model = ChatOllama(model=cand, temperature=0.0)
            # Test with a simple invocation
            test_model.invoke("Hi")
            model = test_model
            print(f"✓ Success!")
            print(f"Using Ollama model: {cand}")
            break
        except Exception as e:
            print(f"✗ Failed: {str(e)}")
            last_err = e
    
    if model is None:
        print("\n❌ Failed to initialize any Ollama model.")
        print("Available models detected: gemma3:1b, gemma3:latest")
        print("Please ensure Ollama is running: ollama serve")
        print("\nThen run `ollama list` to verify models are available.")
        if last_err:
            print(f"\nLast error: {last_err}")
        return

    # System instructions to force JSON-only response
    system_instructions = (
        "You are a concise data assistant. The dataset has these columns: " + str(list(df.columns)) + ".\n"
        "When the user asks for an action, OUTPUT EXACTLY one JSON OBJECT and NOTHING ELSE. "
        "The object must have these fields: "
        '"tool" (one of "clean_feature","plot_histogram","calculate_correlation","find_best_correlation","help"), '
        '"feature" (exact column name from dataset, case-sensitive, or null), '
        '"feature2" (exact column name from dataset, case-sensitive, or null), '
        '"bins" (integer, default 10). '
        "IMPORTANT: Use exact column names as they appear in the dataset. "
        "Examples:\n"
        "- 'Which feature is most related to age?' -> {\"tool\":\"find_best_correlation\",\"feature\":\"age\"}\n"
        "- 'Clean the weight column' -> {\"tool\":\"clean_feature\",\"feature\":\"weight\"}\n"
        "- 'Plot histogram of score' -> {\"tool\":\"plot_histogram\",\"feature\":\"score\",\"bins\":10}\n"
        "If unsure which tool to use, set tool='help'. "
        "Do not add commentary, markdown, or surrounding text. Output only the JSON object."
    )

    print("Ready. Ask something about the dataset (type 'exit' to quit).")

    while True:
        try:
            user = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            break
            
        if not user:
            print("Please enter a query or type 'exit' to quit.")
            continue
            
        if user.lower() in ("exit", "quit", "bye"):
            print("Goodbye.")
            break

        low = user.lower()
        # Local quick responses to avoid calling model for trivial queries
        
        # Check if asking for NUMBER/COUNT of columns (not the column names)
        # BUT exclude if they're asking to clean/plot/analyze a column
        is_analysis_query = any(word in low for word in ["clean", "plot", "histogram", "correlation", "visualize", "fix", "remove", "fill"])
        
        if not is_analysis_query:
            if ("how many column" in low or "no of column" in low or "number of column" in low or 
                "total column" in low or "count of column" in low or "exact number of column" in low):
                print(f"Result: {len(df.columns)} columns")
                continue
            
            # Check if asking for column NAMES or WHAT columns
            if "column" in low and any(word in low for word in ["what", "which", "list", "show", "names", "present"]):
                print("Result: Columns =", list(df.columns))
                continue
                
            # Check for row count queries
            if ("row" in low or "rows" in low) and ("how many" in low or "number of" in low or "no of" in low or "count" in low):
                print(f"Result: {len(df)} rows")
                continue

        prompt = system_instructions + "\n\nUser: " + user + "\n\nRespond with JSON now."

        # --- CALL MODEL (very robust) ---
        try:
            # Use invoke method which is the standard LangChain interface
            resp = model.invoke(prompt)
        except Exception as e:
            print("Model error (while calling):", e)
            continue

        # Normalize response to text
        try:
            llm_text = extract_text_from_response(resp)
        except Exception as e:
            print("Error normalizing model response:", e)
            print("Raw response repr:", repr(resp))
            continue

        # Ensure string
        if not isinstance(llm_text, str):
            try:
                llm_text = str(llm_text)
            except Exception:
                print("Unable to convert model response to text. Raw repr:", repr(resp))
                continue

        # Extract JSON object from response (best effort)
        m = re.search(r'(\{.*\})', llm_text, flags=re.DOTALL)
        if m:
            json_text = m.group(1)
        else:
            json_text = llm_text.strip()

        # Parse JSON
        try:
            parsed = json.loads(json_text)
        except Exception as e:
            print("JSON parsing failed. Raw model output:\n", llm_text)
            print("Error:", e)
            continue

        # Validate with pydantic
        try:
            call = ToolCall.model_validate(parsed)
        except Exception as e:
            print("ToolCall validation error:", e)
            print("Parsed JSON:", parsed)
            continue

        print(f"[Engine decided to run tool: {call.tool}]")
        out = dispatch_tool(df, call, csv_path=csv_path)
        print("Result:", out)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python app.py path/to/data.csv")
    else:
        main(sys.argv[1])