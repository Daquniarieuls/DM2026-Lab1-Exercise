import pandas as pd
import nltk
from typing import List, Dict, Any, Optional
from langchain_core.tools import tool

# Ensure punkt data is available for nltk tokenizer if needed
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt", quiet=True)

try:
    nltk.data.find("tokenizers/punkt_tab")
except LookupError:
    nltk.download("punkt_tab", quiet=True)


def make_tools(session):
    @tool
    def check_missing_tool(text_column: str = "text") -> Dict[str, Any]:
        """Checks for missing or empty text values in the working DataFrame.

        Args:
            text_column: The name of the text column to inspect for missing values.

        Returns:
            A dictionary containing the total rows, number of missing entries, missing percentage, and result_id.
        """
        if session.dataframe is None:
            return {"error": "No dataset loaded in session. Please load a dataset first."}

        df = session.dataframe
        if text_column not in df.columns:
            return {"error": f"Column '{text_column}' not found in dataset."}

        total_rows = len(df)
        
        # Check for NaN / null values or empty / whitespace-only strings
        series = df[text_column]
        is_null = series.isna()
        is_empty = series.astype(str).str.strip() == ""
        missing_mask = is_null | is_empty

        missing_count = int(missing_mask.sum())
        missing_pct = float(missing_count / total_rows * 100) if total_rows > 0 else 0.0

        result_id = session.next_result_id("check_missing")

        summary = {
            "result_id": result_id,
            "text_column": text_column,
            "total_rows": total_rows,
            "missing_count": missing_count,
            "missing_pct": missing_pct,
        }

        session.store_result("check_missing_tool", {"text_column": text_column}, summary)

        return summary

    @tool
    def check_duplicates_tool(subset: Optional[List[str]] = None, drop: bool = False) -> Dict[str, Any]:
        """Checks for duplicate documents across columns and optionally drops them.

        When dropping duplicates, keep=False is used to drop every copy of a duplicated row,
        matching Master's drop_duplicates(keep=False) behavior.

        Args:
            subset: Optional list of column names to consider for identifying duplicates.
                    If None, all columns are checked.
            drop: Whether to drop duplicate rows from session.dataframe in place.

        Returns:
            A dictionary containing total_rows_before, duplicate_count, duplicate_indices,
            drop_requested, dropped status, removed_indices, total_rows_after, and result_id.
        """
        if session.dataframe is None:
            return {"error": "No dataset loaded in session. Please load a dataset first."}

        df = session.dataframe
        total_rows_before = len(df)

        # Count extra duplicate copies for reporting
        duplicate_count = int(df.duplicated(subset=subset, keep="first").sum())

        # Mask identifying all copies of duplicated rows
        all_duplicates_mask = df.duplicated(subset=subset, keep=False)
        duplicate_indices = df.index[all_duplicates_mask].tolist()

        actually_dropped = False
        removed_indices: List[Any] = []

        if drop and len(duplicate_indices) > 0:
            removed_indices = duplicate_indices.copy()
            df.drop_duplicates(subset=subset, keep=False, inplace=True)
            actually_dropped = True

        total_rows_after = len(session.dataframe)
        result_id = session.next_result_id("check_duplicates")

        summary = {
            "result_id": result_id,
            "total_rows_before": total_rows_before,
            "duplicate_count": duplicate_count,
            "duplicate_indices": duplicate_indices,
            "drop_requested": drop,
            "dropped": actually_dropped,
            "removed_indices": removed_indices,
            "total_rows_after": total_rows_after,
        }

        session.store_result("check_duplicates_tool", {"subset": subset, "drop": drop}, summary)

        return summary

    @tool
    def tokenize_tool(text_column: str = "text") -> Dict[str, Any]:
        """Tokenizes each document's text into unigrams using NLTK word tokenizer.

        Args:
            text_column: Name of the column containing document text to tokenize.

        Returns:
            A summary dictionary containing result_id, total_documents, total_tokens, and sample_tokens.
        """
        if session.dataframe is None:
            return {"error": "No dataset loaded in session. Please load a dataset first."}

        df = session.dataframe
        if text_column not in df.columns:
            return {"error": f"Column '{text_column}' not found in dataset."}

        all_tokens = []
        for text in df[text_column]:
            if pd.isna(text) or not str(text).strip():
                all_tokens.append([])
            else:
                tokens = nltk.word_tokenize(str(text).lower())
                all_tokens.append(tokens)

        df["unigrams"] = all_tokens
        session.artifacts["tokens"] = all_tokens

        total_documents = len(df)
        total_tokens = sum(len(t) for t in all_tokens)
        sample_tokens = all_tokens[0] if len(all_tokens) > 0 else []

        result_id = session.next_result_id("tokenize")

        summary = {
            "result_id": result_id,
            "total_documents": total_documents,
            "total_tokens": total_tokens,
            "sample_tokens": sample_tokens,
        }

        session.store_result("tokenize_tool", {"text_column": text_column}, summary)

        return summary

    return [check_missing_tool, check_duplicates_tool, tokenize_tool]
