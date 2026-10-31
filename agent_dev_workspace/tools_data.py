import os
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
                    If None, all scalar (hashable) columns are checked.
            drop: Whether to drop duplicate rows from session.dataframe in place.

        Returns:
            A dictionary containing total_rows_before, duplicate_count, duplicate_indices,
            drop_requested, dropped status, removed_indices, total_rows_after, and result_id.
        """
        if session.dataframe is None:
            return {"error": "No dataset loaded in session. Please load a dataset first."}

        df = session.dataframe
        total_rows_before = len(df)

        # Filter out unhashable columns (such as list columns like 'unigrams') if subset is None
        if subset is None:
            valid_subset = []
            for col in df.columns:
                # Check first non-null element to see if it's unhashable (e.g. list, dict)
                non_nulls = df[col].dropna()
                if not non_nulls.empty and isinstance(non_nulls.iloc[0], (list, dict, set)):
                    continue
                valid_subset.append(col)
        else:
            valid_subset = subset

        # Count extra duplicate copies for reporting
        duplicate_count = int(df.duplicated(subset=valid_subset, keep="first").sum())

        # Mask identifying all copies of duplicated rows
        all_duplicates_mask = df.duplicated(subset=valid_subset, keep=False)
        duplicate_indices = df.index[all_duplicates_mask].tolist()

        actually_dropped = False
        removed_indices: List[Any] = []

        if drop and len(duplicate_indices) > 0:
            removed_indices = duplicate_indices.copy()
            df.drop_duplicates(subset=valid_subset, keep=False, inplace=True)
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

    @tool
    def inspect_data_tool(
        n: int = 5,
        max_text_len: int = 100,
        columns: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Peeks at the first n rows of the working DataFrame with original row indices and truncated text.

        Args:
            n: Number of rows to inspect from the top of the DataFrame.
            max_text_len: Maximum character length for string values before truncating with '...'.
            columns: Optional list of column names to inspect. If None, inspects all columns.

        Returns:
            A summary dictionary containing result_id, total_rows, n_rows, columns, and list of row dicts.
        """
        if session.dataframe is None:
            return {"error": "No dataset loaded in session. Please load a dataset first."}

        df = session.dataframe
        total_rows = len(df)

        cols_to_use = columns if columns is not None else list(df.columns)
        valid_cols = [c for c in cols_to_use if c in df.columns]

        preview_df = df.head(n)
        rows_data = []

        for idx, row in preview_df.iterrows():
            row_dict = {"_index": idx}
            for col in valid_cols:
                val = row[col]
                if isinstance(val, str) and len(val) > max_text_len:
                    row_dict[col] = val[:max_text_len] + "..."
                elif isinstance(val, (list, dict, set, tuple)):
                    row_dict[col] = val
                elif pd.isna(val):
                    row_dict[col] = None
                else:
                    row_dict[col] = val
            rows_data.append(row_dict)

        result_id = session.next_result_id("inspect_data")

        summary = {
            "result_id": result_id,
            "total_rows": total_rows,
            "n_rows": len(rows_data),
            "columns": valid_cols,
            "rows": rows_data,
        }

        session.store_result(
            "inspect_data_tool",
            {"n": n, "max_text_len": max_text_len, "columns": columns},
            summary
        )

        return summary

    @tool
    def list_files_tool(subdirectory: str = ".") -> Dict[str, Any]:
        """Lists files and directories in a given subdirectory relative to the repo root.

        Filters out hidden files/directories (starting with '.') and __pycache__. Ensures path remains inside repo root.

        Args:
            subdirectory: Path relative to repository root to list files/directories from.

        Returns:
            A summary dictionary containing result_id, subdirectory, directories, and files.
        """
        repo_root = os.path.abspath(".")
        target_dir = os.path.abspath(os.path.join(repo_root, subdirectory if subdirectory else "."))

        # Security check: ensure target_dir is inside or equal to repo_root (no directory traversal out of repo)
        if not (target_dir == repo_root or target_dir.startswith(repo_root + os.sep)):
            return {"error": f"Path '{subdirectory}' is outside the repository root."}

        if not os.path.exists(target_dir):
            return {"error": f"Subdirectory '{subdirectory}' does not exist."}

        if not os.path.isdir(target_dir):
            return {"error": f"Path '{subdirectory}' is not a directory."}

        entries = os.listdir(target_dir)

        directories = []
        files = []

        for entry in entries:
            # Skip hidden files/directories and __pycache__
            if entry.startswith(".") or entry == "__pycache__":
                continue

            full_path = os.path.join(target_dir, entry)
            if os.path.isdir(full_path):
                directories.append(entry)
            elif os.path.isfile(full_path):
                files.append(entry)

        directories.sort()
        files.sort()

        result_id = session.next_result_id("list_files")

        summary = {
            "result_id": result_id,
            "subdirectory": subdirectory,
            "directories": directories,
            "files": files,
        }

        session.store_result("list_files_tool", {"subdirectory": subdirectory}, summary)

        return summary

    return [
        check_missing_tool,
        check_duplicates_tool,
        tokenize_tool,
        inspect_data_tool,
        list_files_tool,
    ]
