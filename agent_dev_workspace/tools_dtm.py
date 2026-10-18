import numpy as np
import pandas as pd
from typing import Optional, List, Dict, Any
from sklearn.feature_extraction.text import CountVectorizer
from langchain_core.tools import tool

def make_tools(session):
    @tool
    def build_dtm_tool(
        max_features: Optional[int] = None,
        ngram_range: List[int] = [1, 1],
        stop_words: Optional[str] = None
    ) -> Dict[str, Any]:
        """Builds a Document-Term Matrix (DTM) from session.dataframe text data.

        Converts raw text into a bag-of-words document-term matrix using CountVectorizer.
        Updates session.feature_matrix, session.feature_names, session.artifacts["count_vectorizer"],
        and session.set_labels. Reports matrix sparsity summary stats.

        Args:
            max_features: Optional maximum vocabulary size, keeping top terms by corpus frequency.
            ngram_range: Lower and upper boundary of range of n-grams, as a list of 2 integers, e.g. [1, 1].
            stop_words: Language stop words to remove, e.g. "english" or None.

        Returns:
            Dict containing result_id, n_documents, n_terms, non_zero, total_elements, and sparsity_pct.
        """
        if session.dataframe is None or session.dataframe.empty:
            return {"error": "No dataframe loaded in session. Load dataset first."}

        # Convert n_gram range list to tuple for sklearn
        ngram_tuple = (ngram_range[0], ngram_range[1]) if len(ngram_range) >= 2 else (1, 1)

        texts = session.dataframe["text"].fillna("").tolist()

        vectorizer = CountVectorizer(
            max_features=max_features,
            ngram_range=ngram_tuple,
            stop_words=stop_words
        )
        feature_matrix = vectorizer.fit_transform(texts)
        feature_names = vectorizer.get_feature_names_out().tolist()

        session.feature_matrix = feature_matrix
        session.feature_names = feature_names
        session.artifacts["count_vectorizer"] = vectorizer

        if "category_name" in session.dataframe.columns:
            session.set_labels(session.dataframe["category_name"].values)

        n_rows, n_cols = feature_matrix.shape
        non_zero = int(feature_matrix.nnz)
        total_elements = n_rows * n_cols
        sparsity_pct = float(100.0 * (1.0 - non_zero / total_elements)) if total_elements > 0 else 0.0

        result_id = session.next_result_id("dtm")
        summary = {
            "result_id": result_id,
            "n_documents": n_rows,
            "n_terms": n_cols,
            "non_zero": non_zero,
            "total_elements": total_elements,
            "sparsity_pct": round(sparsity_pct, 4)
        }

        session.store_result("build_dtm_tool", {"max_features": max_features, "ngram_range": ngram_range, "stop_words": stop_words}, summary)
        return summary

    return [build_dtm_tool]
