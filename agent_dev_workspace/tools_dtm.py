import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
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

    @tool
    def term_frequency_tool(
        top_n: Optional[int] = 15
    ) -> Dict[str, Any]:
        """Aggregates term frequencies across all documents in session.feature_matrix.

        Sums counts for each term across all documents, sort terms descending by total frequency
        while preserving original vocabulary order for ties, and returns up to top_n terms.

        Args:
            top_n: Maximum number of top terms to return (default: 15).

        Returns:
            Dict containing result_id, total_terms, and frequencies (term-to-frequency map).
        """
        if session.dataframe is None or session.dataframe.empty:
            return {"error": "No dataframe loaded in session. Load dataset first."}

        if session.feature_matrix is None or session.feature_names is None:
            return {"error": "No feature_matrix built in session. Run build_dtm_tool first."}

        # Sum counts across all documents for each column
        counts = np.asarray(session.feature_matrix.sum(axis=0)).ravel()
        
        # Build list of (original_index, term, count)
        items = [
            (idx, term, int(count))
            for idx, (term, count) in enumerate(zip(session.feature_names, counts))
        ]

        # Stable sort by count descending, preserving original_index order for ties
        items_sorted = sorted(items, key=lambda x: (-x[2], x[0]))

        # Limit to top_n if specified and larger than total items
        if top_n is not None and len(items_sorted) > top_n:
            items_sorted = items_sorted[:top_n]

        frequencies = {term: count for _, term, count in items_sorted}

        result_id = session.next_result_id("tf")
        summary = {
            "result_id": result_id,
            "total_terms": len(session.feature_names),
            "returned_terms": len(frequencies),
            "frequencies": frequencies
        }

        session.store_result("term_frequency_tool", {"top_n": top_n}, summary)
        return summary

    @tool
    def dtm_heatmap_tool(
        n_terms: int = 20,
        n_documents: int = 20
    ) -> Dict[str, Any]:
        """Renders a heatmap of a raw positional slice of session.feature_matrix.

        Extracts the first n_documents rows and n_terms columns from session.feature_matrix,
        renders an annotated heatmap figure stored on session.pending_figure, and returns
        the sliced matrix values as a nested list.

        Args:
            n_terms: Number of terms (columns) to slice from session.feature_names (default: 20).
            n_documents: Number of documents (rows) to slice from session.dataframe (default: 20).

        Returns:
            Dict containing result_id, n_documents, n_terms, terms, document_indices, and matrix_slice.
        """
        if session.dataframe is None or session.dataframe.empty:
            return {"error": "No dataframe loaded in session. Load dataset first."}

        if session.feature_matrix is None or session.feature_names is None:
            return {"error": "No feature_matrix built in session. Run build_dtm_tool first."}

        total_docs, total_terms = session.feature_matrix.shape
        actual_docs = min(n_documents, total_docs)
        actual_terms = min(n_terms, total_terms)

        matrix_slice = session.feature_matrix[:actual_docs, :actual_terms].toarray()
        term_labels = session.feature_names[:actual_terms]
        doc_indices = list(range(actual_docs))

        # Build figure and render heatmap
        fig, ax = plt.subplots(figsize=(max(6, actual_terms * 0.8), max(4, actual_docs * 0.4)))
        sns.heatmap(
            matrix_slice,
            annot=True,
            fmt="d",
            cmap="YlGnBu",
            xticklabels=term_labels,
            yticklabels=[f"Doc {i}" for i in doc_indices],
            ax=ax
        )
        ax.set_title(f"DTM Heatmap (First {actual_docs} Docs x {actual_terms} Terms)")
        ax.set_xlabel("Terms")
        ax.set_ylabel("Documents")
        plt.tight_layout()

        session.pending_figure = fig

        result_id = session.next_result_id("dtm_heatmap")
        summary = {
            "result_id": result_id,
            "n_documents": actual_docs,
            "n_terms": actual_terms,
            "terms": term_labels,
            "document_indices": doc_indices,
            "matrix_slice": matrix_slice.tolist()
        }

        session.store_result("dtm_heatmap_tool", {"n_terms": n_terms, "n_documents": n_documents}, summary)
        return summary

    return [build_dtm_tool, term_frequency_tool, dtm_heatmap_tool]
