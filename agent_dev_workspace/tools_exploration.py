import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity
from langchain_core.tools import tool

def make_tools(session):
    @tool
    def cosine_similarity_tool(doc_index_1: int, doc_index_2: int) -> dict:
        """Compute the cosine similarity between two documents by their row indices in session.dataframe.

        Args:
            doc_index_1 (int): Zero-based row index (0 to len(session.dataframe)-1) of the first document in session.dataframe.
            doc_index_2 (int): Zero-based row index (0 to len(session.dataframe)-1) of the second document in session.dataframe.

        Returns:
            dict: Summary containing result_id, document indices, and the cosine similarity score.
        """
        if session.dataframe is None:
            return {"error": "No dataframe found in session state. Please load a dataset first."}

        vectorizer = session.artifacts.get("count_vectorizer")
        if vectorizer is None:
            return {"error": "No count_vectorizer found in session.artifacts. Please run build_dtm_tool first."}

        if "text" not in session.dataframe.columns:
            return {"error": "DataFrame does not contain a 'text' column."}

        n_docs = len(session.dataframe)
        if not (0 <= doc_index_1 < n_docs) or not (0 <= doc_index_2 < n_docs):
            return {"error": f"Document indices must be between 0 and {n_docs - 1}."}

        text1 = session.dataframe.iloc[doc_index_1]["text"]
        text2 = session.dataframe.iloc[doc_index_2]["text"]

        # Handle None or non-string values safely
        text1_str = "" if text1 is None or (isinstance(text1, float) and np.isnan(text1)) else str(text1)
        text2_str = "" if text2 is None or (isinstance(text2, float) and np.isnan(text2)) else str(text2)

        vecs = vectorizer.transform([text1_str, text2_str])
        score = cosine_similarity(vecs[0], vecs[1])[0][0]

        result_id = session.next_result_id("cosine_similarity")
        summary = {
            "result_id": result_id,
            "doc_index_1": doc_index_1,
            "doc_index_2": doc_index_2,
            "similarity_score": round(float(score), 4)
        }

        session.store_result(
            tool_name="cosine_similarity_tool",
            args={"doc_index_1": doc_index_1, "doc_index_2": doc_index_2},
            summary=summary
        )

        return summary

    @tool
    def feature_correlation_matrix_tool() -> dict:
        """Compute the feature-vs-feature Pearson correlation matrix across the top 20 terms by variance in the DTM and generate a heatmap plot.

        Returns:
            dict: Summary containing result_id, list of top term names, correlation matrix (nested list), and plot metadata.
        """
        if session.feature_matrix is None or session.feature_names is None:
            return {"error": "No document-term matrix found. Please run build_dtm_tool first."}

        dtm = session.feature_matrix
        feature_names = list(session.feature_names)
        n_docs, n_terms = dtm.shape

        if n_terms == 0:
            return {"error": "Document-term matrix has no features."}

        # 1. Compute per-term variance sparsely without converting the whole DTM to dense array
        # Var(X) = E[X^2] - (E[X])^2
        # For sparse matrix dtm (shape: n_docs, n_terms):
        mean = np.asarray(dtm.mean(axis=0)).ravel()
        # Compute mean of squares for sparse matrix
        mean_sq = np.asarray(dtm.power(2).mean(axis=0)).ravel()
        variances = mean_sq - (mean ** 2)

        # 2. Select top 20 terms by variance
        top_k = min(20, n_terms)
        top_indices = np.argsort(variances)[::-1][:top_k]

        top_feature_names = [feature_names[i] for i in top_indices]

        # Slice ONLY the top 20 columns from sparse DTM and convert just that small slice (n_docs x <=20) to dense array
        sub_matrix_dense = dtm[:, top_indices].toarray()

        # 3. Compute Pearson correlation matrix across these top-k terms
        corr_matrix = np.corrcoef(sub_matrix_dense, rowvar=False)

        # Handle edge case if top_k == 1 (returns scalar float)
        if top_k == 1:
            corr_matrix = np.array([[1.0]])

        # 4. Generate heatmap plot and set on session.pending_figure
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(
            corr_matrix,
            annot=True,
            fmt=".2f",
            xticklabels=top_feature_names,
            yticklabels=top_feature_names,
            cmap="coolwarm",
            ax=ax
        )
        ax.set_title("Feature Correlation Heatmap (Top Terms by Variance)")
        plt.tight_layout()

        session.pending_figure = fig

        # Prepare JSON-serializable correlation matrix rounded to 4 decimal places
        corr_matrix_rounded = np.round(corr_matrix, 4).tolist()

        result_id = session.next_result_id("feature_correlation")
        summary = {
            "result_id": result_id,
            "terms": top_feature_names,
            "correlation_matrix": corr_matrix_rounded,
            "n_terms": top_k
        }

        session.store_result(
            tool_name="feature_correlation_matrix_tool",
            args={},
            summary=summary
        )

        return summary

    return [cosine_similarity_tool, feature_correlation_matrix_tool]
