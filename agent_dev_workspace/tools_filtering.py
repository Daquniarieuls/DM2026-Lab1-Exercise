import numpy as np
import pandas as pd
from scipy.sparse import issparse
from langchain_core.tools import tool


def make_tools(session):
    @tool
    def variance_filter_tool(threshold: float = 0.0) -> dict:
        """Flags near-constant terms by variance for removal.

        Args:
            threshold: Terms with variance less than or equal to this threshold
                are flagged for removal.

        Returns:
            A dict summary of the variance analysis including kept/removed terms,
            total terms analyzed, and the result_id.
        """
        if session.feature_matrix is None or session.feature_names is None:
            return {
                "status": "error",
                "message": "No document-term matrix found in session. Please run build_dtm_tool first."
            }

        X = session.feature_matrix
        feature_names = session.feature_names

        # Compute variance across documents (rows) for each term column
        if issparse(X):
            # For sparse matrix: Var(X) = E[X^2] - (E[X])^2
            mean = np.array(X.mean(axis=0)).ravel()
            squared_mean = np.array(X.power(2).mean(axis=0)).ravel()
            variances = squared_mean - (mean ** 2)
        else:
            variances = np.var(X, axis=0)

        # Build complete full report DataFrame required by visualize_result_tool
        # Column 1: term/feature name, Column 2: variance
        report_df = pd.DataFrame({
            "feature": feature_names,
            "variance": variances
        })

        # Sort descending by variance so visualize_result_tool can chart top terms directly
        report_df = report_df.sort_values(by="variance", ascending=False).reset_index(drop=True)

        # Flag terms below or equal to threshold vs above threshold
        kept_df = report_df[report_df["variance"] > threshold]
        removed_df = report_df[report_df["variance"] <= threshold]

        all_kept_terms = kept_df["feature"].tolist()
        all_removed_terms = removed_df["feature"].tolist()

        # Truncate lists in summary to at most 50 terms to keep summary small
        max_summary_terms = 50
        kept_terms_summary = all_kept_terms[:max_summary_terms]
        removed_terms_summary = all_removed_terms[:max_summary_terms]

        result_id = session.next_result_id("variance")

        summary = {
            "result_id": result_id,
            "status": "success",
            "threshold": threshold,
            "total_terms": len(feature_names),
            "n_kept": len(all_kept_terms),
            "n_removed": len(all_removed_terms),
            "kept_terms": kept_terms_summary,
            "removed_terms": removed_terms_summary,
            "kept_terms_truncated": len(all_kept_terms) > max_summary_terms,
            "removed_terms_truncated": len(all_removed_terms) > max_summary_terms,
        }

        if len(all_kept_terms) > max_summary_terms:
            summary["kept_terms_note"] = f"Showing first {max_summary_terms} of {len(all_kept_terms)} kept terms; see full_report for complete list."
        if len(all_removed_terms) > max_summary_terms:
            summary["removed_terms_note"] = f"Showing first {max_summary_terms} of {len(all_removed_terms)} removed terms; see full_report for complete list."

        # Store complete full report for visualize_result_tool or downstream reference
        session.store_result(
            tool_name="variance_filter_tool",
            args={"threshold": threshold},
            summary=summary,
            full_report=report_df
        )

        return summary

    return [variance_filter_tool]
