import numpy as np
import pandas as pd
from scipy.sparse import issparse
from scipy.stats import rankdata
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

    @tool
    def pearson_filter_tool(target_class: str, threshold: float = 0.0) -> dict:
        """Computes Pearson correlation between each term's count and a target class.

        Args:
            target_class: Category label to compute correlation against (one-vs-rest binary encoding).
            threshold: Terms with absolute Pearson correlation coefficient |r| >= threshold are kept.

        Returns:
            A dict summary of the correlation analysis including kept/removed terms,
            total terms analyzed, target class, and result_id.
        """
        if session.feature_matrix is None or session.feature_names is None:
            return {
                "status": "error",
                "message": "No document-term matrix found in session. Please run build_dtm_tool first."
            }

        if session.dataframe is None or "category_name" not in session.dataframe.columns:
            return {
                "status": "error",
                "message": "No valid session dataframe with 'category_name' column found."
            }

        available_categories = session.dataframe["category_name"].unique().tolist()
        if target_class not in available_categories:
            return {
                "status": "error",
                "message": f"Target class '{target_class}' not found in dataset categories. Available categories: {available_categories}"
            }

        labels = session.dataframe["category_name"].values
        y = (labels == target_class).astype(float)

        X = session.feature_matrix
        feature_names = session.feature_names

        # Convert X to dense array for vectorized correlation matrix operations
        if issparse(X):
            X_dense = X.toarray()
        else:
            X_dense = np.asarray(X)

        # Vectorized Pearson correlation calculation
        # Mean-center variables
        X_mean = X_dense.mean(axis=0)
        y_mean = y.mean()

        X_centered = X_dense - X_mean
        y_centered = y - y_mean

        # Standard deviations (N-scaled matching population std / cov)
        std_X = np.std(X_dense, axis=0)
        std_y = np.std(y)

        # Covariance vector across all feature columns simultaneously: Cov(X, y) = mean((X - mu_X) * (y - mu_y))
        cov_X_y = np.dot(y_centered, X_centered) / len(y)

        # Vectorized division with zero-std check handling
        with np.errstate(divide='ignore', invalid='ignore'):
            denom = std_X * std_y
            pearson_r_array = cov_X_y / denom
            pearson_r_array = np.nan_to_num(pearson_r_array, nan=0.0, posinf=0.0, neginf=0.0)

        pearson_r_list = pearson_r_array.tolist()

        # Build full report DataFrame matching required spec contract:
        # First column: term/feature name, Second column: pearson_r value
        report_df = pd.DataFrame({
            "feature": feature_names,
            "pearson_r": pearson_r_list
        })

        # Sort by absolute correlation magnitude |r| descending
        report_df["abs_r"] = report_df["pearson_r"].abs()
        report_df = report_df.sort_values(by="abs_r", ascending=False).drop(columns=["abs_r"]).reset_index(drop=True)

        # Filter terms based on absolute correlation magnitude |r| >= threshold
        kept_df = report_df[report_df["pearson_r"].abs() >= threshold]
        removed_df = report_df[report_df["pearson_r"].abs() < threshold]

        all_kept_terms = kept_df["feature"].tolist()
        all_removed_terms = removed_df["feature"].tolist()

        # Truncate lists in summary to at most 50 terms to keep summary JSON payload small
        max_summary_terms = 50
        kept_terms_summary = all_kept_terms[:max_summary_terms]
        removed_terms_summary = all_removed_terms[:max_summary_terms]

        result_id = session.next_result_id("pearson")

        summary = {
            "result_id": result_id,
            "status": "success",
            "target_class": target_class,
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

        session.store_result(
            tool_name="pearson_filter_tool",
            args={"target_class": target_class, "threshold": threshold},
            summary=summary,
            full_report=report_df
        )

        return summary

    @tool
    def spearman_filter_tool(target_class: str, threshold: float = 0.0, max_terms: int = 1000) -> dict:
        """Computes Spearman rank correlation between each term's count and a target class.

        To optimize performance on large vocabularies, terms can first be filtered to the
        highest-variance columns before calculating rank correlations.

        Args:
            target_class: Category label to compute correlation against (one-vs-rest binary encoding).
            threshold: Terms with absolute Spearman correlation coefficient |r| >= threshold are kept.
            max_terms: Maximum number of highest-variance terms to analyze (default 1000). Set to 0 or None for no limit.

        Returns:
            A dict summary of the correlation analysis including kept/removed terms,
            total terms analyzed, target class, and result_id.
        """
        if session.feature_matrix is None or session.feature_names is None:
            return {
                "status": "error",
                "message": "No document-term matrix found in session. Please run build_dtm_tool first."
            }

        if session.dataframe is None or "category_name" not in session.dataframe.columns:
            return {
                "status": "error",
                "message": "No valid session dataframe with 'category_name' column found."
            }

        available_categories = session.dataframe["category_name"].unique().tolist()
        if target_class not in available_categories:
            return {
                "status": "error",
                "message": f"Target class '{target_class}' not found in dataset categories. Available categories: {available_categories}"
            }

        labels = session.dataframe["category_name"].values
        y = (labels == target_class).astype(float)

        X = session.feature_matrix
        feature_names = session.feature_names

        # Optional variance pre-filtering for speed if total terms exceeds max_terms
        total_terms_count = len(feature_names)
        if max_terms and max_terms > 0 and total_terms_count > max_terms:
            if issparse(X):
                mean = np.array(X.mean(axis=0)).ravel()
                squared_mean = np.array(X.power(2).mean(axis=0)).ravel()
                variances = squared_mean - (mean ** 2)
            else:
                variances = np.var(X, axis=0)

            top_indices = np.argsort(variances)[::-1][:max_terms]
            # Keep indices in sorted order or column order
            top_indices = np.sort(top_indices)
            if issparse(X):
                X_sub = X[:, top_indices].toarray()
            else:
                X_sub = np.asarray(X)[:, top_indices]
            sub_feature_names = [feature_names[i] for i in top_indices]
        else:
            if issparse(X):
                X_sub = X.toarray()
            else:
                X_sub = np.asarray(X)
            sub_feature_names = feature_names

        # Convert term counts and target binary indicators to ranks (axis=0 for columns)
        X_ranked = rankdata(X_sub, method="average", axis=0)
        y_ranked = rankdata(y, method="average")

        # Compute Pearson correlation on ranks (vectorized)
        X_mean = X_ranked.mean(axis=0)
        y_mean = y_ranked.mean()

        X_centered = X_ranked - X_mean
        y_centered = y_ranked - y_mean

        std_X = np.std(X_ranked, axis=0)
        std_y = np.std(y_ranked)

        cov_X_y = np.dot(y_centered, X_centered) / len(y_ranked)

        with np.errstate(divide='ignore', invalid='ignore'):
            denom = std_X * std_y
            spearman_r_array = cov_X_y / denom
            spearman_r_array = np.nan_to_num(spearman_r_array, nan=0.0, posinf=0.0, neginf=0.0)

        spearman_r_list = spearman_r_array.tolist()

        # Build full report DataFrame matching spec contract:
        # First column: term/feature name, Second column: spearman_r value
        report_df = pd.DataFrame({
            "feature": sub_feature_names,
            "spearman_r": spearman_r_list
        })

        # Sort by absolute correlation magnitude |r| descending
        report_df["abs_r"] = report_df["spearman_r"].abs()
        report_df = report_df.sort_values(by="abs_r", ascending=False).drop(columns=["abs_r"]).reset_index(drop=True)

        # Filter terms based on absolute correlation magnitude |r| >= threshold
        kept_df = report_df[report_df["spearman_r"].abs() >= threshold]
        removed_df = report_df[report_df["spearman_r"].abs() < threshold]

        all_kept_terms = kept_df["feature"].tolist()
        all_removed_terms = removed_df["feature"].tolist()

        max_summary_terms = 50
        kept_terms_summary = all_kept_terms[:max_summary_terms]
        removed_terms_summary = all_removed_terms[:max_summary_terms]

        result_id = session.next_result_id("spearman")

        summary = {
            "result_id": result_id,
            "status": "success",
            "target_class": target_class,
            "threshold": threshold,
            "max_terms_analyzed": len(sub_feature_names),
            "total_terms": total_terms_count,
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

        session.store_result(
            tool_name="spearman_filter_tool",
            args={"target_class": target_class, "threshold": threshold, "max_terms": max_terms},
            summary=summary,
            full_report=report_df
        )

        return summary

    return [variance_filter_tool, pearson_filter_tool, spearman_filter_tool]
