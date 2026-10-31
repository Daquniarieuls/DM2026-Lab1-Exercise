import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from typing import Optional, Literal, Dict, Any, List
from langchain_core.tools import tool

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import LabelBinarizer
import umap


def make_tools(session) -> List[Any]:
    """Factory function creating tools bound to the given session."""

    @tool
    def reduce_dimensions_tool(
        method: Literal["pca", "tsne", "umap"] = "pca",
        perplexity: float = 30.0,
        n_neighbors: int = 15,
        random_state: Optional[int] = 42,
    ) -> Dict[str, Any]:
        """Reduces the dimensionality of document-term matrix features to 2D coordinates using PCA, t-SNE, or UMAP.
        
        Plots the 2D embedding scatter plot colored by category on session.pending_figure.

        Args:
            method: Reduction algorithm to use ("pca", "tsne", or "umap"). Default is "pca".
            perplexity: Perplexity parameter for t-SNE (used when method="tsne"). Default is 30.0.
            n_neighbors: Number of neighbors parameter for UMAP (used when method="umap"). Default is 15.
            random_state: Random seed for reproducibility. Default is 42.

        Returns:
            Dict containing result summary, reduced 2D coordinates (capped at 30 docs), method, and variance explained (for PCA).
        """
        if session.feature_matrix is None or session.dataframe is None:
            return {
                "error": "No feature matrix found. Please load a dataset and run build_dtm_tool first."
            }

        X = session.feature_matrix
        if hasattr(X, "toarray"):
            X_dense = X.toarray()
        else:
            X_dense = np.asarray(X)

        n_samples = X_dense.shape[0]

        # Fit selected reduction technique down to 2 dimensions
        explained_variance_ratio = None
        if method == "pca":
            reducer = PCA(n_components=2, random_state=random_state)
            coords = reducer.fit_transform(X_dense)
            explained_variance_ratio = reducer.explained_variance_ratio_.tolist()
        elif method == "tsne":
            if perplexity >= n_samples:
                return {
                    "error": f"Perplexity ({perplexity}) must be strictly less than the number of samples ({n_samples})."
                }
            reducer = TSNE(
                n_components=2,
                perplexity=perplexity,
                random_state=random_state
            )
            coords = reducer.fit_transform(X_dense)
        elif method == "umap":
            if n_neighbors >= n_samples:
                return {
                    "error": f"n_neighbors ({n_neighbors}) must be strictly less than the number of samples ({n_samples})."
                }
            reducer = umap.UMAP(
                n_components=2,
                n_neighbors=n_neighbors,
                random_state=random_state
            )
            coords = reducer.fit_transform(X_dense)
        else:
            return {
                "error": f"Unsupported method '{method}'. Choose from 'pca', 'tsne', or 'umap'."
            }

        # Store full coordinates array in artifacts
        session.artifacts["reduced_coordinates"] = coords

        # Prepare scatter plot for pending_figure requirement
        fig, ax = plt.subplots(figsize=(8, 6))
        
        categories = session.dataframe.get("category_name")
        if categories is None:
            categories = session.dataframe.get("category", ["default"] * n_samples)
        
        palette = sns.color_palette("Set2", n_colors=len(set(categories)))
        sns.scatterplot(
            x=coords[:, 0],
            y=coords[:, 1],
            hue=categories,
            palette=palette,
            s=70,
            ax=ax
        )
        
        ax.set_title(f"2D Document Embeddings ({method.upper()})")
        ax.set_xlabel("Dimension 1")
        ax.set_ylabel("Dimension 2")
        ax.grid(True, linestyle="--", alpha=0.5)
        
        session.pending_figure = fig

        # Limit coordinates in the summary dictionary to at most 30 documents
        coords_summary = coords[:30].tolist()
        
        result_id = session.next_result_id("reduce_dimensions")
        summary = {
            "result_id": result_id,
            "method": method,
            "n_documents": n_samples,
            "coordinates": coords_summary
        }

        if n_samples > 30:
            summary["note"] = f"Showing coordinates for the first 30 of {n_samples} documents. Full coordinates stored in session.artifacts['reduced_coordinates']."

        if explained_variance_ratio is not None:
            summary["explained_variance_ratio"] = explained_variance_ratio

        # Construct full_report as a pandas DataFrame containing coordinates and category
        full_report = pd.DataFrame({
            "dim1": coords[:, 0],
            "dim2": coords[:, 1],
            "category": categories
        })

        session.store_result("reduce_dimensions_tool", {"method": method}, summary, full_report=full_report)
        return summary

    @tool
    def binarize_labels_tool() -> Dict[str, Any]:
        """One-hot binarizes category labels in session.dataframe using sklearn LabelBinarizer.
        
        Stores the resulting binary matrix in session.artifacts['binarized_labels']
        without overwriting session.labels or mutating dataframe columns.

        Returns:
            Dict summary containing result_id, classes, sample count, and binarized indicator preview.
        """
        if session.dataframe is None:
            return {
                "error": "No dataframe found. Please load a dataset first."
            }

        target_col = "category_name"
        if target_col not in session.dataframe.columns:
            if "category" in session.dataframe.columns:
                target_col = "category"
            else:
                return {
                    "error": "No category or category_name column found in dataframe."
                }

        labels_series = session.dataframe[target_col].astype(str)
        
        binarizer = LabelBinarizer()
        binary_matrix = binarizer.fit_transform(labels_series)
        classes = binarizer.classes_.tolist()

        # Handle binary (1-column output from LabelBinarizer) by expanding to 2 classes if binary
        if len(classes) == 2 and binary_matrix.shape[1] == 1:
            binary_matrix = np.hstack([1 - binary_matrix, binary_matrix])

        # Store in artifacts without touching session.labels or adding dataframe columns
        session.artifacts["binarized_labels"] = binary_matrix
        session.artifacts["binarizer_classes"] = classes

        coords_preview = binary_matrix[:30].tolist()
        result_id = session.next_result_id("binarize_labels")

        summary = {
            "result_id": result_id,
            "column": target_col,
            "classes": classes,
            "n_samples": len(binary_matrix),
            "binarized_preview": coords_preview
        }

        full_report = pd.DataFrame(binary_matrix, columns=classes)
        session.store_result("binarize_labels_tool", {}, summary, full_report=full_report)

        return summary

    return [reduce_dimensions_tool, binarize_labels_tool]
