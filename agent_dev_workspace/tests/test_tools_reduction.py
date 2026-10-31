import numpy as np
import pytest

from agent_pipeline.session_state import SessionState
from agent_pipeline.workspace_loader import load_workspace_tools
from tools_dtm import make_tools as make_dtm_tools
from tools_reduction import make_tools as make_reduction_tools


def setup_fixture_session():
    """Helper to initialize fresh SessionState, load sample_fixture.csv, and build DTM."""
    session = SessionState()
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")
    
    # Load dataset
    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })
    
    # Build DTM
    dtm_tools = make_dtm_tools(session)
    build_dtm_tool = next(t for t in dtm_tools if t.name == "build_dtm_tool")
    build_dtm_tool.invoke({})
    
    return session


def test_reduce_dimensions_no_dtm_error():
    """Verify tool returns error dict when feature_matrix is missing."""
    session = SessionState()
    reduction_tools = make_reduction_tools(session)
    reduce_tool = next(t for t in reduction_tools if t.name == "reduce_dimensions_tool")
    
    res = reduce_tool.invoke({"method": "pca"})
    assert "error" in res
    assert "No feature matrix found" in res["error"]


def test_reduce_dimensions_perplexity_error():
    """Verify tool returns error dict when perplexity >= n_samples."""
    session = setup_fixture_session()
    reduction_tools = make_reduction_tools(session)
    reduce_tool = next(t for t in reduction_tools if t.name == "reduce_dimensions_tool")
    
    # sample_fixture.csv has 8 documents; perplexity=30 should trigger an error
    res = reduce_tool.invoke({"method": "tsne", "perplexity": 30.0})
    assert "error" in res
    assert "strictly less than the number of samples" in res["error"]


def test_reduce_dimensions_pca():
    """Check PCA coordinates and explained variance ratio against Known Answer #7."""
    session = setup_fixture_session()
    reduction_tools = make_reduction_tools(session)
    reduce_tool = next(t for t in reduction_tools if t.name == "reduce_dimensions_tool")
    
    res = reduce_tool.invoke({"method": "pca"})
    assert "error" not in res
    assert session.pending_figure is not None
    
    expected_explained_var = [0.7532, 0.1965]
    assert np.allclose(res["explained_variance_ratio"], expected_explained_var, atol=1e-3)
    
    expected_coords = np.array([
        [2.3669, 0.8871],
        [1.7039, 0.2642],
        [1.0408, -0.3587],
        [-2.0759, 1.0129],
        [-1.4554, 0.3405],
        [-0.8349, -0.3319],
        [-0.8349, -0.3319],
        [0.0894, -1.4822]
    ])
    
    coords = np.array(res["coordinates"])
    assert np.allclose(coords, expected_coords, atol=1e-3)


def test_reduce_dimensions_tsne():
    """Check t-SNE coordinates against Known Answer #8."""
    session = setup_fixture_session()
    reduction_tools = make_reduction_tools(session)
    reduce_tool = next(t for t in reduction_tools if t.name == "reduce_dimensions_tool")
    
    res = reduce_tool.invoke({"method": "tsne", "perplexity": 2, "random_state": 42})
    assert "error" not in res
    assert session.pending_figure is not None
    
    expected_coords = np.array([
        [1330.9991, 32.1039],
        [951.2459, 393.5588],
        [530.8209, 682.9791],
        [-1558.6448, -314.1625],
        [-915.5153, -346.3907],
        [-280.0313, -1261.2115],
        [-280.0313, -1261.2115],
        [-701.3674, 1465.3973]
    ])
    
    coords = np.array(res["coordinates"])
    assert np.allclose(coords, expected_coords, atol=1.0)


def test_reduce_dimensions_umap():
    """Check UMAP coordinates against Known Answer #9."""
    session = setup_fixture_session()
    reduction_tools = make_reduction_tools(session)
    reduce_tool = next(t for t in reduction_tools if t.name == "reduce_dimensions_tool")
    
    res = reduce_tool.invoke({"method": "umap", "n_neighbors": 3, "random_state": 42})
    assert "error" not in res
    assert session.pending_figure is not None
    
    expected_coords = np.array([
        [6.6811, -6.8218],
        [6.9907, -7.2245],
        [7.7022, -7.3712],
        [10.2228, -6.2310],
        [9.8759, -5.6537],
        [9.3831, -6.5748],
        [9.2153, -5.8297],
        [8.5860, -7.2622]
    ])
    
    coords = np.array(res["coordinates"])
    assert np.allclose(coords, expected_coords, atol=1e-1)


def test_binarize_labels_no_df_error():
    """Verify tool returns error when session.dataframe is missing."""
    session = SessionState()
    reduction_tools = make_reduction_tools(session)
    binarize_tool = next(t for t in reduction_tools if t.name == "binarize_labels_tool")
    
    res = binarize_tool.invoke({})
    assert "error" in res
    assert "No dataframe found" in res["error"]


def test_binarize_labels():
    """Test binarize_labels_tool on sample_fixture.csv dataset."""
    session = setup_fixture_session()
    original_cols = list(session.dataframe.columns)
    original_labels = session.labels
    
    reduction_tools = make_reduction_tools(session)
    binarize_tool = next(t for t in reduction_tools if t.name == "binarize_labels_tool")
    
    res = binarize_tool.invoke({})
    assert "error" not in res
    assert res["n_samples"] == 8
    assert res["classes"] == ["catA", "catB"]
    
    # Verify matrix stored in artifacts
    matrix = session.artifacts.get("binarized_labels")
    assert matrix is not None
    assert matrix.shape == (8, 2)
    
    # Rows 0-2 & 7 are catA ([1, 0]), rows 3-6 are catB ([0, 1])
    expected_matrix = np.array([
        [1, 0],  # row 0 catA
        [1, 0],  # row 1 catA
        [1, 0],  # row 2 catA
        [0, 1],  # row 3 catB
        [0, 1],  # row 4 catB
        [0, 1],  # row 5 catB
        [0, 1],  # row 6 catB
        [1, 0]   # row 7 catA
    ])
    assert np.array_equal(matrix, expected_matrix)
    
    # Assert dataframe columns were not mutated and session.labels was not overwritten
    assert list(session.dataframe.columns) == original_cols
    assert session.labels is original_labels
