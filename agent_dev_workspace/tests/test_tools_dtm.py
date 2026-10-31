import pytest
import numpy as np
from agent_pipeline.session_state import SessionState
from agent_pipeline.workspace_loader import load_workspace_tools
from tools_dtm import make_tools

def test_build_dtm_tool():
    # 1. Instantiate state
    session = SessionState()

    # 2. Load premade load_dataset_tool
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")

    # Load dataset fixture
    load_res = load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    # 3. Instantiate build_dtm_tool from student's workspace module
    tools = make_tools(session)
    build_dtm_tool = next(t for t in tools if t.name == "build_dtm_tool")

    # Execute build_dtm_tool with defaults
    res = build_dtm_tool.invoke({})

    # Check known answer 4 specs from TEST_FIXTURE.md
    # Vocabulary: alpha, always, beta, delta, gamma -> 5 terms
    # Sparsity: 21 non-zero cells out of 40 total (8 rows * 5 cols) -> sparsity_pct = 47.5
    assert res["n_documents"] == 8
    assert res["n_terms"] == 5
    assert res["non_zero"] == 21
    assert res["total_elements"] == 40
    assert pytest.approx(res["sparsity_pct"], abs=1e-3) == 47.5

    # Check session state populated correctly
    assert session.feature_matrix is not None
    assert session.feature_matrix.shape == (8, 5)
    assert session.feature_names == ["alpha", "always", "beta", "delta", "gamma"]
    assert "count_vectorizer" in session.artifacts


def test_term_frequency_tool():
    # Fresh session state for test independence
    session = SessionState()

    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")

    # Test error handling when no dataframe loaded
    tools = make_tools(session)
    term_frequency_tool = next(t for t in tools if t.name == "term_frequency_tool")
    err_res = term_frequency_tool.invoke({})
    assert "error" in err_res

    # Load dataset
    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    # Re-instantiate tools with updated session context
    tools = make_tools(session)
    build_dtm_tool = next(t for t in tools if t.name == "build_dtm_tool")
    term_frequency_tool = next(t for t in tools if t.name == "term_frequency_tool")

    # Build DTM first
    build_dtm_tool.invoke({})

    # Execute term_frequency_tool
    res = term_frequency_tool.invoke({})

    # Check known answer 14 specs from TEST_FIXTURE.md:
    # alpha: 6, always: 7, beta: 3, delta: 4, gamma: 7
    assert res["total_terms"] == 5
    assert res["returned_terms"] == 5

    freqs = res["frequencies"]
    assert freqs["alpha"] == 6
    assert freqs["always"] == 7
    assert freqs["beta"] == 3
    assert freqs["delta"] == 4
    assert freqs["gamma"] == 7

    # Check ordering: highest count first, ties preserved by feature index order
    keys = list(freqs.keys())
    assert keys == ["always", "gamma", "alpha", "delta", "beta"]


def test_dtm_heatmap_tool():
    session = SessionState()

    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")

    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    tools = make_tools(session)
    build_dtm_tool = next(t for t in tools if t.name == "build_dtm_tool")
    dtm_heatmap_tool = next(t for t in tools if t.name == "dtm_heatmap_tool")

    # Build DTM first
    build_dtm_tool.invoke({})

    # Default call: n_terms=20, n_documents=20 (exceeds fixture dimensions, returns full 8x5)
    res_full = dtm_heatmap_tool.invoke({})

    # Check Known Answer 21 specs from TEST_FIXTURE.md
    assert res_full["n_documents"] == 8
    assert res_full["n_terms"] == 5
    assert res_full["terms"] == ["alpha", "always", "beta", "delta", "gamma"]

    expected_full = [
        [3, 1, 1, 0, 0],
        [2, 1, 1, 0, 0],
        [1, 1, 1, 0, 0],
        [0, 1, 0, 1, 3],
        [0, 1, 0, 1, 2],
        [0, 1, 0, 1, 1],
        [0, 1, 0, 1, 1],
        [0, 0, 0, 0, 0]
    ]
    assert res_full["matrix_slice"] == expected_full
    assert session.pending_figure is not None

    # Slice call: n_terms=2, n_documents=3 (returns 3x2 top-left corner)
    res_slice = dtm_heatmap_tool.invoke({"n_terms": 2, "n_documents": 3})
    assert res_slice["n_documents"] == 3
    assert res_slice["n_terms"] == 2
    assert res_slice["terms"] == ["alpha", "always"]

    expected_slice = [
        [3, 1],
        [2, 1],
        [1, 1]
    ]
    assert res_slice["matrix_slice"] == expected_slice
