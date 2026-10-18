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
