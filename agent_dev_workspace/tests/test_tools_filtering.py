import numpy as np
import pytest

from agent_pipeline.session_state import SessionState
from agent_pipeline.workspace_loader import load_workspace_tools
from tools_dtm import make_tools as make_dtm_tools
from tools_filtering import make_tools as make_filtering_tools


def test_variance_filter_tool():
    # 1. Create a single session instance
    session = SessionState()

    # 2. Load dataset using premade load_dataset_tool
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")
    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    # 3. Build DTM using student's build_dtm_tool from tools_dtm.py
    dtm_tools = make_dtm_tools(session)
    build_dtm_tool = next(t for t in dtm_tools if t.name == "build_dtm_tool")
    dtm_result = build_dtm_tool.invoke({})
    assert "result_id" in dtm_result
    assert dtm_result["n_documents"] == 8
    assert dtm_result["n_terms"] == 5

    # 4. Initialize variance_filter_tool from tools_filtering.py
    filtering_tools = make_filtering_tools(session)
    variance_filter_tool = next(t for t in filtering_tools if t.name == "variance_filter_tool")

    # 5. Invoke variance_filter_tool with threshold=0.15 (Known answer #5)
    result = variance_filter_tool.invoke({"threshold": 0.15})

    assert result["status"] == "success"
    assert result["total_terms"] == 5
    assert result["n_kept"] == 4
    assert result["n_removed"] == 1
    assert "always" in result["removed_terms"]
    for term in ["alpha", "beta", "delta", "gamma"]:
        assert term in result["kept_terms"]

    # 6. Verify stored full_report matches known answer #5 variances
    stored_result = session.results_store[result["result_id"]]
    full_report = stored_result["full_report"]

    expected_variances = {
        "alpha": 1.1875,
        "always": 0.109375,
        "beta": 0.234375,
        "delta": 0.25,
        "gamma": 1.109375
    }

    for _, row in full_report.iterrows():
        feature = row["feature"]
        var_val = row["variance"]
        assert feature in expected_variances
        assert var_val == pytest.approx(expected_variances[feature], abs=1e-4)
