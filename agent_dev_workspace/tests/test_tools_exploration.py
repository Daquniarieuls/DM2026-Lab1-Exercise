import pytest
import numpy as np
from agent_pipeline.session_state import SessionState
from agent_pipeline.workspace_loader import load_workspace_tools
from tools_dtm import make_tools as make_dtm_tools
from tools_exploration import make_tools as make_exploration_tools

def test_cosine_similarity_tool():
    # Single session instance shared across setup and tool invocations
    session = SessionState()

    # Load dataset using premade load_dataset_tool
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")

    load_result = load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })
    assert load_result["n_documents"] == 8

    # Build DTM to store count_vectorizer in session.artifacts
    dtm_tools = make_dtm_tools(session)
    build_dtm_tool = next(t for t in dtm_tools if t.name == "build_dtm_tool")
    dtm_result = build_dtm_tool.invoke({})

    # Load exploration tools
    exploration_tools = make_exploration_tools(session)
    cosine_sim_tool = next(t for t in exploration_tools if t.name == "cosine_similarity_tool")

    # Known answer 16 checks:
    # 0 vs 1 (both catA, near-identical text): 0.9847
    res_0_1 = cosine_sim_tool.invoke({"doc_index_1": 0, "doc_index_2": 1})
    assert res_0_1["similarity_score"] == pytest.approx(0.9847, abs=1e-3)

    # 0 vs 3 (catA vs catB, share only "always"): 0.0909
    res_0_3 = cosine_sim_tool.invoke({"doc_index_1": 0, "doc_index_2": 3})
    assert res_0_3["similarity_score"] == pytest.approx(0.0909, abs=1e-3)

    # 3 vs 4 (both catB, near-identical text): 0.9847
    res_3_4 = cosine_sim_tool.invoke({"doc_index_1": 3, "doc_index_2": 4})
    assert res_3_4["similarity_score"] == pytest.approx(0.9847, abs=1e-3)

    # 0 vs 7 (catA vs the empty-text row): 0.0
    res_0_7 = cosine_sim_tool.invoke({"doc_index_1": 0, "doc_index_2": 7})
    assert res_0_7["similarity_score"] == pytest.approx(0.0, abs=1e-3)


def test_cosine_similarity_tool_missing_vectorizer():
    # Separate fresh session
    session = SessionState()

    # Load dataset so session.dataframe exists, but do NOT run build_dtm_tool
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")
    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    exploration_tools = make_exploration_tools(session)
    cosine_sim_tool = next(t for t in exploration_tools if t.name == "cosine_similarity_tool")

    res = cosine_sim_tool.invoke({"doc_index_1": 0, "doc_index_2": 1})
    assert "error" in res
    assert "count_vectorizer" in res["error"]


def test_feature_correlation_matrix_tool():
    # Fresh session for feature_correlation_matrix_tool
    session = SessionState()

    # Load dataset
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")
    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    # Build DTM
    dtm_tools = make_dtm_tools(session)
    build_dtm_tool = next(t for t in dtm_tools if t.name == "build_dtm_tool")
    build_dtm_tool.invoke({})

    # Run feature_correlation_matrix_tool
    exploration_tools = make_exploration_tools(session)
    feature_corr_tool = next(t for t in exploration_tools if t.name == "feature_correlation_matrix_tool")
    res = feature_corr_tool.invoke({})

    # Check terms order by variance descending (known answer 19):
    # alpha, gamma, delta, beta, always
    expected_terms = ["alpha", "gamma", "delta", "beta", "always"]
    assert res["terms"] == expected_terms
    assert res["n_terms"] == 5

    # Known answer 19 correlation matrix:
    #             alpha    gamma    delta     beta   always
    # alpha      1.0000  -0.5718  -0.6882   0.8885   0.2601
    # gamma     -0.5718   1.0000   0.8307  -0.6435   0.3140
    # delta     -0.6882   0.8307   1.0000  -0.7746   0.3780
    # beta       0.8885  -0.6435  -0.7746   1.0000   0.2928
    # always     0.2601   0.3140   0.3780   0.2928   1.0000
    expected_matrix = [
        [ 1.0000, -0.5718, -0.6882,  0.8885,  0.2601],
        [-0.5718,  1.0000,  0.8307, -0.6435,  0.3140],
        [-0.6882,  0.8307,  1.0000, -0.7746,  0.3780],
        [ 0.8885, -0.6435, -0.7746,  1.0000,  0.2928],
        [ 0.2601,  0.3140,  0.3780,  0.2928,  1.0000]
    ]

    actual_matrix = np.array(res["correlation_matrix"])
    assert actual_matrix.shape == (5, 5)
    assert np.allclose(actual_matrix, expected_matrix, atol=1e-3)

    # Check plotting figure set on session
    assert session.pending_figure is not None


def test_feature_correlation_matrix_tool_missing_dtm():
    # Fresh session without DTM built
    session = SessionState()

    exploration_tools = make_exploration_tools(session)
    feature_corr_tool = next(t for t in exploration_tools if t.name == "feature_correlation_matrix_tool")

    res = feature_corr_tool.invoke({})
    assert "error" in res
    assert "build_dtm_tool" in res["error"]
