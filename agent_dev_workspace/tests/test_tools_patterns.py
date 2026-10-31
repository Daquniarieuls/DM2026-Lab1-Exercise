import pytest
import numpy as np
from agent_pipeline.session_state import SessionState
from agent_pipeline.workspace_loader import load_workspace_tools
from tools_patterns import make_tools


def _setup_session_and_tools():
    session = SessionState()
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")
    
    # Load dataset
    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })
    
    # Load student tools
    student_tools = make_tools(session)
    mine_patterns_tool = next(t for t in student_tools if t.name == "mine_patterns_tool")
    return session, mine_patterns_tool


def test_known_answer_17_fpgrowth():
    """Test known answer #17: mine_patterns_tool with fpgrowth and percentile filtering."""
    # 1. Test variance & term_frequency filtering (keep zero terms)
    session1, mine_patterns_tool1 = _setup_session_and_tools()
    
    res_var = mine_patterns_tool1.invoke({
        "category_name": "catA",
        "filtering_method": "variance",
        "algorithm": "fpgrowth",
        "min_sup": 1
    })
    assert res_var.get("n_patterns", 0) == 0
    assert "no terms survived filtering" in res_var.get("note", "")

    res_tf = mine_patterns_tool1.invoke({
        "category_name": "catB",
        "filtering_method": "term_frequency",
        "algorithm": "fpgrowth",
        "min_sup": 1
    })
    assert res_tf.get("n_patterns", 0) == 0
    assert "no terms survived filtering" in res_tf.get("note", "")

    # 2. Test tfidf filtering with fpgrowth
    session2, mine_patterns_tool2 = _setup_session_and_tools()
    
    # catA tfidf fpgrowth
    res_catA = mine_patterns_tool2.invoke({
        "category_name": "catA",
        "filtering_method": "tfidf",
        "algorithm": "fpgrowth",
        "min_sup": 1
    })
    assert res_catA.get("n_patterns") == 1
    patterns_catA = res_catA.get("patterns", [])
    assert len(patterns_catA) == 1
    assert patterns_catA[0]["pattern"] == ["alpha"]
    assert pytest.approx(patterns_catA[0]["support"], abs=1e-3) == 3.0

    # catB tfidf fpgrowth
    res_catB = mine_patterns_tool2.invoke({
        "category_name": "catB",
        "filtering_method": "tfidf",
        "algorithm": "fpgrowth",
        "min_sup": 1
    })
    assert res_catB.get("n_patterns") == 1
    patterns_catB = res_catB.get("patterns", [])
    assert len(patterns_catB) == 1
    assert patterns_catB[0]["pattern"] == ["gamma"]
    assert pytest.approx(patterns_catB[0]["support"], abs=1e-3) == 4.0


def test_known_answer_20_topk_and_maxfpgrowth():
    """Test known answer #20: mine_patterns_tool with topk and maxfpgrowth algorithms."""
    # 1. Test topk algorithm
    session1, mine_patterns_tool1 = _setup_session_and_tools()
    
    res_topk_catA = mine_patterns_tool1.invoke({
        "category_name": "catA",
        "filtering_method": "tfidf",
        "algorithm": "topk",
        "k": 1
    })
    assert res_topk_catA.get("n_patterns") == 1
    assert res_topk_catA["patterns"][0]["pattern"] == ["alpha"]
    assert pytest.approx(res_topk_catA["patterns"][0]["support"], abs=1e-3) == 3.0

    res_topk_catB = mine_patterns_tool1.invoke({
        "category_name": "catB",
        "filtering_method": "tfidf",
        "algorithm": "topk",
        "k": 1
    })
    assert res_topk_catB.get("n_patterns") == 1
    assert res_topk_catB["patterns"][0]["pattern"] == ["gamma"]
    assert pytest.approx(res_topk_catB["patterns"][0]["support"], abs=1e-3) == 4.0

    # 2. Test maxfpgrowth algorithm (returns 0 patterns on single itemset)
    session2, mine_patterns_tool2 = _setup_session_and_tools()
    
    res_max_catA = mine_patterns_tool2.invoke({
        "category_name": "catA",
        "filtering_method": "tfidf",
        "algorithm": "maxfpgrowth",
        "min_sup": 1
    })
    assert res_max_catA.get("n_patterns", 0) == 0

    res_max_catB = mine_patterns_tool2.invoke({
        "category_name": "catB",
        "filtering_method": "tfidf",
        "algorithm": "maxfpgrowth",
        "min_sup": 1
    })
    assert res_max_catB.get("n_patterns", 0) == 0
