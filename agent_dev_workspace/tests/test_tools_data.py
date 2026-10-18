import pytest
from agent_pipeline.session_state import SessionState
from agent_pipeline.workspace_loader import load_workspace_tools
from tools_data import make_tools


def test_check_missing_tool_known_answer_2():
    # Single session reused for load_dataset_tool and check_missing_tool
    session = SessionState()
    
    # Load premade load_dataset_tool
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")
    
    # Load sample fixture CSV into session
    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })
    
    # Load student-built check_missing_tool
    tools = make_tools(session)
    check_missing_tool = next(t for t in tools if t.name == "check_missing_tool")
    
    # Execute check_missing_tool
    result = check_missing_tool.invoke({"text_column": "text"})
    
    # Assert against Known Answer #2: exactly 1 missing entry (out of 8 rows)
    assert result["total_rows"] == 8
    assert result["missing_count"] == 1
    assert "result_id" in result


def test_check_duplicates_tool_known_answer_3():
    # Fresh session for duplicate checking without dropping
    session = SessionState()
    
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")
    
    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })
    
    tools = make_tools(session)
    check_duplicates_tool = next(t for t in tools if t.name == "check_duplicates_tool")
    
    # Test duplicate check without dropping
    result_check = check_duplicates_tool.invoke({"drop": False})
    
    assert "result_id" in result_check
    assert result_check["total_rows_before"] == 8
    assert result_check["duplicate_count"] == 1
    assert result_check["duplicate_indices"] == [5, 6]
    assert result_check["dropped"] is False
    assert result_check["total_rows_after"] == 8

    # Fresh session for testing duplicate dropping (keep=False)
    session_drop = SessionState()
    premade_tools_drop, _ = load_workspace_tools("premade_tools", session_drop)
    load_dataset_tool_drop = next(t for t in premade_tools_drop if t.name == "load_dataset_tool")
    
    load_dataset_tool_drop.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })
    
    tools_drop = make_tools(session_drop)
    check_duplicates_tool_drop = next(t for t in tools_drop if t.name == "check_duplicates_tool")
    
    # Test duplicate check WITH dropping
    result_drop = check_duplicates_tool_drop.invoke({"drop": True})
    
    assert "result_id" in result_drop
    assert result_drop["total_rows_before"] == 8
    assert result_drop["duplicate_count"] == 1
    assert result_drop["dropped"] is True
    assert result_drop["removed_indices"] == [5, 6]
    assert result_drop["total_rows_after"] == 6  # keep=False drops both copies


def test_tokenize_tool_known_answer_10():
    session = SessionState()

    # Load premade load_dataset_tool
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")

    # Load dataset into session
    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    # Instantiate student's tools_data tools
    tools = make_tools(session)
    tokenize_tool = next(t for t in tools if t.name == "tokenize_tool")

    # Run tokenize_tool
    result = tokenize_tool.invoke({"text_column": "text"})

    assert "error" not in result, f"tokenize_tool failed with error: {result.get('error')}"
    assert result["total_documents"] == 8

    # Check unigrams stored in dataframe against Known Answer 10 from TEST_FIXTURE.md
    expected_tokens = [
        ["always", "alpha", "alpha", "alpha", "beta"],
        ["always", "alpha", "alpha", "beta"],
        ["always", "alpha", "beta"],
        ["always", "gamma", "gamma", "gamma", "delta"],
        ["always", "gamma", "gamma", "delta"],
        ["always", "gamma", "delta"],
        ["always", "gamma", "delta"],
        [],
    ]

    actual_unigrams = session.dataframe["unigrams"].tolist()
    assert actual_unigrams == expected_tokens
