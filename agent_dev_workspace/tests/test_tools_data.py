import pytest
import pandas as pd
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


def test_check_duplicates_tool_with_unhashable_columns():
    # Fresh session to explicitly test check_duplicates_tool after tokenization
    session = SessionState()
    
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")
    
    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })
    
    tools = make_tools(session)
    tokenize_tool = next(t for t in tools if t.name == "tokenize_tool")
    check_duplicates_tool = next(t for t in tools if t.name == "check_duplicates_tool")
    
    # First tokenize (adds 'unigrams' column containing Python list objects)
    tokenize_tool.invoke({"text_column": "text"})
    assert "unigrams" in session.dataframe.columns
    assert isinstance(session.dataframe["unigrams"].iloc[0], list)

    # Calling check_duplicates_tool with subset=None must safely skip 'unigrams'
    res = check_duplicates_tool.invoke({"drop": False})
    assert res["duplicate_count"] == 1
    assert res["duplicate_indices"] == [5, 6]


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


def test_inspect_data_tool_standalone_and_pipeline():
    session = SessionState()
    
    # 1. Load premade dataset tool
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")
    
    # Load sample fixture
    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    # 2. Instantiate student tools from tools_data.py
    tools = make_tools(session)
    inspect_data_tool = next(t for t in tools if t.name == "inspect_data_tool")
    tokenize_tool = next(t for t in tools if t.name == "tokenize_tool")
    check_duplicates_tool = next(t for t in tools if t.name == "check_duplicates_tool")

    # 3. Test inspect_data_tool on initial state with specific columns ("text" and "category_name")
    initial_res = inspect_data_tool.invoke({"n": 3, "max_text_len": 15, "columns": ["text", "category_name"]})
    assert initial_res["result_id"].startswith("inspect_data_")
    assert initial_res["total_rows"] == 8
    assert initial_res["n_rows"] == 3
    assert initial_res["columns"] == ["text", "category_name"]
    assert len(initial_res["rows"]) == 3
    
    # Exact check of keys in row dicts (only _index, text, category_name)
    assert set(initial_res["rows"][0].keys()) == {"_index", "text", "category_name"}
    assert initial_res["rows"][0]["_index"] == 0
    assert initial_res["rows"][1]["_index"] == 1
    assert initial_res["rows"][2]["_index"] == 2
    
    # Exact check of truncated string content ("always alpha alpha alpha beta" -> "always alpha al...")
    assert initial_res["rows"][0]["text"] == "always alpha al..."
    assert initial_res["rows"][0]["category_name"] == "catA"

    # 4. Run other pipeline tools first: tokenize then drop duplicates
    tokenize_tool.invoke({"text_column": "text"})
    check_duplicates_tool.invoke({"drop": True})

    # After drop_duplicates(keep=False), 2 duplicate rows (indices 5, 6) are dropped, leaving 6 rows
    assert len(session.dataframe) == 6

    # 5. Test inspect_data_tool again after pipeline modifications
    pipeline_res = inspect_data_tool.invoke({"n": 10})
    assert pipeline_res["total_rows"] == 6
    assert pipeline_res["n_rows"] == 6
    assert "unigrams" in pipeline_res["columns"]
    
    # Stricter index check: rows 5 & 6 dropped, so remaining indices are [0, 1, 2, 3, 4, 7]
    actual_indices = [r["_index"] for r in pipeline_res["rows"]]
    assert actual_indices == [0, 1, 2, 3, 4, 7]

    # Check that unigrams in row 0 are present as a list
    assert pipeline_res["rows"][0]["unigrams"] == ["always", "alpha", "alpha", "alpha", "beta"]


def test_list_files_tool_known_answer_11():
    session = SessionState()

    tools = make_tools(session)
    list_files_tool = next(t for t in tools if t.name == "list_files_tool")

    # Execute list_files_tool on subdirectory="newdataset" according to Known Answer #11
    result = list_files_tool.invoke({"subdirectory": "newdataset"})

    assert "result_id" in result
    assert result["subdirectory"] == "newdataset"
    assert result["directories"] == []
    assert result["files"] == ["Reddit-stock-sentiment.csv"]
