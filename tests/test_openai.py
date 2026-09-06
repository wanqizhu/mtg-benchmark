from harness.providers.openai import (
    _resume_input,
    _tool_defs,
    map_stop_reason,
    response_content_blocks,
    usage_from_openai,
)
from harness.tools import FileTool


def test_usage_normalizes_cached_and_reasoning_without_double_counting():
    usage = usage_from_openai(
        {
            "input_tokens": 120,
            "input_tokens_details": {
                "cached_tokens": 90,
                "cache_write_tokens": 20,
            },
            "output_tokens": 80,
            "output_tokens_details": {"reasoning_tokens": 55},
            "total_tokens": 200,
        }
    )
    assert usage["input_tokens"] == 10
    assert usage["cache_read_input_tokens"] == 90
    assert usage["cache_creation_input_tokens"] == 20
    assert usage["cache_write_input_tokens"] == 20
    assert usage["output_tokens"] == 80
    assert usage["thinking_tokens"] == 55
    assert usage["completion_tokens"] == 80


def test_response_items_become_thinking_text_and_tool_blocks():
    response = {
        "status": "completed",
        "output": [
            {
                "id": "rs_1",
                "type": "reasoning",
                "summary": [
                    {"type": "summary_text", "text": "Check the relevant rule."},
                    {"type": "summary_text", "text": "Then calculate lethal."},
                ],
            },
            {
                "id": "msg_1",
                "type": "message",
                "content": [{"type": "output_text", "text": "Final answer."}],
            },
            {
                "id": "fc_1",
                "type": "function_call",
                "call_id": "call_1",
                "name": "grep",
                "arguments": '{"pattern":"702.35","path":"/rules"}',
            },
        ],
    }
    blocks = response_content_blocks(response)
    assert blocks[0] == {
        "type": "thinking",
        "thinking": "Check the relevant rule.\n\nThen calculate lethal.",
    }
    assert blocks[1] == {"type": "text", "text": "Final answer."}
    assert blocks[2] == {
        "type": "tool_use",
        "id": "call_1",
        "name": "grep",
        "input": {"pattern": "702.35", "path": "/rules"},
        "response_item_id": "fc_1",
    }
    assert map_stop_reason(response, blocks) == "tool_use"


def test_incomplete_response_maps_to_max_tokens():
    response = {
        "status": "incomplete",
        "incomplete_details": {"reason": "max_output_tokens"},
        "output": [],
    }
    assert map_stop_reason(response, []) == "max_tokens"


def test_openai_tools_use_responses_api_shape():
    tool = FileTool(
        name="grep",
        description="Search",
        input_schema={
            "type": "object",
            "properties": {"pattern": {"type": "string"}},
            "required": ["pattern"],
        },
        _fn=lambda **kwargs: "ok",
    )
    assert _tool_defs([tool]) == [
        {
            "type": "function",
            "name": "grep",
            "description": "Search",
            "parameters": {
                "type": "object",
                "properties": {"pattern": {"type": "string"}},
                "required": ["pattern"],
                "additionalProperties": False,
            },
            "strict": True,
        }
    ]


def test_resume_uses_response_id_and_pending_function_outputs():
    turns = [
        {"role": "user", "api_input": [{"role": "user", "content": "solve"}]},
        {"role": "assistant", "response_id": "resp_1"},
        {
            "role": "user",
            "api_input": [
                {
                    "type": "function_call_output",
                    "call_id": "call_1",
                    "output": "rule text",
                }
            ],
        },
    ]
    response_id, api_input = _resume_input(turns)
    assert response_id == "resp_1"
    assert api_input[0]["type"] == "function_call_output"
