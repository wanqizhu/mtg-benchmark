from harness.providers.xai import (
    assistant_content_blocks,
    map_stop_reason,
    messages_from_transcript,
    parse_tool_call_arguments,
    queue_max_token_continue,
    usage_from_xai,
    _tool_defs,
)
from harness.tools import FileTool


def test_usage_normalizes_cached_and_reasoning_tokens():
    usage = usage_from_xai(
        {
            "prompt_tokens": 120,
            "completion_tokens": 80,
            "prompt_tokens_details": {"cached_tokens": 90},
            "completion_tokens_details": {"reasoning_tokens": 55},
        }
    )
    assert usage["input_tokens"] == 30
    assert usage["output_tokens"] == 80
    assert usage["cache_read_input_tokens"] == 90
    assert usage["cache_creation_input_tokens"] == 0
    assert usage["prompt_tokens"] == 120
    assert usage["thinking_tokens"] == 55
    assert usage["completion_tokens"] == 80


def test_usage_adds_reasoning_when_missing_from_completion():
    usage = usage_from_xai(
        {
            "prompt_tokens": 645,
            "completion_tokens": 1,
            "prompt_tokens_details": {"cached_tokens": 512},
            "completion_tokens_details": {"reasoning_tokens": 361},
        }
    )
    assert usage["output_tokens"] == 362
    assert usage["thinking_tokens"] == 361
    assert usage["input_tokens"] == 133
    assert usage["cache_read_input_tokens"] == 512


def test_tool_defs_are_openai_functions():
    tool = FileTool(
        name="grep",
        description="Search",
        input_schema={"type": "object", "properties": {"pattern": {"type": "string"}}},
        _fn=lambda **kwargs: "ok",
    )
    assert _tool_defs([tool]) == [
        {
            "type": "function",
            "function": {
                "name": "grep",
                "description": "Search",
                "parameters": {"type": "object", "properties": {"pattern": {"type": "string"}}},
            },
        }
    ]


def test_stop_reason_mapping():
    assert map_stop_reason("stop") == "end_turn"
    assert map_stop_reason("tool_calls") == "tool_use"
    assert map_stop_reason("length") == "max_tokens"


def test_assistant_blocks_and_resume_messages_keep_reasoning():
    api_message = {
        "role": "assistant",
        "content": "final",
        "reasoning_content": "think",
        "tool_calls": [
            {
                "id": "call_1",
                "type": "function",
                "function": {"name": "grep", "arguments": '{"pattern": "flash"}'},
            }
        ],
    }
    blocks = assistant_content_blocks(api_message)
    assert blocks[0] == {"type": "thinking", "thinking": "think"}
    assert blocks[1] == {"type": "text", "text": "final"}
    assert blocks[2]["name"] == "grep"
    assert parse_tool_call_arguments(api_message["tool_calls"][0]) == {"pattern": "flash"}

    turns = [
        {"role": "system", "text": "sys"},
        {"role": "user", "text": "solve", "api_message": {"role": "user", "content": "solve"}},
        {"role": "assistant", "api_message": api_message},
        {
            "role": "user",
            "content": [{"type": "tool_result", "tool_use_id": "call_1", "content": "hit"}],
            "api_messages": [{"role": "tool", "tool_call_id": "call_1", "content": "hit"}],
        },
        {
            "role": "user",
            "text": "continue",
            "reason": "continue_after_max_tokens",
            "api_message": {"role": "user", "content": "continue"},
        },
    ]
    messages = messages_from_transcript(turns)
    assert messages[0] == {"role": "system", "content": "sys"}
    assert messages[1] == {"role": "user", "content": "solve"}
    assert messages[2]["reasoning_content"] == "think"
    assert messages[3] == {"role": "tool", "tool_call_id": "call_1", "content": "hit"}
    assert messages[4] == {"role": "user", "content": "continue"}


def test_resume_does_not_queue_continue_when_budget_is_exhausted():
    from harness.providers.base import append_max_token_continue

    turns = [{"role": "assistant", "stop_reason": "max_tokens"}]
    assert append_max_token_continue(turns, 5, 5) == 5
    assert len(turns) == 1
    assert append_max_token_continue(turns, 0, 0) == 0
    assert len(turns) == 1


def test_resume_queues_continue_after_saved_max_tokens_turn():
    turns = [
        {"role": "user", "text": "solve"},
        {"role": "assistant", "stop_reason": "max_tokens", "text": None},
    ]
    messages = [{"role": "user", "content": "solve"}, {"role": "assistant", "content": ""}]
    count = queue_max_token_continue(turns, messages, 0, 5)
    assert count == 1
    assert turns[-1]["reason"] == "continue_after_max_tokens"
    assert messages[-1] == {"role": "user", "content": "continue"}
    assert queue_max_token_continue(turns, messages, count, 5) == 1
