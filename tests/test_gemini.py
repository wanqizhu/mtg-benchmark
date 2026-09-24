from google.genai import types

from harness.providers.gemini import (
    _tool_defs,
    content_blocks,
    content_from_dict,
    content_to_dict,
    contents_from_transcript,
    map_stop_reason,
    merge_stream_parts,
    usage_from_gemini,
)
from harness.tools import FileTool


def test_usage_counts_thoughts_as_output_and_cached_as_read():
    usage = usage_from_gemini(
        {
            "prompt_token_count": 1000,
            "cached_content_token_count": 400,
            "candidates_token_count": 50,
            "thoughts_token_count": 700,
            "tool_use_prompt_token_count": 20,
            "total_token_count": 1770,
        }
    )
    assert usage["input_tokens"] == 620
    assert usage["cache_read_input_tokens"] == 400
    assert usage["output_tokens"] == 750
    assert usage["thinking_tokens"] == 700
    assert usage["prompt_tokens"] == 1020
    assert usage["completion_tokens"] == 50


def test_tool_defs_use_json_schema_declarations():
    tool = FileTool(
        name="grep",
        description="Search",
        input_schema={"type": "object", "properties": {"pattern": {"type": "string"}}, "required": ["pattern"]},
        _fn=lambda **kwargs: "ok",
    )
    [gemini_tool] = _tool_defs([tool])
    [decl] = gemini_tool.function_declarations
    assert decl.name == "grep"
    assert decl.parameters_json_schema == tool.input_schema


def test_content_roundtrip_preserves_thought_signature():
    content = types.Content(
        role="model",
        parts=[
            types.Part(text="plan", thought=True),
            types.Part(
                function_call=types.FunctionCall(id="fc1", name="grep", args={"pattern": "flash"}),
                thought_signature=b"\x00\x01sig",
            ),
        ],
    )
    payload = content_to_dict(content)
    restored = content_from_dict(payload)
    assert restored.parts[1].thought_signature == b"\x00\x01sig"
    assert restored.parts[1].function_call.args == {"pattern": "flash"}

    blocks = content_blocks(restored)
    assert blocks[0] == {"type": "thinking", "thinking": "plan"}
    assert blocks[1] == {"type": "tool_use", "id": "fc1", "name": "grep", "input": {"pattern": "flash"}}
    assert map_stop_reason(types.FinishReason.STOP, blocks) == "tool_use"
    assert map_stop_reason(types.FinishReason.STOP, []) == "end_turn"
    assert map_stop_reason(types.FinishReason.MAX_TOKENS, []) == "max_tokens"


def test_merge_stream_parts_joins_text_and_keeps_signatures():
    parts = [
        types.Part(text="I ", thought=True),
        types.Part(text="think.", thought=True),
        types.Part(text="Ans", thought=False),
        types.Part(text="wer", thought=False),
        types.Part(text="", thought_signature=b"sig-a"),
        types.Part(text="more", thought_signature=b"sig-b"),
        types.Part(function_call=types.FunctionCall(name="read", args={"line_number": 1})),
    ]
    merged = merge_stream_parts(parts)
    assert [p.text for p in merged[:3]] == ["I think.", "Answer", "more"]
    assert merged[0].thought is True
    assert merged[1].thought_signature == b"sig-a"
    assert merged[2].thought_signature == b"sig-b"
    assert merged[3].function_call.name == "read"


def test_contents_from_transcript_replays_api_content():
    user = types.Content(role="user", parts=[types.Part(text="solve")])
    model = types.Content(
        role="model",
        parts=[types.Part(function_call=types.FunctionCall(name="grep", args={}), thought_signature=b"s")],
    )
    reply = types.Content(
        role="user",
        parts=[types.Part(function_response=types.FunctionResponse(name="grep", response={"result": "hit"}))],
    )
    turns = [
        {"role": "system", "text": "sys"},
        {"role": "user", "text": "solve", "api_content": content_to_dict(user)},
        {"role": "assistant", "api_content": content_to_dict(model)},
        {"role": "user", "content": [], "api_content": content_to_dict(reply)},
    ]
    contents = contents_from_transcript(turns)
    assert [c.role for c in contents] == ["user", "model", "user"]
    assert contents[1].parts[0].thought_signature == b"s"
    assert contents[2].parts[0].function_response.response == {"result": "hit"}
