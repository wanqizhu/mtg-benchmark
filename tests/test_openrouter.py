import json

import httpx

from harness.model_names import parse_model_name
from harness.providers.openrouter import OPENROUTER_BASE_URL, OpenRouterProvider
from harness.providers.xai import (
    _merge_reasoning_details,
    _merge_usage,
    assistant_content_blocks,
    usage_from_xai,
)
from harness.tools import FileTool


def _sse(events: list[dict]) -> bytes:
    lines = [f"data: {json.dumps(event)}\n\n" for event in events]
    lines.append("data: [DONE]\n\n")
    return "".join(lines).encode("utf-8")


def test_usage_captures_reported_cost_and_cache_writes():
    usage = usage_from_xai(
        {
            "prompt_tokens": 1000,
            "completion_tokens": 300,
            "prompt_tokens_details": {"cached_tokens": 200, "cache_write_tokens": 100},
            "completion_tokens_details": {"reasoning_tokens": 250},
            "cost": 0.0123,
        }
    )
    assert usage["input_tokens"] == 700
    assert usage["cache_read_input_tokens"] == 200
    assert usage["cache_creation_input_tokens"] == 100
    assert usage["reported_cost_usd"] == 0.0123

    total: dict = {}
    _merge_usage(total, usage)
    _merge_usage(total, usage)
    assert total["reported_cost_usd"] == 0.0246
    assert total["input_tokens"] == 1400


def test_assistant_blocks_read_openrouter_reasoning_field():
    blocks = assistant_content_blocks({"role": "assistant", "content": "x", "reasoning": "why"})
    assert blocks[0] == {"type": "thinking", "thinking": "why"}


def test_merge_reasoning_details_coalesces_streamed_chunks():
    merged = _merge_reasoning_details(
        [
            {"type": "reasoning.text", "id": "r1", "index": 0, "text": "ab"},
            {"type": "reasoning.text", "id": "r1", "index": 0, "text": "cd", "signature": "sig"},
            {"type": "reasoning.encrypted", "id": "r2", "index": 1, "data": "zz"},
        ]
    )
    assert merged == [
        {"type": "reasoning.text", "id": "r1", "index": 0, "text": "abcd", "signature": "sig"},
        {"type": "reasoning.encrypted", "id": "r2", "index": 1, "data": "zz"},
    ]


def test_raw_openrouter_slug_without_pin_uses_default_routing():
    provider = OpenRouterProvider(api_key="test-key")
    spec = parse_model_name("openrouter/some-lab/new-model-thinking-low")
    body = provider._request_body(spec=spec, messages=[], tools=[], effort="low")
    assert "provider" not in body
    assert body["reasoning"] == {"effort": "low"}


def test_openrouter_rollout_streams_tools_and_echoes_reasoning_details():
    requests: list[dict] = []
    turn_events = [
        # Turn 1: reasoning + a tool call.
        [
            {
                "provider": "Z.ai",
                "choices": [
                    {
                        "delta": {
                            "reasoning": "look it up",
                            "reasoning_details": [{"type": "reasoning.text", "id": "r1", "text": "look "}],
                        }
                    }
                ]
            },
            {
                "choices": [
                    {
                        "delta": {
                            "reasoning_details": [{"type": "reasoning.text", "id": "r1", "text": "it up"}],
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "id": "call_1",
                                    "function": {"name": "grep", "arguments": '{"pattern": "702"}'},
                                }
                            ],
                        },
                        "finish_reason": "tool_calls",
                    }
                ]
            },
            {"choices": [], "usage": {"prompt_tokens": 10, "completion_tokens": 5, "cost": 0.001}},
        ],
        # Turn 2: final text.
        [
            {"choices": [{"delta": {"content": "Answer"}, "finish_reason": "stop"}]},
            {"choices": [], "usage": {"prompt_tokens": 20, "completion_tokens": 2, "cost": 0.002}},
        ],
    ]

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        requests.append({"headers": dict(request.headers), "body": body})
        events = turn_events[len(requests) - 1]
        return httpx.Response(200, content=_sse(events), headers={"content-type": "text/event-stream"})

    provider = OpenRouterProvider(api_key="test-key")
    provider._client = httpx.Client(
        base_url=OPENROUTER_BASE_URL,
        transport=httpx.MockTransport(handler),
        headers={"Authorization": "Bearer test-key"},
    )
    tool = FileTool(
        name="grep",
        description="Search",
        input_schema={"type": "object", "properties": {"pattern": {"type": "string"}}},
        _fn=lambda **kwargs: f"hit:{kwargs['pattern']}",
    )
    spec = parse_model_name("glm-5.3-thinking-high")
    transcript = provider.run_rollout(
        spec=spec, system="sys", prompt="solve", tools=[tool], max_turns=5
    )

    first = requests[0]["body"]
    assert first["model"] == "z-ai/glm-5.3"
    assert first["reasoning"] == {"effort": "high"}
    assert first["usage"] == {"include": True}
    assert first["provider"] == {"order": ["z-ai/fp8"], "allow_fallbacks": False}
    assert "reasoning_effort" not in first
    assert first["tools"][0]["function"]["name"] == "grep"

    second = requests[1]["body"]["messages"]
    assistant = second[2]
    assert assistant["reasoning"] == "look it up"
    assert assistant["reasoning_details"] == [{"type": "reasoning.text", "id": "r1", "text": "look it up"}]
    assert assistant["tool_calls"][0]["id"] == "call_1"
    assert second[3] == {"role": "tool", "tool_call_id": "call_1", "content": "hit:702"}

    assert transcript.provider == "openrouter"
    assert transcript.tool_call_count == 1
    assert transcript.usage["reported_cost_usd"] == 0.003
    config = next(t for t in transcript.turns if t["role"] == "config")
    assert config["provider_pin"] == ["z-ai/fp8"]
    assistant_turns = [t for t in transcript.turns if t["role"] == "assistant"]
    assert assistant_turns[0]["served_by"] == "Z.ai"
    assert "served_by" not in assistant_turns[1]
    assert assistant_turns[0]["stop_reason"] == "tool_use"
    assert assistant_turns[0]["content"][0] == {"type": "thinking", "thinking": "look it up"}
    assert assistant_turns[1]["text"] == "Answer"
    assert assistant_turns[1]["stop_reason"] == "end_turn"
