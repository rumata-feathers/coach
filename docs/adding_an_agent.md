# Adding a new agent

This guide walks through adding a new agent (e.g., `Researcher`) for v1.
Follow these steps in order — they match how every existing agent was built.

---

## Step 1: Define the I/O contract

Add typed Pydantic models to `src/career_coach/models/agent_io.py`:

```python
class ResearcherInput(BaseModel):
    topic: str = Field(..., min_length=1)
    user_facts: dict[str, Any] = Field(default_factory=dict)
    turn_id: UUID

class ResearcherOutput(BaseModel):
    summary: str
    sources: list[str] = Field(default_factory=list)
    confidence: float = Field(..., ge=0.0, le=1.0)
```

Rules:
- Every field must have a type annotation.
- Forbidden: `dict[str, Any]` as a top-level output — wrap it in a named model.
- Forbidden: raw strings at the boundary — use literal types or enums.

---

## Step 2: Write the prompt template

Create `config/prompts/researcher.j2`:

```jinja2
{# researcher.j2 — Researcher agent system prompt
   Receives: topic, user_facts
   Produces: ResearcherOutput JSON
#}
You are the Researcher agent in a career coaching system.
...

## Output format
Respond with JSON only:
{
  "summary": "...",
  "sources": [...],
  "confidence": 0.0-1.0
}
```

Rules:
- First line of the template MUST be a comment block explaining its purpose.
- All variables must be either validated by Jinja2's `StrictUndefined` or have
  default guards (`{% if x %}...{% endif %}`).
- End with an explicit output-format section so the model can't drift.

---

## Step 3: Register the agent in models.yaml

```yaml
# config/models.yaml
researcher:
  provider: huggingface
  model: Qwen/Qwen3-235B-A22B
  temperature: 0.3
  max_tokens: 3000
```

The key (`researcher`) is the agent name passed to `super().__init__()`.

---

## Step 4: Implement the agent class

Create `src/career_coach/agents/researcher.py`:

```python
from career_coach.agents.base import Agent
from career_coach.llm.client import Message
from career_coach.llm.factory import LLMFactory
from career_coach.models.agent_io import ResearcherInput, ResearcherOutput

class Researcher(Agent):
    def __init__(self, factory: LLMFactory) -> None:
        super().__init__("researcher", factory)  # matches models.yaml key

    async def run(
        self,
        input_data: ResearcherInput,
        *,
        turn_id: UUID | None = None,
    ) -> ResearcherOutput:
        prompt = self.render_prompt(
            "researcher.j2",
            topic=input_data.topic,
            user_facts=input_data.user_facts,
        )
        messages = [Message(role="user", content=prompt)]
        t0 = self.now_ms()
        error_str: str | None = None
        response = await self.complete(messages, response_format="json")
        latency = self.now_ms() - t0

        try:
            output = _parse_output(response.text)
        except (json.JSONDecodeError, ValidationError, ...) as exc:
            error_str = str(exc)
            output = _fallback_output(str(exc))

        await self.log_call(
            turn_id=turn_id,
            input_payload={"topic": input_data.topic},
            output_payload=output.model_dump(),
            latency_ms=latency,
            tokens_in=response.tokens_in,
            tokens_out=response.tokens_out,
            error=error_str,
        )
        return output
```

Rules:
- **Always call `self.log_call()`** — even on error. Pass `error=str(exc)`.
- **Always return a safe fallback** — never let a parse error reach the caller.
- **Use `self.render_prompt()`** — never inline prompt text.
- **Use `self.complete()`** — never import an LLM SDK directly.

---

## Step 5: Write unit tests

Create `tests/agents/test_researcher.py`:

```python
from pathlib import Path
from career_coach.agents.researcher import Researcher
from career_coach.llm.factory import LLMFactory
from career_coach.llm.mock import MockLLMClient

_CONFIG = Path(__file__).resolve().parents[2] / "config" / "models.yaml"

def _make_factory(mock):
    factory = LLMFactory(config_path=_CONFIG)
    factory.register_client("huggingface", mock)
    return factory

async def test_researcher_returns_valid_output() -> None:
    mock = MockLLMClient(default_response=_good_json())
    researcher = Researcher(_make_factory(mock))
    output = await researcher.run(_make_input())
    assert isinstance(output, ResearcherOutput)
    assert output.summary

async def test_researcher_falls_back_on_malformed_json() -> None:
    mock = MockLLMClient(default_response="not json")
    researcher = Researcher(_make_factory(mock))
    output = await researcher.run(_make_input())
    assert isinstance(output, ResearcherOutput)  # must not crash
```

Minimum test coverage:
- Valid JSON → valid output
- Malformed JSON → safe fallback (must not raise)
- Correct model name used
- JSON format requested
- Prompt contains key context variables

---

## Step 6: Wire into the pipeline (if needed)

If the agent is on the critical path, add it to
`src/career_coach/pipeline/turn.py`. If it's a background task, follow the
Profiler pattern: call `asyncio.create_task(agent.run_and_save(...))`.

---

## Checklist

- [ ] `models/agent_io.py` — typed input + output models
- [ ] `config/prompts/researcher.j2` — Jinja2 template with output schema
- [ ] `config/models.yaml` — agent entry with provider + model
- [ ] `agents/researcher.py` — implements `Agent`, calls `log_call()`
- [ ] `tests/agents/test_researcher.py` — ≥ 5 unit tests
- [ ] `pipeline/turn.py` or background task — wired in
