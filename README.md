# Tech Intelligence Agent

A single-agent [LangGraph](https://github.com/langchain-ai/langgraph) pipeline that answers multi-step questions about the software ecosystem by cross-referencing **GitHub** (what the community builds) with **HackerNews** (what the community thinks). Built on a locally-hosted `qwen2.5:7b` model via Ollama.

> Built for COMP2701 Assessment 2 — a domain intelligence agent exercising tool design, stateful LangGraph orchestration, guardrails, and LLM-as-judge evaluation.

## What it does

Given a query like *"What are the most popular open source Python frameworks for building LLM agents?"*, the agent:

1. Sanitises the input against empty, overlong, injected, and nonsensical queries
2. Extracts short keywords for the GitHub search API and passes the natural-language query to HackerNews
3. Calls both APIs and stores the raw results in shared graph state
4. Synthesises 4–5 cross-referenced insights from both sources
5. Self-checks that the synthesis is actually grounded in the tool results (retrying up to 3 times if not)
6. Formats a structured, 150–300 word final answer

## Architecture

```
START → fetch → [guardrail / route] → analyse → check → respond → END
          ↑                              ↑         │
          └── retry (up to 3x) ──────────┘  loop back if ungrounded
```

| Node | Responsibility |
|---|---|
| `fetch` | Runs the guardrail first, extracts GitHub keywords, calls both tools, decides whether to proceed, retry, or bail out |
| `analyse` | Filters raw API results down to relevant fields and asks the LLM to extract cross-referenced insights |
| `check` | Verifies the analysis is actually grounded in the tool results; routes back to `analyse` if not (capped at 3 attempts) |
| `respond` | Formats the grounded analysis into a final 150–300 word answer, or a graceful fallback if nothing usable was found |

Compiled with `MemorySaver` for per-thread checkpointing.

## Tools

| Tool | API | Returns |
|---|---|---|
| `search_github` | `api.github.com/search/repositories` | Repos sorted by stars — name, description, stars, language, URL |
| `search_hackernews` | `hn.algolia.com/api/v1/search` | Stories — title, URL, points, comments, date |

## Guardrail

Sits at the top of `fetch_node`, before any API call. It:

- Rejects empty or whitespace-only queries
- Blocks a defined set of prompt-injection patterns (e.g. *"ignore previous instructions"*)
- Truncates queries over 200 characters
- Falls back to an LLM sanity check for nonsensical or subtly malicious input the pattern list misses

**Known limitation:** paraphrased injection attempts can bypass both the pattern list and the LLM check.

## Evaluation

A 10-case golden dataset (4 normal, 3 edge, 3 adversarial) scored with an LLM-as-judge against per-category rubrics covering query relevance, use of source data, response length, and evidence citation (adversarial cases are scored instead on crash-safety, injection resistance, and graceful failure).

**Result: 85% overall** across all categories.

**Key finding:** the judge's `overall_score` was model-generated rather than computed from its own per-criterion boolean outputs, which flattened the results — a case with a strong summary and a case with a weak one both scored 85%, making the ranking unreliable for spotting real failure modes.

**Proposed fix:** derive `overall_score` deterministically from the judge's own boolean criterion checks instead of trusting a separately-generated number, at the cost of shifting reliability onto the criterion checks themselves.

## Stack

- [LangGraph](https://github.com/langchain-ai/langgraph) — stateful agent orchestration
- [LangChain](https://github.com/langchain-ai/langchain) (`langchain-ollama`, `langchain-core`) — LLM + tool interfaces
- [Ollama](https://ollama.com/) running `qwen2.5:7b` — local inference
- `httpx`, `pydantic` — API calls and schema validation

## Running it

Built and tested in Google Colab on a T4 GPU. The notebook installs Ollama, pulls `qwen2.5:7b`, and installs Python dependencies in the setup cells — run those once per session, then execute the remaining cells top to bottom.

```
1. Runtime → Change runtime type → T4 GPU
2. Run the setup cells (Ollama install, model pull, pip install)
3. Run the tool library, state schema, node, and graph cells
4. Run the eval suite / self-check assertions
```

## Notes

This was built as a learning exercise to demonstrate tool schema design, stateful graph routing with retries and short-circuits, a functioning guardrail, and a working (if imperfect) LLM-as-judge eval loop — not as a polished product. See the eval results above for an honest look at where the evaluation methodology itself falls short.
