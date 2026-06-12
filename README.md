# Media Agent with AI

An agentic system that turns a blog post into a professional **infographic image**, self-evaluates the result, and iterates until it meets quality criteria.

Give it a URL → it reads the blog, generates an infographic, critiques its own output for accuracy, spelling, and aesthetics, and regenerates with feedback until the image passes (or it exhausts its retry budget).

---

## How it works

```
        ┌─────────────────────────────────────────────────────────────┐
        │            Orchestrator Agent  (InfographicAgent)            │
        │              ADK LlmAgent · gemini-3-flash-preview           │
        └───────────────────────────────┬─────────────────────────────┘
                                         │ calls tool
                                         ▼
                          infographic_workflow(url)
                                         │
        ┌────────────────────────────────┼──────────────────────────────┐
        ▼                                ▼                               ▼
  fetch_blog_content          generate_infographic            evaluate_infographic
  (Content Fetcher)        (Image Generation Agent)            (Evaluation Agent)
  requests.get(url)        gemini-3.1-flash-image-preview      gemini-3-flash-preview
                           "Nano Banana"                       (vision / critic)
                                         │                               │
                                         └────────► feedback loop ◄───────┘
                                             (up to 3 attempts)
```

The whole pipeline writes a timestamped trace to `infographic_agent.log`.

---

## Agents & components

| Component | Role | Model / Tech |
|-----------|------|--------------|
| **Orchestrator Agent** (`root_agent` / `InfographicAgent`) | Top-level ADK `LlmAgent`. Receives the user request, reasons about it, and calls the workflow tool. | `gemini-3-flash-preview` (via `VertexGemini`) |
| **Content Fetcher** (`fetch_blog_content`) | Downloads the blog HTML and returns a truncated text slice for prompt efficiency. | `requests` |
| **Image Generation Agent** (`generate_infographic`) | Converts blog content into a professional infographic PNG. Accepts feedback to fix prior issues. | `gemini-3.1-flash-image-preview` ("Nano Banana") |
| **Evaluation Agent** (`evaluate_infographic`) | Vision critic. Scores the image on factual accuracy, spelling, and aesthetic alignment. Returns `PASS` or actionable feedback. | `gemini-3-flash-preview` (vision) |
| **Workflow Controller** (`infographic_workflow`) | The generate → evaluate → refine loop (max 3 attempts). Exposed to the orchestrator as its single tool. | — |
| **Logging** (`log_step`) | Cross-cutting helper that records every step. | writes `infographic_agent.log` |

---

## The feedback loop

1. **Fetch** — pull the blog text from the URL.
2. **Generate** — create an infographic from that text.
3. **Evaluate** — judge the image against the blog content.
   - If the verdict is exactly `PASS` → return success.
   - Otherwise → capture the feedback and go back to step 2.
4. Repeat up to **3 attempts**, then return the last version.

---

## Requirements

- **Python 3.11+**
- A Google Cloud project with **Vertex AI** enabled and access to the Gemini models used above.
- A `helper.py` module exposing `authenticate()` that returns `(credentials, project_id)`.
- Environment variable `GOOGLE_VERTEX_BASE_URL` pointing at your Vertex base URL.

### Python packages

```bash
pip install requests pillow google-genai google-adk
```

> The exact package/version set may vary with your environment. Pin versions in a `requirements.txt` for reproducibility.

---

## Setup

1. **Clone / copy** the project files so you have `Media_Agent_with_AI.py` and your `helper.py` in the same directory.

2. **Authentication.** Ensure `helper.authenticate()` is wired to your credentials, e.g.:

   ```python
   # helper.py
   def authenticate():
       # return (credentials, project_id)
       ...
   ```

3. **Set environment variables.**

   ```bash
   export GOOGLE_VERTEX_BASE_URL="https://your-vertex-base-url"
   ```

---

## Usage

### Run the bundled example

The script ships with a sample blog and an async entry point:

```bash
python Media_Agent_with_AI.py
```

This runs the orchestrator agent against the default blog URL, prints the agent's text/tool-call events, and saves the resulting infographic as `infographic_<timestamp>.png`.

### Use your own blog

Edit the `BLOG` variable inside `main()` (or call the workflow directly):

```python
from Media_Agent_with_AI import infographic_workflow

result = infographic_workflow("https://example.com/your-blog-post")
print(result)
```

### Drive it through the agent

```python
import asyncio
from google.genai import types
from google.adk.runners import InMemoryRunner
from Media_Agent_with_AI import root_agent

async def run():
    runner = InMemoryRunner(agent=root_agent, app_name="image_agent")
    session = await runner.session_service.create_session(
        app_name="image_agent", user_id="user"
    )
    msg = types.Content(
        role="user",
        parts=[types.Part(text="Create an infographic from this blog: https://example.com/post")],
    )
    async for event in runner.run_async(
        user_id="user", session_id=session.id, new_message=msg
    ):
        if event.content and event.content.parts:
            for part in event.content.parts:
                if part.text:
                    print(part.text)

asyncio.run(run())
```

---

## Outputs

| File | Description |
|------|-------------|
| `infographic_<YYYYMMDD_HHMMSS>.png` | The generated infographic image. |
| `infographic_agent.log` | Timestamped, step-by-step trace of fetch / generate / evaluate / retry. |

---

## Configuration & tuning

| Setting | Where | Default |
|---------|-------|---------|
| Max retry attempts | `max_attempts` in `infographic_workflow` | `3` |
| Blog text limit | slice in `fetch_blog_content` | `5000` chars |
| Fetch timeout | `requests.get(..., timeout=...)` | `10` s |
| Image model | `model=` in `generate_infographic` | `gemini-3.1-flash-image-preview` |
| Evaluation model | `model=` in `evaluate_infographic` | `gemini-3-flash-preview` |
| Orchestrator model | `VertexGemini(model=...)` in `root_agent` | `gemini-3-flash-preview` |

---

## Notes & limitations

- **Heuristic extraction.** `fetch_blog_content` returns raw HTML text (truncated). For complex pages, consider adding proper HTML-to-text parsing.
- **Non-deterministic output.** Generative models produce different results on each run; the same input may yield different infographics and evaluations.
- **Pass criterion is strict.** The loop only short-circuits when the evaluator returns exactly `PASS`. Adjust the evaluation prompt if you want looser or stricter behavior.
- **Costs.** Each attempt makes image-generation and vision-evaluation calls; up to 3 attempts can mean several model calls per run.

---

## Project structure

```
.
├── Media_Agent_with_AI.py    # Agents, tools, workflow, and entry point
├── helper.py                 # Provides authenticate() -> (credentials, project_id)
├── infographic_agent.log     # Generated at runtime
└── infographic_*.png         # Generated infographics
```
