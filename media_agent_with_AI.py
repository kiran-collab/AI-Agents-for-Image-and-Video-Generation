"""
Media Agent Build with AI

An agentic system that turns a blog post into a professional infographic image,
self-evaluates the result, and iterates until it meets quality criteria.

----------------------------------------------------------------------------
Agents / Components
----------------------------------------------------------------------------
Orchestrator Agent (root_agent, an ADK LlmAgent named "InfographicAgent")
    The top-level agent. Driven by a Gemini model (gemini-3-flash-preview via
    the VertexGemini wrapper). It receives the user's request, reasons about
    it, and calls the `infographic_workflow` tool to do the work. Instructed to
    act as an infographic expert and to log its progress.

Content Fetcher (fetch_blog_content)
    Retrieval component. Downloads the blog post HTML from a URL and returns a
    truncated slice of the text to keep downstream prompts efficient.

Image Generation Agent (generate_infographic)
    Creative component backed by the Gemini image model
    (gemini-3.1-flash-image-preview, "Nano Banana"). Converts the blog content
    into a professional infographic PNG. Accepts optional feedback so it can
    correct issues flagged on a previous attempt.

Evaluation Agent (evaluate_infographic)
    Critic / judge component backed by a Gemini vision model
    (gemini-3-flash-preview). Inspects the generated image against the blog
    content for (1) factual accuracy, (2) spelling, and (3) aesthetic
    alignment. Returns 'PASS' when satisfied, otherwise actionable feedback.

Workflow Controller (infographic_workflow)
    The generate -> evaluate -> refine loop. Coordinates the generation and
    evaluation agents, feeding evaluation feedback back into generation for up
    to 3 attempts. Exposed to the orchestrator agent as its single tool.

Logging (log_step)
    Cross-cutting helper that records every step to `infographic_agent.log`.

----------------------------------------------------------------------------
End-to-end flow
----------------------------------------------------------------------------
    1. Fetch the text content of a blog post from a URL.
    2. Generate an infographic image from that content (image generation agent).
    3. Evaluate the image for factual accuracy, spelling, and aesthetics
       (evaluation agent).
    4. If it fails evaluation, regenerate with feedback (up to 3 attempts).

Note: Requires `helper.authenticate()` and the relevant environment variables
(e.g. GOOGLE_VERTEX_BASE_URL) to be available at runtime.
"""

import os
import datetime
import requests

from google import genai
from google.genai import types
from google.adk.agents.llm_agent import LlmAgent
from google.adk.models import Gemini
from google.adk.runners import InMemoryRunner
import PIL.Image


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
LOG_FILE = "infographic_agent.log"


def log_step(message: str):
    """Append a timestamped message to the agent log file."""
    timestamp = datetime.datetime.now().isoformat()
    with open(LOG_FILE, "a") as f:
        f.write(f"[{timestamp}] {message}\n")


# ---------------------------------------------------------------------------
# Authentication / client setup
# ---------------------------------------------------------------------------
from helper import authenticate

credentials, project_id = authenticate()

# Vertex AI client used for content/image generation and evaluation.
client = genai.Client(
    project=project_id,
    location="global",
    credentials=credentials,
    http_options=types.HttpOptions(
        base_url=os.getenv("GOOGLE_VERTEX_BASE_URL")
    ),
)


# ---------------------------------------------------------------------------
# Tool functions
# ---------------------------------------------------------------------------
def fetch_blog_content(url: str) -> str:
    """Fetch the text content of a blog post.

    Returns up to the first 5000 characters of the raw page text to keep the
    downstream prompt efficient, or an error string on failure.
    """
    log_step(f"Fetching blog content from: {url}")
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        # Simple text extraction (heuristic)
        content = response.text
        log_step("Successfully fetched blog content.")
        return content[:5000]  # Limit content for prompt efficiency
    except Exception as e:
        log_step(f"Error fetching blog content: {e}")
        return f"Error: {e}"


def generate_infographic(blog_content: str, feedback: str = "") -> str:
    """Generate an infographic image using the Gemini image model (Nano Banana).

    If `feedback` from a previous evaluation is provided, it is appended to the
    prompt so the model can fix prior issues. Saves the PNG to disk and returns
    the filename, or an error string on failure.
    """
    log_step("Generating infographic...")
    prompt = f"Create a professional infographic based on this blog content: {blog_content}."
    if feedback:
        log_step(f"Applying feedback for regeneration: {feedback}")
        prompt += f" Please fix the following issues from the previous attempt: {feedback}"

    try:
        response = client.models.generate_content(
            model="gemini-3.1-flash-image-preview",
            contents=prompt,
            config=types.GenerateContentConfig(
                response_modalities=["IMAGE"],
            ),
        )

        # Pull the first inline image part out of the response and save it.
        image_part = next(p for p in response.candidates[0].content.parts if p.inline_data)
        image_filename = f"infographic_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        with open(image_filename, "wb") as f:
            f.write(image_part.inline_data.data)

        log_step(f"Infographic saved as {image_filename}")
        return image_filename
    except Exception as e:
        log_step(f"Error generating infographic: {e}")
        return f"Error: {e}"


def evaluate_infographic(image_path: str, blog_content: str) -> str:
    """Evaluate an infographic for factual accuracy, spelling, and aesthetics.

    Returns the literal string 'PASS' if the image meets all criteria,
    otherwise returns specific feedback for improvement (or an error string).
    """
    log_step(f"Evaluating infographic: {image_path}")
    try:
        img = PIL.Image.open(image_path)

        evaluation_prompt = (
            "Analyze this infographic against the following blog content for:\n"
            "1. Factual Accuracy: Does it correctly represent the information?\n"
            "2. Spelling: Are there any typos or spelling errors in the text?\n"
            "3. Aesthetic Alignment: Does the style match a professional blog?\n\n"
            "If it fails any criteria, provide specific feedback for improvement. "
            "If it passes all criteria, respond ONLY with 'PASS'.\n\n"
            f"Blog Content: {blog_content}"
        )

        response = client.models.generate_content(
            model="gemini-3-flash-preview",
            contents=[img, evaluation_prompt],
        )

        result = response.text.strip()
        log_step(f"Evaluation result: {result}")
        return result
    except Exception as e:
        log_step(f"Error evaluating infographic: {e}")
        return f"Evaluation Error: {e}"


def infographic_workflow(url: str) -> str:
    """Main workflow: fetch, generate, evaluate, and regenerate as needed.

    Tries up to 3 times: each attempt generates an infographic and evaluates it.
    On 'PASS' it returns immediately; otherwise the evaluation feedback is fed
    into the next generation attempt.
    """
    log_step(f"Starting infographic workflow for URL: {url}")

    content = fetch_blog_content(url)
    if "Error" in content:
        return content

    max_attempts = 3
    feedback = ""

    for attempt in range(max_attempts):
        log_step(f"Attempt {attempt + 1} of {max_attempts}")
        image_path = generate_infographic(content, feedback)

        if "Error" in image_path:
            return image_path

        eval_result = evaluate_infographic(image_path, content)

        if eval_result == "PASS":
            log_step("Infographic passed evaluation.")
            return f"Success! Infographic saved at: {image_path}"
        else:
            feedback = eval_result
            log_step(f"Regenerating due to feedback: {feedback}")

    log_step("Reached maximum attempts. Returning last version.")
    return f"Workflow completed with feedback (check log): {image_path}"


# ---------------------------------------------------------------------------
# ADK model + agent definition
# ---------------------------------------------------------------------------
class VertexGemini(Gemini):
    """Gemini model subclass that injects a custom Vertex client.

    Overrides `api_client` to return a cached genai.Client configured with the
    environment's credentials and base URL.
    """

    _client: genai.Client = None  # class-level cache

    @property
    def api_client(self) -> genai.Client:
        if self._client is None:
            self._client = genai.Client(
                project=project_id,
                location="global",
                credentials=credentials,
                http_options=types.HttpOptions(
                    base_url=os.getenv("GOOGLE_VERTEX_BASE_URL")
                ),
            )
        return self._client


# Define the ADK Agent that drives the workflow.
root_agent = LlmAgent(
    name="InfographicAgent",
    model=VertexGemini(model="gemini-3-flash-preview"),
    instruction=(
        "You are an expert at creating infographics from blog posts. "
        "Use the tools provided to fetch content, generate an image, and validate it. "
        "Always log your progress."
    ),
    tools=[infographic_workflow],
)


# ---------------------------------------------------------------------------
# Run the agent
# ---------------------------------------------------------------------------
async def main():
    """Run the infographic agent against a sample blog post."""
    BLOG = "https://cloud.google.com/blog/products/ai-machine-learning/ultimate-prompting-guide-for-lyria-3-pro"

    runner = InMemoryRunner(agent=root_agent, app_name="image_agent")

    session = await runner.session_service.create_session(
        app_name="image_agent", user_id="user"
    )

    user_message = types.Content(
        role="user",
        parts=[types.Part(text=(f"Create an infographic from this blog: {BLOG}"))],
    )

    async for event in runner.run_async(
        user_id="user",
        session_id=session.id,
        new_message=user_message,
    ):
        if event.content and event.content.parts:
            for part in event.content.parts:
                if part.text:
                    print(part.text)
                if part.function_call:
                    print(f"\u2192 Calling: {part.function_call.name}")


if __name__ == "__main__":
    # Run the async workflow via the event loop:
    import asyncio

    asyncio.run(main())
