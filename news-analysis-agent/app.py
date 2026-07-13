
from dotenv import load_dotenv
load_dotenv()
from smolagents import CodeAgent, DuckDuckGoSearchTool, HfApiModel, tool
import yaml
from tools.final_answer import FinalAnswerTool
from tools.visit_webpage import VisitWebpageTool
from tools.web_search import DuckDuckGoSearchTool

from Gradio_UI import GradioUI




# Основная модель агента — для ReAct-цикла (нужен код)
model = HfApiModel(
    max_tokens=2096,
    temperature=0.5,
    model_id='Qwen/Qwen2.5-Coder-32B-Instruct',
)

# Отдельная модель для интерпретативного анализа — нужно рассуждение, не код
model_analysis = HfApiModel(
    max_tokens=2096,
    temperature=0.2,
    model_id='Qwen/Qwen2.5-72B-Instruct',
)


@tool
def analyze_news_text(text: str) -> str:
    """Analyzes a news article: classifies its sentiment, then performs a Derridean-style
    double reading (reconstructs the dominant framing, then identifies the internal
    opposition/exclusion that destabilizes it).
    Args:
        text: The news article text to analyze.
    """
    prompt = f"""Analyze this news text in three labeled steps.

STEP 1 — Sentiment: Classify the overall sentiment as Positive, Negative, or Neutral,
with one sentence justifying the classification.

STEP 2 — Doubling commentary: Reconstruct the text's dominant framing on its own terms.
What hierarchy or opposition does it assert (e.g. us/them, order/chaos, legitimate/illegitimate)?
What is its explicit argument?

STEP 3 — Deconstructive reading: Find a specific WORD OR PHRASE in the text describing
the "solution" side of the opposition that could equally describe the "problem" side —
a lexical echo, not just a shared abstract goal. The move is NOT "both sides ultimately
want the same thing." The move IS "the word used for the cure is the same kind of word
used for the disease."

Example A: A text argues "we must act firmly against the aggressor to preserve peace."
Opposition: peace/aggression. Deconstructive move: "firmly" and "act against" are
words of force and disruption — the same vocabulary that would describe the aggression
itself. The peace-preserving action is lexically indistinguishable from aggression.

Example B: A text argues "the surgeon must cut into healthy tissue to remove the tumor."
Opposition: healing/harming. Deconstructive move: "cut into healthy tissue" is, by
definition, an act of harm — the healing act is described using the exact vocabulary
of the harm (tissue damage) it exists to prevent.

Do not just paraphrase a word into a synonym and call it deconstruction. Do not say
"both sides share a goal." Find the actual shared or contradictory vocabulary.

Text: {text[:2000]}

Respond with three clearly labeled sections."""

    response = model_analysis([{"role": "user", "content": prompt}])
    return response.content


final_answer = FinalAnswerTool()
web_search = DuckDuckGoSearchTool()
visit_webpage = VisitWebpageTool()

with open("prompts.yaml", 'r') as stream:
    prompt_templates = yaml.safe_load(stream)

agent = CodeAgent(
    model=model,
    tools=[final_answer, web_search, visit_webpage, analyze_news_text],
    max_steps=8,
    verbosity_level=1,
    grammar=None,
    planning_interval=None,
    name=None,
    description=None,
    prompt_templates=prompt_templates
)

GradioUI(agent).launch()