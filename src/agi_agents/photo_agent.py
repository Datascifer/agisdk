"""
PhotoWorkflowAgent

A custom REAL Agent that:
- Reads the browser observation (goal, URL, AX tree, last_action, etc.)
- Classifies the goal into one of 5 photography pipeline stages:
  1) Ingestion & Asset Management
  2) Tagging & Categorization
  3) Editing & Enhancements
  4) Storytelling & Content Creation
  5) Publishing & Distribution
- Builds a structured prompt for an LLM describing:
  - The user goal
  - The inferred pipeline stage
  - The current page + elements (from axtree)
  - The allowed REAL actions (click/fill/send_msg_to_user/report_infeasible/noop)
- Asks the LLM for exactly one next action string, and returns that to the harness.

This is “real logic” in the sense that:
- It actually calls an LLM (OpenAI client) at each step.
- It is aware of your product’s photography workflow and uses that to guide strategy.
- It turns low-level web observations into a higher-level, photography-aware decision loop.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import re

from openai import OpenAI, OpenAIError  # Make sure openai is in pyproject/requirements.

from agisdk.REAL.agent import Agent, AbstractAgentArgs
from agisdk.REAL.logging import logger

# Autoselect and apply image preset
from agi_agents.preset_image import auto_select_best_and_apply_preset

class PhotoWorkflowAgent(Agent):
    ...
    def some_internal_method(...):
        best_path, processed_img, info = auto_select_best_and_apply_preset(
            image_paths=burst_paths,
            preset_name="social_pop",
        )
        ...

# ---------------------------------------------------------------------------
# 1. Domain: photography pipeline categories (your 5 big buckets)
# ---------------------------------------------------------------------------

PHOTO_PIPELINE_CATEGORIES = {
    "ingestion": {
        "label": "Ingestion, Organization & Asset Management",
        "keywords": [
            "upload",
            "ingest",
            "ingestion",
            "import",
            "sync",
            "culling",
            "sort",
            "sorting",
            "backup",
            "archive",
            "duplicate",
            "blurry",
            "cloud",
            "drive",
            "gallery",
            "metadata",
            "exif",
            "iptc",
        ],
    },
    "tagging": {
        "label": "Tagging, Categorization & Contextual Understanding",
        "keywords": [
            "tag",
            "tagging",
            "label",
            "categorize",
            "category",
            "face",
            "subject",
            "color",
            "shot list",
            "shot-list",
            "style",
            "version",
            "crop",
        ],
    },
    "editing": {
        "label": "Editing & Enhancements",
        "keywords": [
            "edit",
            "editing",
            "light",
            "lighting",
            "exposure",
            "contrast",
            "noise",
            "sharpen",
            "batch",
            "color correction",
            "grade",
            "grading",
            "thumbnail",
            "audio",
            "sync",
        ],
    },
    "storytelling": {
        "label": "Storytelling & Content Creation",
        "keywords": [
            "highlight reel",
            "recap",
            "recap video",
            "story arc",
            "multiday",
            "multi-day",
            "sequence",
            "timeline",
            "creative variation",
            "variation",
            "crop for platform",
        ],
    },
    "publishing": {
        "label": "Publishing & Distribution",
        "keywords": [
            "hashtag",
            "hashtags",
            "post",
            "posting",
            "publish",
            "publishing",
            "deliverable",
            "client gallery",
            "proof",
            "approval",
            "deadline",
            "delivery",
            "schedule",
            "social",
        ],
    },
}


def _classify_goal(goal: str) -> str:
    """
    Very lightweight heuristic classifier:
    - Scans the goal text for pipeline-specific keywords.
    - Returns one of: "ingestion", "tagging", "editing", "storytelling", "publishing", or "unknown".

    We use this as a hint to the LLM (not a hard constraint).
    """
    if not goal:
        return "unknown"

    goal_lower = goal.lower()
    best_match = "unknown"
    best_score = 0

    for stage, cfg in PHOTO_PIPELINE_CATEGORIES.items():
        score = sum(1 for kw in cfg["keywords"] if kw in goal_lower)
        if score > best_score:
            best_score = score
            best_match = stage

    return best_match if best_score > 0 else "unknown"


# ---------------------------------------------------------------------------
# 2. Agent implementation
# ---------------------------------------------------------------------------


class PhotoWorkflowAgent(Agent):
    """
    REAL-compatible agent for the AGI hackathon.

    Design goals:
    - Use the REAL observation structure (goal, url, axtree_object, etc.).
    - Inject photography-specific understanding (your 5 buckets) into the system prompt.
    - Use an LLM to pick a next REAL action (click/fill/press/send_msg_to_user/report_infeasible/noop).
    """

    def __init__(
        self,
        model_name: str = "gpt-4o",
        use_html: bool = False,
        use_axtree: bool = True,
        use_screenshot: bool = True,
        temperature: float = 0.1,
    ) -> None:
        self.model_name = model_name
        self.use_html = use_html
        self.use_axtree = use_axtree
        self.use_screenshot = use_screenshot
        self.temperature = temperature

        # OpenAI client (reads OPENAI_API_KEY from environment by default)
        self.client = OpenAI()

        # Optional: store action history if you want multi-step memory beyond obs["chat_messages"]
        self.action_history: List[str] = []

        logger.info(
            f"[PhotoWorkflowAgent] initialized with model={model_name}, "
            f"use_html={use_html}, use_axtree={use_axtree}, use_screenshot={use_screenshot}"
        )

    # ----- Observation preprocessing ------------------------------------------------

    def obs_preprocessor(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert raw REAL observation into a compact summary for the LLM.

        REAL observation fields (from README):
        {
            'chat_messages': [...],
            'goal': '...',
            'goal_object': [...],
            'open_pages_urls': [...],
            'active_page_index': 0,
            'url': '...',
            'screenshot': np.array(...),
            'dom_object': {...},
            'axtree_object': {...},
            'extra_element_properties': {...},
            'focused_element_bid': '...',
            'last_action': '...',
            'last_action_error': '...',
            'elapsed_time': 0.0,
            'browser': {...}
        }

        We:
        - Keep goal/url/last_action/error directly.
        - Flatten axtree into a human-readable list of elements (bid, role, name) with a small cap.
        - Add a "goal_stage" classification based on your photography pipeline.
        """
        goal: str = obs.get("goal", "") or ""
        url: str = obs.get("url", "") or ""
        last_action: str = obs.get("last_action", "") or ""
        last_error: str = obs.get("last_action_error", "") or ""
        elapsed_time: float = obs.get("elapsed_time", 0.0) or 0.0

        goal_stage = _classify_goal(goal)

        # Flatten accessibility tree into a small list of element descriptors
        elements: List[Dict[str, Any]] = []
        if self.use_axtree and obs.get("axtree_object") is not None:
            elements = self._flatten_axtree(obs["axtree_object"], limit=80)

        processed = {
            "goal": goal,
            "goal_stage": goal_stage,
            "url": url,
            "last_action": last_action,
            "last_action_error": last_error,
            "elapsed_time": elapsed_time,
            "elements": elements,  # e.g. [{"bid": "bid:123", "role": "button", "name": "Upload"}]
        }

        return processed

    def _flatten_axtree(self, axtree: Any, limit: int = 80) -> List[Dict[str, Any]]:
        """
        Generic traversal of the accessibility tree:
        - Walks dicts/lists recursively.
        - Collects nodes that look like UI elements: have a "bid" or at least a "role"/"name".
        - Returns at most `limit` nodes for token efficiency.

        This is intentionally simple and robust against format changes.
        """
        results: List[Dict[str, Any]] = []

        def _walk(node: Any) -> None:
            nonlocal results
            if len(results) >= limit:
                return

            if isinstance(node, dict):
                bid = node.get("bid")
                role = node.get("role") or node.get("tag") or ""
                name = node.get("name") or node.get("text") or node.get("label") or ""

                if bid or role or name:
                    results.append(
                        {
                            "bid": bid,
                            "role": role,
                            "name": name,
                        }
                    )

                # Recurse into children / generic keys
                for v in node.values():
                    if isinstance(v, (dict, list)):
                        _walk(v)

            elif isinstance(node, list):
                for item in node:
                    _walk(item)

        _walk(axtree)
        return results

    # ----- Prompt building ----------------------------------------------------------

    def _build_messages(self, processed: Dict[str, Any]) -> List[Dict[str, str]]:
        """
        Build the chat messages for the LLM.

        System message:
        - Describes the agent as a photographer-first workflow assistant.
        - Explains the 5 pipeline stages and how they relate to the user goal.
        - Describes the REAL action space and strict output format.

        User message:
        - Injects the current goal, inferred stage, URL, last action/error, and a list of elements.
        - Asks for exactly ONE next action as a valid action string.
        """
        goal = processed["goal"]
        stage_key = processed["goal_stage"]
        url = processed["url"]
        last_action = processed["last_action"]
        last_error = processed["last_action_error"]
        elapsed_time = processed["elapsed_time"]
        elements = processed["elements"]

        if stage_key in PHOTO_PIPELINE_CATEGORIES:
            stage_label = PHOTO_PIPELINE_CATEGORIES[stage_key]["label"]
        else:
            stage_label = "Unknown / mixed stage"

        # Short description of each category for the system prompt
        stage_descriptions = [
            f"- Ingestion & Asset Management: ingesting, sorting, culling, deduplicating, backing up, managing metadata and galleries.",
            f"- Tagging & Categorization: tagging faces/subjects/colors, categorizing image types, enforcing shot lists, maintaining style/versioning.",
            f"- Editing & Enhancements: lighting, color, noise/sharpening, batch edits, creative video edits, thumbnails, style matching.",
            f"- Storytelling & Content Creation: highlight reels, recap videos, story arcs across days, creative variations, platform-ready crops.",
            f"- Publishing & Distribution: hashtags, social-ready exports, client deliverables/proofs, approvals, delivery timeline tracking.",
        ]

        system_text = (
            "You are PhotoWorkflowAgent, an autonomous web agent helping a photographer-first "
            "platform automate the entire creative pipeline.\n\n"
            "The pipeline stages are:\n"
            + "\n".join(stage_descriptions)
            + "\n\n"
            "You operate inside a browser environment (REAL / BrowserGym). At each step you must "
            "choose ONE action from the REAL action space:\n"
            "- click('element_bid')               # click on a UI element by its bid\n"
            "- fill('element_bid', 'text')        # type text into an input field\n"
            "- press('Enter') / press('Tab') ...  # keyboard keys by name\n"
            "- goto('https://...')                # navigate to a URL (rare, usually the environment handles this)\n"
            "- send_msg_to_user('message')        # explain progress or final result\n"
            "- report_infeasible('reason')        # if the requested task truly cannot be done\n"
            "- noop(ms)                           # wait for a number of milliseconds\n\n"
            "You MUST output exactly ONE valid action string (like: click('bid:123')) and nothing else. "
            "Do not add explanations or extra text."
        )

        # Compress element list into a compact text table for the LLM
        element_lines = []
        for el in elements[:40]:  # extra safety cap
            bid = el.get("bid") or "None"
            role = el.get("role") or ""
            name = el.get("name") or ""
            element_lines.append(f"- bid={bid!r}, role={role!r}, name={name!r}")

        elements_block = "\n".join(element_lines) if element_lines else "(no elements visible)"

        user_text = (
            f"GOAL: {goal}\n"
            f"INFERRED PIPELINE STAGE: {stage_label} (key={stage_key})\n"
            f"CURRENT URL: {url}\n"
            f"ELAPSED TIME (s): {elapsed_time:.1f}\n"
            f"LAST ACTION: {last_action or '(none)'}\n"
            f"LAST ACTION ERROR: {last_error or '(none)'}\n\n"
            "VISIBLE ELEMENTS (from accessibility tree):\n"
            f"{elements_block}\n\n"
            "Decide the single best next action to move towards completing the GOAL, "
            "keeping the inferred pipeline stage in mind.\n"
            "Return just one action string (e.g., click('bid:123') or fill('bid:456', 'wedding gallery 2024'))."
        )

        messages: List[Dict[str, str]] = [
            {"role": "system", "content": system_text},
            {"role": "user", "content": user_text},
        ]
        return messages

    # ----- Action selection ---------------------------------------------------------

    def get_action(self, obs: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
        """
        Core decision loop: observation -> LLM call -> action string.

        Steps:
        1. Preprocess obs into a compact, photography-aware summary.
        2. Build system+user messages that:
           - Encode your 5 task buckets
           - Provide relevant page elements and context
           - Strictly specify the action string format
        3. Call the LLM and parse out a single action string.
        4. Fall back to 'noop(1000)' if something goes wrong.
        """
        processed = self.obs_preprocessor(obs)
        messages = self._build_messages(processed)

        logger.debug(
            f"[PhotoWorkflowAgent] step: url={processed.get('url')}, "
            f"goal_stage={processed.get('goal_stage')}"
        )

        raw_output = ""
        action = "noop(1000)"  # safe default
        info: Dict[str, Any] = {
            "goal_stage": processed.get("goal_stage"),
            "debug": "",
        }

        try:
            completion = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=self.temperature,
            )

            raw_output = completion.choices[0].message.content.strip()
            logger.debug(f"[PhotoWorkflowAgent] raw LLM output: {raw_output!r}")

            action = self._extract_action(raw_output) or "noop(1000)"
            info["debug"] = f"raw_output={raw_output!r}"

        except OpenAIError as e:
            # If the API fails, we log and fall back to noop so the episode doesn't crash.
            logger.error(f"[PhotoWorkflowAgent] OpenAI error: {e}")
            info["debug"] = f"OpenAIError: {e!r}"
            action = "noop(1000)"

        # Track history (optional, useful for debugging)
        self.action_history.append(action)

        return action, info

    def _extract_action(self, text: str) -> str | None:
        """
        Parse the model's output and pull out ONE valid action string.

        We expect something like:
          click('bid:123')
          fill("bid:456", "search term")
          send_msg_to_user("done")
          report_infeasible("reason")
          noop(1000)

        Strategy:
        - Use a regex to grab the first function-call-looking thing.
        - Strip surrounding code fences or quoting if the model added them.
        """
        if not text:
            return None

        # Remove code fences if the LLM wrapped the answer in ```python ... ```
        fenced = re.sub(r"^```[a-zA-Z]*\s*|\s*```$", "", text.strip(), flags=re.MULTILINE)

        # Very simple "function(...)" pattern – this is enough for REAL actions.
        m = re.search(r"([a-zA-Z_]+\(.*\))", fenced)
        if m:
            return m.group(1).strip()

        # If nothing matches, we give up.
        return None

    # ----- Lifecycle ---------------------------------------------------------------

    def close(self) -> None:
        """
        Cleanup any resources (none needed for OpenAI client, but we log for completeness).
        """
        logger.info(
            f"[PhotoWorkflowAgent] closing, total_actions={len(self.action_history)}"
        )


# ---------------------------------------------------------------------------
# 3. AgentArgs for harness integration
# ---------------------------------------------------------------------------


@dataclass
class PhotoWorkflowAgentArgs(AbstractAgentArgs):
    """
    Harness-facing config object (like DemoAgentArgs / ManualAgentArgs).

    The REAL harness will:
    - Construct this dataclass (optionally from CLI/config).
    - Call make_agent() to get a concrete PhotoWorkflowAgent instance.
    - Pass observations to get_action() in a loop.

    agent_name is useful for logging / leaderboard metadata.
    """

    agent_name: str = "photo-workflow-agent"

    model_name: str = "gpt-4o"
    use_html: bool = False
    use_axtree: bool = True
    use_screenshot: bool = True
    temperature: float = 0.1

    def make_agent(self) -> PhotoWorkflowAgent:
        return PhotoWorkflowAgent(
            model_name=self.model_name,
            use_html=self.use_html,
            use_axtree=self.use_axtree,
            use_screenshot=self.use_screenshot,
            temperature=self.temperature,
        )