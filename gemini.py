import os
from flask import Flask, jsonify, request
from dotenv import load_dotenv
from google import genai

load_dotenv(override=True)
 
app = Flask(__name__)

API_KEY = os.getenv("GEMINI_API_KEY")
if not API_KEY: 
    raise ValueError("GEMINI_API_KEY not found in .env")

client = genai.Client(api_key=API_KEY)

# ── Load Guidance ─────────────────────────────────────────────────────────────

LOAD_GUIDANCE = {
    "low": (
        "The student seems to have a good grasp of the basics. "
        "Push them to think a bit deeper — are there any details, examples, "
        "or conditions they haven't considered yet?"
    ),
    "medium": (
        "The student is partially there but missing something important. "
        "Pick the ONE biggest gap in their answer and nudge them toward it "
        "using simple, everyday language. No jargon."
    ),
    "high": (
        "The student is lost or overwhelmed. Give ONE tiny, simple nudge "
        "that gets them unstuck — like a friend pointing them in the right direction. "
        "Use short words, a relatable example if it helps, and keep it very warm."
    ),
}

# ── Prompt Builder ────────────────────────────────────────────────────────────

def build_prompt(problem: str, answer: str, label: str, prev_hint: str) -> str:
    load_guidance = LOAD_GUIDANCE.get(label, LOAD_GUIDANCE["medium"])
    answer_text   = answer.strip() or "(the student hasn't written anything yet)"

    prev_section = ""
    if prev_hint.strip():
        prev_section = (
            f"HINT YOU ALREADY GAVE:\n\"{prev_hint}\"\n"
            "→ Don't repeat this. Move on to the next most important thing they're missing.\n\n"
        )

    return f"""You are a strict computer science evaluator generating a corrective hint.

QUESTION:
{problem}

STUDENT ANSWER:
{answer_text}

COGNITIVE LOAD:
{label.upper()}
{load_guidance}

{prev_section}OBJECTIVE:
Identify the single most critical missing or incorrect technical concept and guide the student to fix it.

INTERNAL EVALUATION (do not output):
- What are the key technical components required for a correct answer?
- Which of those components are missing, vague, or incorrect?
- What is the highest-impact correction needed right now?

HINT RULES:
- Output exactly 1–2 sentences.
- Use precise technical terminology (e.g., linear structure, hierarchical structure, indexing, relationships, ordering).
- The hint must be directly tied to the question domain.
- Explicitly point out what is missing, incorrect, or too vague.
- Do NOT include examples like toys/books or unrelated analogies.
- Do NOT provide full definitions or complete answers.
- Do NOT include praise, conversational tone, or filler.
- Do NOT repeat the question or previous hint.
- If the answer is empty: specify the core concept(s) that must be defined.
- Prefer actionable corrections (e.g., “include classification…”, “distinguish between…”, “mention structure types…”).

QUALITY CONSTRAINTS:
- Specific > generic
- Technical > descriptive
- Relevant > broad
- Minimal but precise

OUTPUT FORMAT:
Return only the hint text (no labels, no explanation).

HINT:"""

# ── Route ─────────────────────────────────────────────────────────────────────

@app.route("/generate-hint", methods=["POST"])
def generate_hint():
    data = request.get_json(force=True)

    problem   = data.get("problem", "")
    answer    = data.get("answer", "")
    label     = data.get("label", "medium")
    prev_hint = data.get("prev_hint", "")

    prompt = build_prompt(problem, answer, label, prev_hint)

    models_to_try = ["gemini-2.5-flash-lite", "gemini-2.5-flash"]

    for model in models_to_try:
        try:
            response = client.models.generate_content(
                model=model,
                contents=prompt,
            )
            raw_text = (response.text or "").strip()
            lines = [l.strip() for l in raw_text.splitlines() if l.strip()]
            hint  = " ".join(lines[:2])
            return jsonify({"hint": hint})

        except Exception as e:
            print(f"[Gemini error] model={model} error={e}")
            is_503 = "503" in str(e) or "UNAVAILABLE" in str(e)
            if not is_503:
                return jsonify({"hint": "Try breaking the problem into smaller steps."})

    return jsonify({"hint": "Try breaking the problem into smaller steps."})

if __name__ == "__main__":
    app.run(debug=True, port=6000)
