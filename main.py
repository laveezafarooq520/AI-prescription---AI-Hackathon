import json
import os

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI, OpenAIError
from pydantic import BaseModel, Field

load_dotenv()

OPENAI_MODEL = os.getenv("LLM_FAST_MODEL") or os.getenv("OPENAI_MODEL", "gpt-4.1-mini")

app = FastAPI(title="AI Prescription API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)

SYSTEM_PROMPT = """You are a pharmacology reference assistant. Given a medicine name,
return general, standard prescribing information for it as JSON with exactly these keys:

{
  "found": true,
  "medicine_name": "brand or common name",
  "generic_name": "active ingredient(s)",
  "drug_class": "pharmacological class",
  "uses": ["condition it treats", ...],
  "dosage": {
    "adults": "typical adult dose and frequency",
    "children": "typical pediatric dose, or 'Not recommended' / 'Consult a doctor'"
  },
  "how_to_take": "with/without food, timing, duration notes",
  "side_effects": ["common side effect", ...],
  "warnings": ["important precaution or contraindication", ...],
  "interactions": ["notable drug or food interaction", ...]
}

If the input is not a recognizable medicine, return {"found": false, "message": "short reason"}.
Be accurate and concise. Use standard reference doses only; never invent a medicine."""

INTERACTION_PROMPT = """You are a pharmacology reference assistant. Given a list of medicines,
check every pair for known drug-drug interactions and return JSON with exactly these keys:

{
  "found": true,
  "medicines": ["normalized medicine name", ...],
  "overall_severity": "none" | "minor" | "moderate" | "major",
  "summary": "one or two sentence plain-language summary",
  "interactions": [
    {
      "medicines": ["drug A", "drug B"],
      "severity": "minor" | "moderate" | "major",
      "effect": "what happens when taken together",
      "advice": "what a patient should do"
    }
  ]
}

Only list pairs that actually interact; use an empty list and "none" if there are no known interactions.
If any input is not a recognizable medicine, return {"found": false, "message": "which name was not recognized"}.
Be accurate and concise; never invent interactions."""


class PrescriptionRequest(BaseModel):
    medicine: str = Field(..., min_length=1, max_length=100)


class InteractionRequest(BaseModel):
    medicines: list[str] = Field(..., min_length=2, max_length=5)


def get_client() -> OpenAI:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise HTTPException(
            status_code=500,
            detail="OPENAI_API_KEY is not set. Add it to backend/.env and restart the server.",
        )
    return OpenAI(api_key=api_key)


def ask_json(system_prompt: str, user_message: str) -> dict:
    """Send a prompt to OpenAI and return the parsed JSON answer, raising 404 if not found."""
    client = get_client()
    try:
        completion = client.chat.completions.create(
            model=OPENAI_MODEL,
            response_format={"type": "json_object"},
            temperature=0.2,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
        )
    except OpenAIError as e:
        raise HTTPException(status_code=502, detail=f"OpenAI request failed: {e}")

    try:
        data = json.loads(completion.choices[0].message.content)
    except (json.JSONDecodeError, TypeError):
        raise HTTPException(status_code=502, detail="Could not read the AI response. Please try again.")

    if not data.get("found", False):
        raise HTTPException(status_code=404, detail=data.get("message") or "Medicine not recognized.")

    return data


@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.post("/api/prescription")
def prescription(req: PrescriptionRequest):
    medicine = req.medicine.strip()
    if not medicine:
        raise HTTPException(status_code=400, detail="Please enter a medicine name.")
    return ask_json(SYSTEM_PROMPT, f"Medicine: {medicine}")


@app.post("/api/interactions")
def interactions(req: InteractionRequest):
    medicines = [m.strip() for m in req.medicines if m.strip()]
    if len(medicines) < 2:
        raise HTTPException(status_code=400, detail="Please enter at least two medicines.")
    if any(len(m) > 100 for m in medicines):
        raise HTTPException(status_code=400, detail="Medicine names must be 100 characters or less.")
    return ask_json(INTERACTION_PROMPT, "Medicines: " + ", ".join(medicines))
