# AI Prescription

Enter a medicine name and get its standard prescription information (uses, dosage, how to take, side effects, warnings, interactions), generated with OpenAI.

- `backend/` — FastAPI server (`POST /api/prescription`)
- `frontend/` — React (Vite) home page

## 1. Backend

```bash
cd backend
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
```

Open `backend/.env` and set your key:

```
OPENAI_API_KEY=sk-...
LLM_FAST_MODEL=gpt-4.1-mini
```

Run it:

```bash
uvicorn main:app --reload --port 8000
```

## 2. Frontend

In a second terminal:

```bash
cd frontend
npm install
npm run dev
```

Open http://localhost:5173. The dev server forwards `/api` requests to the backend on port 8000.

> The output is AI-generated for educational purposes and is not medical advice.
