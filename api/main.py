from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from google import genai
from google.genai import types
from typing import Optional
import re
import json
import os

app = FastAPI()

# CORS so the frontend (even if hosted on a different domain) can call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Optional fallback key if you ever want a server-wide default.
# With the "each user pastes their own key" setup, this can stay unset.
DEFAULT_GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

SYSTEM_PROMPT = """You are a medical lab report analyzer. The user will upload an image or PDF of a medical lab report.
Your job is to carefully analyze it and return a JSON response (and ONLY JSON, no markdown, no extra text) in this exact format:
{
  "summary": "A 2-3 sentence plain English summary of what this report is and the patient's overall status.",
  "normal_values": ["Parameter 1: value (normal range)", "Parameter 2: value (normal range)"],
  "abnormal_values": ["Parameter: value — Reference range: X-Y. This is HIGH/LOW."],
  "possible_conditions": ["Condition name: brief explanation of why this might be indicated"],
  "recommendations": ["Recommendation 1", "Recommendation 2", "Always consult a doctor for proper diagnosis and treatment"]
}
Rules:
- If this is NOT a medical lab report, return: {"error": "This does not appear to be a medical lab report. Please upload a valid lab report."}
- Be thorough but use simple, clear language a non-doctor can understand.
- For abnormal values, clearly state if the value is HIGH or LOW compared to normal range.
- Always include Consult a qualified doctor in recommendations.
- Return ONLY valid JSON. No preamble, no markdown backticks.
"""

ALLOWED_MIME_TYPES = {"image/jpeg", "image/jpg", "image/png", "image/webp", "application/pdf"}


@app.get("/")
async def root():
    return FileResponse(
        os.path.join(os.path.dirname(__file__), "..", "index.html")
    )


@app.post("/api/analyze")
async def analyze_report(file: UploadFile = File(...), x_gemini_key: Optional[str] = Header(None)):
    try:
        api_key = x_gemini_key or DEFAULT_GEMINI_API_KEY
        if not api_key:
            return JSONResponse(
                status_code=400,
                content={"error": "No Gemini API key provided. Please paste your API key in the box above."},
            )

        client = genai.Client(api_key=api_key)

        mime_type = file.content_type

        if mime_type not in ALLOWED_MIME_TYPES:
            return JSONResponse(
                status_code=400,
                content={"error": f"Unsupported file type: {mime_type}. Please upload a JPG, PNG, WEBP, or PDF."},
            )

        file_bytes = await file.read()

        if not file_bytes:
            return JSONResponse(status_code=400, content={"error": "Uploaded file is empty."})

        response = client.models.generate_content(
            model="gemini-2.0-flash",  # Using 2.0 flash as it's common and stable
            contents=[
                types.Part.from_bytes(data=file_bytes, mime_type=mime_type),
                SYSTEM_PROMPT
            ]
        )

        raw = response.text.strip()
        raw = re.sub(r"^```(?:json)?", "", raw).strip()
        raw = re.sub(r"```$", "", raw).strip()

        result = json.loads(raw)
        return result
    except json.JSONDecodeError:
        return JSONResponse(
            status_code=500,
            content={"error": "Could not parse the AI's response. Please try again."},
        )
    except Exception as e:
        error_str = str(e)
        if "API_KEY_INVALID" in error_str or "API key not valid" in error_str:
            return JSONResponse(
                status_code=401,
                content={"error": "That Gemini API key looks invalid. Please check and paste it again."},
            )
        return JSONResponse(status_code=500, content={"error": error_str})
