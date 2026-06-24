from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from google import genai
from google.genai import types
import json
import re
import os

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

SYSTEM_PROMPT = """You are a medical lab report analyzer. Analyze the provided image or PDF.
Return ONLY a JSON response in this exact format:
{
  "summary": "2-3 sentence summary",
  "normal_values": ["param: val"],
  "abnormal_values": ["param: val (High/Low)"],
  "possible_conditions": ["name: reason"],
  "recommendations": ["advice"]
}
If not a medical report, return {"error": "Invalid report"}.
"""

@app.post("/api/analyze")
async def analyze(file: UploadFile = File(...), apiKey: str = Form(...)):
    try:
        client = genai.Client(api_key=apiKey)
        content = await file.read()
        
        # Determine mime type
        ext = os.path.splitext(file.filename)[1].lower()
        mime_map = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png", ".pdf": "application/pdf"}
        mime_type = mime_map.get(ext, "image/jpeg")

        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=[
                types.Part.from_bytes(data=content, mime_type=mime_type),
                SYSTEM_PROMPT
            ]
        )
        
        raw = response.text.strip()
        raw = re.sub(r"^\`\`\`(?:json)?", "", raw).strip()
        raw = re.sub(r"\`\`\`$", "", raw).strip()
        return json.loads(raw)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
def home():
    return {"message": "MediScan API is running"}
