from fastapi import FastAPI, Request, Response, Cookie
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from transformers.pipelines import pipeline
import base64
import subprocess
import os
from attestation import Attestation

app = FastAPI()

# Load a compact language model (e.g., DistilGPT2)
text_gen = pipeline("text-generation", model="distilgpt2")

# Mount templates and static files
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

class ChatInput(BaseModel):
    message: str

@app.get("/", response_class=HTMLResponse)
async def get_ui(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/favicon.ico")
async def favicon():
    return FileResponse(os.path.join("static", "favicon.ico"))

@app.post("/chat")
async def chat(input: ChatInput, attested: str = Cookie(default=None)):
    if attested != "true":
        return JSONResponse(status_code=403, content={"error": "Environment not attested. Please attest environment first."})
    try:
        response = text_gen(input.message, max_length=100, num_return_sequences=1)
        if not response or not isinstance(response, list) or "generated_text" not in response[0]:
            return JSONResponse(status_code=500, content={"error": "Model did not return a valid response."})
        return {"response": response[0]["generated_text"]}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.get("/attest")
async def attest(response: Response):
    try:
        att = Attestation()
        # Generate attestation report and read the report file
        report_path = att.generate_report()
        with open(report_path, "rb") as f:
            report = f.read()
        report_b64 = base64.b64encode(report).decode("utf-8")
        # Verify the attestation report
        print(f"Verifying attestation report at {report_path}")
        verification_result = att.verify_report()
        print("Verification Result - ", verification_result)
        if verification_result == "verified":
            response.set_cookie(key="attested", value="true", httponly=True)
        else:
            response.delete_cookie(key="attested")
        return {
            "attestation_report_base64": report_b64,
            "verification_result": verification_result
        }
    except FileNotFoundError:
        return JSONResponse(status_code=404, content={"error": "attestation utility or file not found"})
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
