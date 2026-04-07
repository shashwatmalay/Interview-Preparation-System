from __future__ import annotations

import uuid
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from dotenv import load_dotenv
import os
import smtplib
from email.message import EmailMessage
from reportlab.lib.pagesizes import LETTER
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer

from app.agents.agent_1_parser import run_agent1
from app.agents.agent_2_researcher import run_agent2
from app.agents.agent_3_qa_gen import run_agent3

BASE_DIR = Path(__file__).resolve().parents[1]
OUTPUT_DIR = BASE_DIR / "app" / "output"
FRONTEND_DIR = BASE_DIR / "frontend"
IMG_DIR = FRONTEND_DIR / "imgvid"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="Interview Preparation System API")

load_dotenv()

# Allow local dev from any origin (adjust later for production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve generated files (e.g., text outputs)
app.mount("/outputs", StaticFiles(directory=str(OUTPUT_DIR)), name="outputs")

# Serve frontend assets at their expected paths
if IMG_DIR.exists():
    app.mount("/imgvid", StaticFiles(directory=str(IMG_DIR)), name="imgvid")


@app.get("/favicon.ico")
def favicon():
    ico_path = IMG_DIR / "favicon001.ico"
    if ico_path.exists():
        return FileResponse(str(ico_path))
    return JSONResponse({"detail": "Not found"}, status_code=404)

RUN_STORE: Dict[str, Dict[str, Any]] = {}

def build_pdf(run_id: str, title: str, questions: list[str], meta: Dict[str, Any]) -> Path:
    pdf_path = OUTPUT_DIR / f"interview_qa_{run_id}.pdf"
    styles = getSampleStyleSheet()
    story = []

    story.append(Paragraph(title, styles["Title"]))
    story.append(Spacer(1, 12))

    subtitle = f"Answer length: {meta.get('answer_length', '-')}; Rounds: {meta.get('interview_rounds', '-')}"
    story.append(Paragraph(subtitle, styles["Normal"]))
    story.append(Spacer(1, 12))

    for i, q in enumerate(questions, start=1):
        story.append(Paragraph(f"{i}. {q}", styles["BodyText"]))
        story.append(Spacer(1, 8))

    doc = SimpleDocTemplate(str(pdf_path), pagesize=LETTER)
    doc.build(story)
    return pdf_path


def _guess_smtp_host(user_email: Optional[str]) -> Optional[str]:
    if not user_email:
        return None
    if user_email.lower().endswith("@gmail.com"):
        return "smtp.gmail.com"
    if user_email.lower().endswith("@outlook.com") or user_email.lower().endswith("@hotmail.com"):
        return "smtp.office365.com"
    return None


def try_send_email(pdf_path: Path, to_email: Optional[str], subject: str, body: str) -> str:
    if not to_email:
        return "Email disabled."

    smtp_host = os.getenv("SMTP_HOST") or _guess_smtp_host(os.getenv("SMTP_USER"))
    smtp_port = int(os.getenv("SMTP_PORT", "587"))
    smtp_user = os.getenv("SMTP_USER")
    smtp_pass = os.getenv("SMTP_PASS") or os.getenv("SMTP_PASSWORD")
    from_email = os.getenv("FROM_EMAIL", smtp_user or "")

    if not (smtp_host and smtp_user and smtp_pass and from_email):
        return "Email not sent (missing SMTP env vars)."

    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = from_email
    msg["To"] = to_email
    msg.set_content(body)

    msg.add_attachment(pdf_path.read_bytes(), maintype="application", subtype="pdf", filename=pdf_path.name)

    with smtplib.SMTP(smtp_host, smtp_port) as server:
        server.starttls()
        server.login(smtp_user, smtp_pass)
        server.send_message(msg)

    return "Email sent with PDF attachment."


@app.get("/")
def serve_index() -> FileResponse:
    index_file = FRONTEND_DIR / "index.html"
    if index_file.exists():
        return FileResponse(str(index_file))
    return FileResponse(str((BASE_DIR / "README.md")))


@app.post("/workflow/run")
async def run_workflow(
    resume: UploadFile = File(...),
    jd: UploadFile = File(...),
    interview_rounds: str = Form(...),
    answer_length: str = Form("answer_medium"),
    company: Optional[str] = Form(None),
    role: Optional[str] = Form(None),
    send_email: str = Form("false"),
    to_email: Optional[str] = Form(None),
):
    run_id = datetime.utcnow().strftime("%Y%m%d%H%M%S") + "-" + uuid.uuid4().hex[:6]

    # Save uploads for traceability
    resume_path = OUTPUT_DIR / f"{run_id}_resume_{resume.filename}"
    jd_path = OUTPUT_DIR / f"{run_id}_jd_{jd.filename}"
    resume_path.write_bytes(await resume.read())
    jd_path.write_bytes(await jd.read())

    try:
        # Agent 1: parse resume
        agent1_out = run_agent1(str(resume_path))
        agent1_json_path = OUTPUT_DIR / f"{run_id}_agent1.json"
        agent1_json_path.write_text(json.dumps(agent1_out, ensure_ascii=False, indent=2), encoding="utf-8")

        # Agent 2: company research (uses EXA + Google)
        agent2_out = run_agent2(agent1_out, company_override=company, role_override=role)
        agent2_json_path = OUTPUT_DIR / f"{run_id}_agent2.json"
        agent2_json_path.write_text(json.dumps(agent2_out, ensure_ascii=False, indent=2), encoding="utf-8")

        # Agent 3: Q&A generation
        agent3_out = run_agent3(
            agent1_data=agent1_out,
            agent2_data=agent2_out,
            agent1_path=str(agent1_json_path),
            agent2_path=str(agent2_json_path),
            interview_rounds=interview_rounds,
            answer_length=answer_length,
        )
    except Exception as e:
        return JSONResponse({"detail": f"Workflow failed: {e}"}, status_code=500)

    top_20_questions = agent3_out.get("top_20_questions") or []

    text_filename = f"top_questions_{run_id}.txt"
    text_path = OUTPUT_DIR / text_filename
    text_path.write_text("\n".join(f"{i+1}. {q}" for i, q in enumerate(top_20_questions)), encoding="utf-8")

    meta = {
        "answer_length": answer_length,
        "interview_rounds": interview_rounds,
        "to_email": to_email,
    }

    pdf_path = build_pdf(
        run_id=run_id,
        title="Interview Preparation - Top Questions",
        questions=top_20_questions,
        meta=meta,
    )

    email_status = try_send_email(
        pdf_path=pdf_path,
        to_email=to_email if send_email == "true" else None,
        subject="Your Interview Prep Questions",
        body="Attached is your interview prep PDF. Good luck!",
    )

    payload: Dict[str, Any] = {
        "run_id": run_id,
        "candidate_name": Path(resume.filename).stem if resume.filename else "-",
        "company": company or "-",
        "role": role or "-",
        "message": email_status,
        "pdf_download_url": f"/outputs/{pdf_path.name}",
        "outputs": {
            "top_questions_txt": f"/outputs/{text_filename}",
            "agent1_json": f"/outputs/{agent1_json_path.name}",
            "agent2_json": f"/outputs/{agent2_json_path.name}",
        },
        "agent3_output": {
            "top_20_questions": top_20_questions,
            "top_30": agent3_out.get("top_30", []),
        },
        "meta": meta,
    }

    RUN_STORE[run_id] = payload
    return JSONResponse(payload)


@app.get("/runs/{run_id}")
def get_run(run_id: str):
    run = RUN_STORE.get(run_id)
    if not run:
        return JSONResponse({"detail": "Run not found."}, status_code=404)
    return JSONResponse(run)
