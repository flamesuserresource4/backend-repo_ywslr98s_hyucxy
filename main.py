import os
import re
from collections import Counter, defaultdict
from typing import List, Optional, Dict, Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class GradeRequest(BaseModel):
    answer_key: str = Field(..., description="Answer key. Can be 'ABC...' or lines like '1) A' or '1A'.")
    submissions: str = Field(..., description="Multiple students' submissions. Blocks separated by blank lines. First non-empty line in each block is the student's name. Following lines contain answers like '1) a', '1a', '2 - B', etc.")


class MissedQuestion(BaseModel):
    question: int
    student_answer: Optional[str]
    correct_answer: str


class StudentReport(BaseModel):
    name: str
    raw_score: int
    total_questions: int
    percentage: float
    suggested_grade: str
    missed: List[MissedQuestion]


class AggregateSummary(BaseModel):
    total_students: int
    ranking: List[Dict[str, Any]]
    most_missed_questions: List[Dict[str, Any]]


class GradeResponse(BaseModel):
    reports: List[StudentReport]
    summary: AggregateSummary


GRADE_BANDS = [
    (100, "cel"),
    (90, "bdb"),
    (75, "db"),
    (50, "dst"),
    (30, "dop"),
    (0, "ndst"),
]


def normalize_token(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9]", "", s).strip()


def parse_answer_line(line: str) -> Optional[tuple[int, str]]:
    # Remove surrounding whitespace and ignore empty/comment-like lines
    raw = line.strip()
    if not raw:
        return None
    # Normalize to alnum only for simple patterns like "12A" or "1A"
    alnum = normalize_token(raw)
    m = re.match(r"^(\d+)([A-Za-z])$", alnum)
    if m:
        return int(m.group(1)), m.group(2).upper()
    # Try more flexible match with separators still in the raw string
    m2 = re.search(r"(\d+)\s*[\).:\-]*\s*([A-Za-z])", raw)
    if m2:
        return int(m2.group(1)), m2.group(2).upper()
    return None


def parse_key(answer_key: str) -> List[str]:
    lines = [l for l in answer_key.splitlines() if l.strip()]
    if not lines:
        # maybe compact string like "ABCACB"
        compact = normalize_token(answer_key)
        if not compact:
            raise ValueError("Answer key is empty")
        # if contains digits, treat as lines like 1A2B... We'll split into pairs number+letter
        if re.search(r"\d", compact):
            pairs = re.findall(r"(\d+)([A-Za-z])", compact)
            if not pairs:
                raise ValueError("Could not parse answer key")
            max_q = 0
            key_map: Dict[int, str] = {}
            for q, ans in pairs:
                qn = int(q)
                key_map[qn] = ans.upper()
                max_q = max(max_q, qn)
            return [key_map.get(i, None) or "" for i in range(1, max_q + 1)]
        else:
            return [ch.upper() for ch in compact if ch.isalpha()]
    # If lines include numbers, parse number->answer
    any_digit = any(re.search(r"\d", l) for l in lines)
    if any_digit:
        key_map: Dict[int, str] = {}
        max_q = 0
        for line in lines:
            parsed = parse_answer_line(line)
            if parsed:
                q, ans = parsed
                key_map[q] = ans
                max_q = max(max_q, q)
        if not key_map:
            raise ValueError("Could not parse answer key lines")
        return [key_map.get(i, None) or "" for i in range(1, max_q + 1)]
    # Else treat each line as a letter in order
    letters = []
    for l in lines:
        al = normalize_token(l)
        if not al:
            continue
        letters.append(al[0].upper())
    return letters


def split_blocks(text: str) -> List[List[str]]:
    blocks: List[List[str]] = []
    current: List[str] = []
    for line in text.splitlines():
        if line.strip() == "":
            if current:
                blocks.append(current)
                current = []
        else:
            current.append(line)
    if current:
        blocks.append(current)
    return blocks


def parse_submissions(submissions: str) -> List[tuple[str, Dict[int, str]]]:
    blocks = split_blocks(submissions)
    results: List[tuple[str, Dict[int, str]]] = []
    for block in blocks:
        # First non-empty line is name
        name = block[0].strip()
        answers: Dict[int, str] = {}
        for line in block[1:]:
            parsed = parse_answer_line(line)
            if parsed:
                q, ans = parsed
                answers[q] = ans
        # Also support compact lines like "ABC..." without numbers
        # If no numeric lines parsed, treat subsequent lines as concatenated letters
        if not answers:
            letters = [normalize_token(l) for l in block[1:] if normalize_token(l)]
            if letters:
                seq = "".join(letters)
                # keep only letters
                seq = "".join([ch for ch in seq if ch.isalpha()])
                for idx, ch in enumerate(seq, start=1):
                    answers[idx] = ch.upper()
        results.append((name, answers))
    return results


def grade_student(key: List[str], answers: Dict[int, str]) -> tuple[int, List[MissedQuestion]]:
    raw = 0
    missed: List[MissedQuestion] = []
    for i, correct in enumerate(key, start=1):
        stu = answers.get(i)
        if not correct:
            # No key for this question, skip scoring
            continue
        if stu and stu.upper() == correct.upper():
            raw += 1
        else:
            missed.append(MissedQuestion(question=i, student_answer=stu.upper() if stu else None, correct_answer=correct.upper()))
    return raw, missed


def suggest_grade(percentage: float) -> str:
    for threshold, label in GRADE_BANDS:
        if percentage >= threshold:
            return label
    return "ndst"


@app.get("/")
def read_root():
    return {"message": "Hello from FastAPI Backend!"}


@app.get("/api/hello")
def hello():
    return {"message": "Hello from the backend API!"}


@app.get("/test")
def test_database():
    """Test endpoint to check if database is available and accessible"""
    response = {
        "backend": "✅ Running",
        "database": "❌ Not Available",
        "database_url": None,
        "database_name": None,
        "connection_status": "Not Connected",
        "collections": []
    }

    try:
        # Try to import database module
        from database import db

        if db is not None:
            response["database"] = "✅ Available"
            response["database_url"] = "✅ Configured"
            response["database_name"] = db.name if hasattr(db, 'name') else "✅ Connected"
            response["connection_status"] = "Connected"

            # Try to list collections to verify connectivity
            try:
                collections = db.list_collection_names()
                response["collections"] = collections[:10]  # Show first 10 collections
                response["database"] = "✅ Connected & Working"
            except Exception as e:
                response["database"] = f"⚠️  Connected but Error: {str(e)[:50]}"
        else:
            response["database"] = "⚠️  Available but not initialized"

    except ImportError:
        response["database"] = "❌ Database module not found (run enable-database first)"
    except Exception as e:
        response["database"] = f"❌ Error: {str(e)[:50]}"

    # Check environment variables
    import os
    response["database_url"] = "✅ Set" if os.getenv("DATABASE_URL") else "❌ Not Set"
    response["database_name"] = "✅ Set" if os.getenv("DATABASE_NAME") else "❌ Not Set"

    return response


@app.post("/api/grade", response_model=GradeResponse)
def grade(req: GradeRequest):
    try:
        key_list = parse_key(req.answer_key)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid answer key: {str(e)}")

    students = parse_submissions(req.submissions)
    if not students:
        raise HTTPException(status_code=400, detail="No valid student submissions found")

    reports: List[StudentReport] = []
    miss_counter: Counter[int] = Counter()

    for name, answers in students:
        raw, missed = grade_student(key_list, answers)
        total = sum(1 for k in key_list if k)
        percentage = round((raw / total) * 100 if total else 0.0, 2)
        for m in missed:
            miss_counter[m.question] += 1
        report = StudentReport(
            name=name,
            raw_score=raw,
            total_questions=total,
            percentage=percentage,
            suggested_grade=suggest_grade(percentage),
            missed=missed,
        )
        reports.append(report)

    # Ranking
    ranking = sorted(
        (
            {"name": r.name, "raw_score": r.raw_score, "percentage": r.percentage}
            for r in reports
        ),
        key=lambda x: (-x["raw_score"], x["name"].lower()),
    )

    # Most missed questions
    if miss_counter:
        max_miss = max(miss_counter.values())
        most_missed = [
            {"question": q, "missed_by": c}
            for q, c in miss_counter.items() if c == max_miss
        ]
        most_missed.sort(key=lambda x: x["question"])  # stable order
    else:
        most_missed = []

    summary = AggregateSummary(
        total_students=len(reports),
        ranking=ranking,
        most_missed_questions=most_missed,
    )

    return GradeResponse(reports=reports, summary=summary)


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
