from typing import List
from fastapi import Depends, FastAPI
from pydantic import BaseModel
from model.sbert import Model, __version__

app = FastAPI()

class GradeRequest(BaseModel):
    model_answer: List[str]
    model_answer_ids: List[int]
    student_answer: str
    max_grades: List[float]


class GradeResponse(BaseModel):
    model_answer_ids: List[int]
    segmented_student_answer: List[List[int]]
    scores: List[float]
    confidence: List[float]


@app.get("/")
def home():
    return {"health_check": "OK", "grading_model_version": __version__}

@app.post("/grade", response_model=GradeResponse)
def predict_grade(request: GradeRequest, model: Model = Depends(Model)):
    model_answer_ids, student_answer_indicies, confidence, scores= model.predict(request)
    return GradeResponse(
        model_answer_ids=model_answer_ids,
        segmented_student_answer=student_answer_indicies,
        scores=scores,
        confidence=confidence,
    )

class SegmentRequest(BaseModel):
    answer: str

class SegmentResponse(BaseModel):
    segments: List[List[int]]

@app.post("/segment", response_model=SegmentResponse)
def segment(request: SegmentRequest, model: Model = Depends(Model)):
    return SegmentResponse(segments=model.segment_text(request.answer))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=80)
