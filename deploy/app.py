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


@app.get("/")
def home():
    return {"health_check": "OK", "grading_model_version": __version__}

@app.post("/grade", response_model=GradeResponse)
def predict_grade(request: GradeRequest, model: Model = Depends(Model)):
    # question_pairs, scores = model.predict(request)
    return GradeResponse(
        model_answer_ids=[0, 1, 3],
        segmented_student_answer=[[0, 4], [4, 10], [11, 20]],
        scores=[0, 0, 1]
    )

class SegmentRequest(BaseModel):
    answer: str

class SegmentResponse(BaseModel):
    segements: List[List[int]]
    cls: List[int]

@app.post("/segment", response_model=SegmentResponse)
def segment(request: SegmentRequest):
    return SegmentResponse(
        segements=[[0,10], [11, 20]],
        cls=[0, 1]
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=80)
