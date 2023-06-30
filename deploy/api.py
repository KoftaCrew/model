from typing import List
from fastapi import Depends, FastAPI
from pydantic import BaseModel
from model.model import Model


app = FastAPI()


class GradeRequest(BaseModel):
    model_answer: List[str]
    student_answer: List[str]
    max_grades: List[float]


class GradeResponse(BaseModel):
    question_pairs: List[List[str]]
    scores: List[float]
    

@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.post("/predict", response_model=GradeResponse)
def predict(request: GradeRequest, model: Model = Depends(Model)):
    question_pairs, scores = model.predict(request)
    return GradeResponse(question_pairs=question_pairs, scores=scores)