from typing import List
from fastapi import Depends, FastAPI
from pydantic import BaseModel
from model.sbert import Model, __version__


app = FastAPI()


class GradeRequest(BaseModel):
    model_answer: List[str]
    student_answer: List[str]
    max_grades: List[float]


class GradeResponse(BaseModel):
    question_pairs: List[List[str]]
    scores: List[float]
    

@app.get("/")
def home():
    return {"health_check": "OK", "grading_model_version": __version__}


@app.post("/grade", response_model=GradeResponse)
def predict_grade(request: GradeRequest, model: Model = Depends(Model)):
    question_pairs, scores = model.predict(request)
    return GradeResponse(question_pairs=question_pairs, scores=scores)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=80)
