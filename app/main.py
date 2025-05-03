from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from .recommender import SHLRecommender
from .models import RecommendationRequest, RecommendationResponse

app = FastAPI(title="SHL Assessment Recommender")

# CORS setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize recommender
recommender = SHLRecommender()

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

@app.post("/recommend", response_model=RecommendationResponse)
async def get_recommendations(request: RecommendationRequest):
    try:
        results = recommender.recommend(
            query=request.query,
            max_duration=request.max_duration
        )
        return {"recommendations": results}
    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail=f"Recommendation failed: {str(e)}"
        )
# In your main.py
@app.get("/debug")
async def debug_info():
    return {
        "assessment_count": len(recommender.df),
        "sample_assessments": recommender.df[['Name', 'Assessment Length (Minutes)']]
            .rename(columns={'Name': 'name', 'Assessment Length (Minutes)': 'duration'})
            .to_dict('records')[:5]
    }