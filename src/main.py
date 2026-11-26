from contextlib import asynccontextmanager

from fastapi import FastAPI, status
from pydantic import BaseModel, ConfigDict, Field

from .data import FEATURES
from .predict import load_metrics, load_model, predict_data

# The first wine in the dataset (a cultivar_1 wine), shown as the example on the /docs page
EXAMPLE = {
    "alcohol": 14.23,
    "malic_acid": 1.71,
    "ash": 2.43,
    "alcalinity_of_ash": 15.6,
    "magnesium": 127.0,
    "total_phenols": 2.8,
    "flavanoids": 3.06,
    "nonflavanoid_phenols": 0.28,
    "proanthocyanins": 2.29,
    "color_intensity": 5.64,
    "hue": 1.04,
    "od280_od315_of_diluted_wines": 3.92,
    "proline": 1065.0,
}


class WineSample(BaseModel):
    """
    The 13 chemical measurements of one wine. Each limit is wider than anything in the
    dataset, so real wines pass and impossible values (like negative alcohol) get a 422.
    Unknown fields are rejected too, so a misspelled name isn't silently ignored.
    """
    model_config = ConfigDict(extra="forbid", json_schema_extra={"examples": [EXAMPLE]})

    alcohol: float = Field(ge=0, le=20)
    malic_acid: float = Field(ge=0, le=10)
    ash: float = Field(ge=0, le=5)
    alcalinity_of_ash: float = Field(ge=0, le=40)
    magnesium: float = Field(ge=0, le=250)
    total_phenols: float = Field(ge=0, le=6)
    flavanoids: float = Field(ge=0, le=8)
    nonflavanoid_phenols: float = Field(ge=0, le=1)
    proanthocyanins: float = Field(ge=0, le=5)
    color_intensity: float = Field(ge=0, le=20)
    hue: float = Field(ge=0, le=3)
    od280_od315_of_diluted_wines: float = Field(ge=0, le=6)
    proline: float = Field(ge=0, le=2500)


class Prediction(BaseModel):
    label: str
    probability: float
    probabilities: dict[str, float]


class BatchRequest(BaseModel):
    samples: list[WineSample] = Field(min_length=1, max_length=100)


class BatchResponse(BaseModel):
    predictions: list[Prediction]


ml = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the model and its metrics once, when the server starts, instead of on every request."""
    ml["model"] = load_model()
    ml["metrics"] = load_metrics()
    yield
    ml.clear()


app = FastAPI(title="Wine Cultivar Classifier", lifespan=lifespan)


@app.get("/", status_code=status.HTTP_200_OK)
def health_ping():
    return {"status": "healthy"}


@app.get("/model-info")
def model_info():
    """What model is being served, when it was trained, and how it scored on the test wines."""
    return {**ml["metrics"], "classes": [str(c) for c in ml["model"].classes_], "features": FEATURES}


# The prediction endpoints are plain `def`, not `async def`: prediction is CPU work, and
# FastAPI runs plain functions in a thread pool so they don't hold up other requests.
@app.post("/predict", response_model=Prediction)
def predict(sample: WineSample):
    """
    Predict the cultivar of one wine.
    Returns the most likely cultivar, its probability, and the probability of each cultivar.
    """
    return predict_data(ml["model"], [sample.model_dump()])[0]


@app.post("/predict/batch", response_model=BatchResponse)
def predict_batch(batch: BatchRequest):
    """Predict the cultivar of 1 to 100 wines in one request."""
    return {"predictions": predict_data(ml["model"], [s.model_dump() for s in batch.samples])}
