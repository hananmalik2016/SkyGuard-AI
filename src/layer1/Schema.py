from pydantic import BaseModel, Field
from datetime import datetime

class AWSObservation(BaseModel):
    station_id: str
    timestamp: datetime
    temperature: float = Field(ge=-50.0, le=55.0) 
    pressure: float = Field(ge=600.0, le=1050.0)
    humidity: float = Field(ge=0.0, le=100.0)