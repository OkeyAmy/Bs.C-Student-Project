"""Pydantic request/response contracts for the prediction API."""

from typing import Literal

from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    """The 11 model features as a single student record.

    Only fields known BEFORE the predicted semester's results exist.
    """

    prior_tier: int = Field(ge=1, le=4, description="Current CGPA tier (1=lowest .. 4=highest actually observed).")
    registered_prev: int = Field(ge=1, le=15, description="Courses registered in the previous semester.")
    passed_prev: int = Field(ge=0, le=15, description="Courses passed in the previous semester.")
    carryover_prev: int = Field(ge=0, le=15, description="Courses carried over from the previous semester.")
    registered_curr: int = Field(ge=1, le=15, description="Courses registered in the current semester (known at start).")

    institution_name: str = Field(default="Unknown")
    faculty: str = Field(default="Unknown")
    qualification_type: str = Field(default="Unknown")
    programme_category: str = Field(default="Undergraduate")
    current_academic_level: str = Field(default="Unknown")
    avg_grade_prev: str = Field(default="Unknown", description="Average letter grade in the previous semester (A/B/C).")


class PredictResponse(BaseModel):
    """Result of a single prediction, expressed in CGPA/GPA language."""

    tier: int = Field(ge=0, le=4)
    cgpa_band: str
    class_name: str
    at_risk: bool
    at_risk_reason: list[str] = Field(default_factory=list)
    numeric_input: dict = Field(default_factory=dict)