from datetime import date, datetime, timedelta
from typing import Literal

from pydantic import BaseModel, Field, model_validator

from .models import Annotation, Outbreak


class LoginIn(BaseModel):
    username: str = Field(min_length=1, max_length=64)
    password: str = Field(min_length=1, max_length=256)


class AnnotationIn(BaseModel):
    outbreak_id: int
    label: Literal["outbreak", "not_outbreak", "unsure", "skip"]
    confidence: int | None = Field(default=None, ge=1, le=5)
    comment: str | None = Field(default=None, max_length=5000)
    time_spent_ms: int | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def _confidence_matches_label(self):
        if self.label == "skip":
            self.confidence = None
        elif self.confidence is None:
            raise ValueError("confidence (1-5) is required unless the label is 'skip'")
        if self.comment is not None:
            self.comment = self.comment.strip() or None
        return self


class AnnotationOut(BaseModel):
    id: int
    outbreak_id: int
    label: str
    confidence: int | None
    comment: str | None
    created_at: datetime

    @classmethod
    def of(cls, a: Annotation) -> "AnnotationOut":
        return cls(id=a.id, outbreak_id=a.outbreak_id, label=a.label, confidence=a.confidence,
                   comment=a.comment, created_at=a.created_at)


class OutbreakOut(BaseModel):
    id: int
    disease: str
    location: str
    event: str
    start_date: date
    end_date: date
    duration: int
    total: float | None
    values: list[float | None]
    dates: list[date]
    my_annotation: AnnotationOut | None = None

    @classmethod
    def of(cls, o: Outbreak, mine: Annotation | None = None) -> "OutbreakOut":
        return cls(
            id=o.id, disease=o.disease, location=o.location, event=o.event,
            start_date=o.start_date, end_date=o.end_date, duration=o.duration, total=o.total,
            values=o.values, dates=[o.start_date + timedelta(weeks=i) for i in range(len(o.values))],
            my_annotation=AnnotationOut.of(mine) if mine else None,
        )
