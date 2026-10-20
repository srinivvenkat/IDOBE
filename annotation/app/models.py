from datetime import date, datetime, timezone

from sqlalchemy import (
    JSON, Boolean, CheckConstraint, Date, DateTime, Float, ForeignKey, Index, Integer, String, Text,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship
from sqlalchemy.types import TypeDecorator

LABELS = ("outbreak", "not_outbreak", "unsure", "skip")
# Why a series was shown: drawn at random, picked because the model was unsure about it, reopened by the
# expert from their own history, or opened directly by link. "random" draws give an unbiased sample for
# measuring the model.
SERVED_AS = ("random", "uncertain", "revisit", "lookup")
VOTE_LABELS = ("outbreak", "not_outbreak", "unsure")  # labels that count toward coverage


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class UTCDateTime(TypeDecorator):
    """Timezone-aware UTC datetimes on every backend.

    SQLite drops the offset; Postgres returns values in the session time zone.
    """

    impl = DateTime(timezone=True)
    cache_ok = True

    def process_bind_param(self, value, dialect):
        if value is not None and value.tzinfo is None:
            raise ValueError("naive datetime; use timezone-aware UTC")
        return value.astimezone(timezone.utc) if value is not None else None

    def process_result_value(self, value, dialect):
        if value is None:
            return None
        return value.replace(tzinfo=timezone.utc) if value.tzinfo is None else value.astimezone(timezone.utc)


class Base(DeclarativeBase):
    pass


class User(Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(primary_key=True)
    username: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    password_hash: Mapped[str] = mapped_column(String(255))
    display_name: Mapped[str | None] = mapped_column(String(128))
    affiliation: Mapped[str | None] = mapped_column(String(255))
    expertise: Mapped[str | None] = mapped_column(String(255))
    is_admin: Mapped[bool] = mapped_column(Boolean, default=False)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime, default=utcnow)


class Outbreak(Base):
    __tablename__ = "outbreaks"

    # Same value as unique_id in raw_data/outbreaks_disease_location.csv.
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=False)
    disease: Mapped[str] = mapped_column(String(128), index=True)
    location: Mapped[str] = mapped_column(String(128), index=True)
    event: Mapped[str] = mapped_column(String(64))
    start_date: Mapped[date] = mapped_column(Date)
    end_date: Mapped[date] = mapped_column(Date)
    duration: Mapped[int] = mapped_column(Integer)
    total: Mapped[float | None] = mapped_column(Float)
    # Weekly values; value i is for the week ending start_date + 7*i. Trailing blanks removed.
    values: Mapped[list] = mapped_column(JSON)
    source_file: Mapped[str] = mapped_column(String(255))
    loaded_at: Mapped[datetime] = mapped_column(UTCDateTime, default=utcnow)


class Annotation(Base):
    """Append-only: an edit adds a new row. A user's current answer is their row with the highest id."""

    __tablename__ = "annotations"
    __table_args__ = (
        CheckConstraint("label IN ('outbreak','not_outbreak','unsure','skip')", name="ck_annotation_label"),
        CheckConstraint(
            "(label = 'skip' AND confidence IS NULL) OR (label <> 'skip' AND confidence BETWEEN 1 AND 5)",
            name="ck_annotation_confidence",
        ),
        Index("ix_annotations_outbreak_user_id", "outbreak_id", "user_id", "id"),
        Index("ix_annotations_user_id", "user_id", "id"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    outbreak_id: Mapped[int] = mapped_column(ForeignKey("outbreaks.id", ondelete="CASCADE"))
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"))
    label: Mapped[str] = mapped_column(String(16))
    confidence: Mapped[int | None] = mapped_column(Integer)
    comment: Mapped[str | None] = mapped_column(Text)
    time_spent_ms: Mapped[int | None] = mapped_column(Integer)
    client_info: Mapped[str | None] = mapped_column(String(512))
    app_version: Mapped[str | None] = mapped_column(String(32))
    served_as: Mapped[str | None] = mapped_column(String(16))
    created_at: Mapped[datetime] = mapped_column(UTCDateTime, default=utcnow)

    user: Mapped[User] = relationship()
    outbreak: Mapped[Outbreak] = relationship()


class Assignment(Base):
    """Log of every series served by the queue, and why."""

    __tablename__ = "assignments"
    __table_args__ = (Index("ix_assignments_user_outbreak", "user_id", "outbreak_id", "id"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"))
    outbreak_id: Mapped[int] = mapped_column(ForeignKey("outbreaks.id", ondelete="CASCADE"))
    reason: Mapped[str] = mapped_column(String(16))
    model_run_id: Mapped[int | None] = mapped_column(ForeignKey("model_runs.id", ondelete="SET NULL"))
    served_at: Mapped[datetime] = mapped_column(UTCDateTime, default=utcnow)


class ModelRun(Base):
    """One training run: what it was trained on and how well it did."""

    __tablename__ = "model_runs"

    id: Mapped[int] = mapped_column(primary_key=True)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime, default=utcnow)
    model_type: Mapped[str] = mapped_column(String(32))
    n_votes: Mapped[int] = mapped_column(Integer)          # expert Outbreak / Not outbreak answers used
    n_outbreaks: Mapped[int] = mapped_column(Integer)      # distinct series those answers cover
    train_seconds: Mapped[float] = mapped_column(Float)
    metrics: Mapped[dict] = mapped_column(JSON)            # cross-validation and random-sample evaluation


class ModelPrediction(Base):
    """The latest model's score for every series (replaced wholesale by each training run)."""

    __tablename__ = "model_predictions"
    __table_args__ = (Index("ix_model_predictions_uncertainty", "uncertainty"),)

    outbreak_id: Mapped[int] = mapped_column(ForeignKey("outbreaks.id", ondelete="CASCADE"), primary_key=True)
    run_id: Mapped[int] = mapped_column(ForeignKey("model_runs.id", ondelete="CASCADE"))
    p_outbreak: Mapped[float] = mapped_column(Float)
    uncertainty: Mapped[float] = mapped_column(Float)      # 1 at p = 0.5, 0 at p = 0 or 1
