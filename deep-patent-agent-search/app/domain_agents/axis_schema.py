"""Lightweight axis rating schema compatible with report assembler expectations."""

from __future__ import annotations

from pydantic import BaseModel, Field


class AxisBadges(BaseModel):
    coverage: str = Field(..., description="low|medium|high")
    confidence: str = Field(..., description="low|medium|high")
    evidence_direction: str = Field(..., description="favorable|mixed|unclear|adverse")


class AxisRating(BaseModel):
    section_id: str = Field(..., description="Machine-readable section id, e.g. mechanistic_plausibility")
    display_name: str = Field(..., description="Human-friendly name")
    grade: str = Field(..., description="Letter grade A-F or U")
    grade_modifier: str | None = Field(default=None, description="sparse|conflicting|liability|data_desert|none")
    summary: str = Field(..., description="2-4 sentence narrative")
    decision_hook: str | None = Field(default=None, description="One-sentence decision guidance")
    badges: AxisBadges = Field(..., description="Quality badges")
    pros: list[str] = Field(default_factory=list, description="Positive bullets")
    caveats: list[str] = Field(default_factory=list, description="Caveats/risks bullets")
    evidence: list[str] = Field(default_factory=list, description="Evidence citations or IDs")

