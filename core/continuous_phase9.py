"""Phase 09: model-agnostic token emission from Phase08 decisions.

Each update represents one temporal inference window. Inference must cover
roughly successive camera windows; use timestamps in seconds, not frame counters.
"""
from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
from typing import Optional
import math


class State(str, Enum):
    IDLE = 'IDLE'
    CANDIDATE = 'CANDIDATE'
    STABLE = 'STABLE'
    WAIT_RELEASE = 'WAIT_RELEASE'


@dataclass(frozen=True)
class Phase09Config:
    confidence_threshold: float = 0.85  # fallback only; prefer Phase08 calibrated threshold
    min_consecutive: int = 3
    release_consecutive: int = 3
    min_token_interval_s: float = 0.4
    max_gap_s: float = 1.5
    rest_label: str = 'REST'
    unknown_label: str = 'UNKNOWN'
    # A transition to UNKNOWN alone is not proof that the signer released a sign.
    unknown_counts_as_release: bool = False


@dataclass(frozen=True)
class TokenEvent:
    label: str
    timestamp_s: float
    confidence: float


class ContinuousRecognizer:
    """One event per confirmed sign. Same token needs explicit REST release.

    Decisions of REST are special (release), not emitted as words. UNKNOWN is
    rejected, and does not count as REST unless explicitly configured.
    """
    def __init__(self, cfg: Phase09Config):
        if cfg.min_consecutive < 1 or cfg.release_consecutive < 1:
            raise ValueError('Consecutive counts must be positive')
        if cfg.min_token_interval_s < 0 or cfg.max_gap_s <= 0:
            raise ValueError('Invalid timing values')
        self.cfg = cfg
        self.reset()

    def reset(self):
        self.state = State.IDLE
        self.candidate: Optional[str] = None
        self.candidate_count = 0
        self.release_count = 0
        self.last_timestamp: Optional[float] = None
        self.last_emit_timestamp = float('-inf')
        self.last_token: Optional[str] = None
        self.events: list[TokenEvent] = []

    def _classify(self, label: str, confidence: float) -> str:
        if label == self.cfg.rest_label:
            return 'release'
        if label == self.cfg.unknown_label:
            return 'release' if self.cfg.unknown_counts_as_release else 'unknown'
        if not math.isfinite(confidence) or confidence < self.cfg.confidence_threshold:
            return 'unknown'
        return 'sign'

    def update(self, label: str, confidence: float, timestamp_s: float) -> Optional[TokenEvent]:
        if not math.isfinite(timestamp_s) or not (0 <= confidence <= 1):
            raise ValueError('Invalid timestamp/confidence')
        if self.last_timestamp is not None:
            if timestamp_s < self.last_timestamp:
                raise ValueError('Timestamps must be monotonic')
            if timestamp_s - self.last_timestamp > self.cfg.max_gap_s:
                # Lost tracking / paused stream: do not silently release an emitted sign.
                self.candidate = None
                self.candidate_count = 0
                if self.state in (State.CANDIDATE, State.STABLE):
                    self.state = State.IDLE
                self.release_count = 0
        self.last_timestamp = timestamp_s
        kind = self._classify(label, confidence)

        if self.state == State.WAIT_RELEASE:
            if kind == 'release':
                self.release_count += 1
                if self.release_count >= self.cfg.release_consecutive:
                    self.state = State.IDLE
                    self.candidate = None
                    self.candidate_count = 0
                    self.release_count = 0
            else:
                self.release_count = 0
            return None

        if kind != 'sign':
            self.candidate = None
            self.candidate_count = 0
            self.state = State.IDLE
            return None

        if label == self.candidate:
            self.candidate_count += 1
        else:
            self.candidate = label
            self.candidate_count = 1
        self.state = State.CANDIDATE

        if self.candidate_count < self.cfg.min_consecutive:
            return None
        self.state = State.STABLE
        if timestamp_s - self.last_emit_timestamp < self.cfg.min_token_interval_s:
            return None
        event = TokenEvent(label=label, timestamp_s=timestamp_s, confidence=confidence)
        self.events.append(event)
        self.last_token = label
        self.last_emit_timestamp = timestamp_s
        self.state = State.WAIT_RELEASE
        self.release_count = 0
        return event

    def snapshot(self) -> dict:
        return {
            'state': self.state.value,
            'candidate': self.candidate,
            'candidate_count': self.candidate_count,
            'release_count': self.release_count,
            'tokens': [event.label for event in self.events],
            'last_token': self.last_token,
        }
