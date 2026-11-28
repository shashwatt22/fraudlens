"""Chronological replay of audited, strictly prior features for the frozen V3 model."""

from collections import deque
from dataclasses import dataclass
import math
from uuid import uuid4

import numpy as np
import pandas as pd
import streamlit as st

from src.data_loader import AMOUNT, ID, TIME
from . import explanations, store
from .store import ROWS, THRESHOLD, DataUnavailable

TICK_SECONDS = 1.0
SPEEDS = (1, 5, 20, 50, 100)
FEED_LIMIT, QUEUE_LIMIT = 30, 30
REPLAY_KEY = "v3_replay"


@dataclass(frozen=True, slots=True)
class Transaction:
    tx_id: int
    elapsed_second: int
    amount: float | None


@dataclass(frozen=True, slots=True)
class ScoredTransaction:
    sequence: int
    transaction: Transaction
    score: float

    @property
    def flagged(self):
        return self.score >= THRESHOLD


@dataclass(frozen=True, slots=True)
class AlertEvent:
    """A local historical-review event, with an idempotency key and no fraud label."""

    event_id: str
    replay_id: str
    sequence: int
    transaction_id: int
    elapsed_second: int
    score: float
    threshold: float = THRESHOLD
    source: str = "fraudlens.v3.historical_replay"


class Replay:
    """A single-session cursor. One successful tick commits at most one bounded batch."""

    def __init__(self, total=ROWS):
        if total <= 0:
            raise ValueError("Replay needs a non-empty source.")
        self.total = total
        self.reset()

    def reset(self):
        self.replay_id = uuid4().hex
        self.processed = self.flagged = 0
        self.status = "Ready"
        self.next_tick = None
        self.recent = deque(maxlen=FEED_LIMIT)
        self.review = deque(maxlen=QUEUE_LIMIT)

    def start(self, now):
        if self.status in ("Ready", "Paused"):
            self.status = "Running"
            self.next_tick = now

    def pause(self):
        if self.status == "Running":
            self.status = "Paused"
            self.next_tick = None

    def tick(self, now, speed, score_batch):
        if self.status != "Running" or now < self.next_tick:
            return ()
        if speed not in SPEEDS:
            raise ValueError("Unsupported replay speed.")
        events = self._advance(speed, score_batch)
        # Schedule from this tick. Delayed clients never trigger a catch-up burst.
        if self.status == "Running":
            self.next_tick = now + TICK_SECONDS
        return events

    def step(self, score_batch):
        if self.status in ("Ready", "Paused"):
            events = self._advance(1, score_batch)
            if self.status != "Complete":
                self.status = "Paused"
            return events
        return ()

    def _advance(self, count, score_batch):
        start, stop = self.processed, min(self.processed + count, self.total)
        batch = tuple(score_batch(start, stop))
        if len(batch) != stop - start:
            raise DataUnavailable("Inference returned the wrong batch length.")
        rows, events = [], []
        last_second = self.recent[-1].transaction.elapsed_second if self.recent else -1
        for sequence, (transaction, score) in enumerate(batch, start + 1):
            score = float(score)
            if transaction.elapsed_second < last_second or not math.isfinite(score) or not 0 <= score <= 1:
                raise DataUnavailable("Replay batch order or risk scores could not be verified.")
            last_second = transaction.elapsed_second
            row = ScoredTransaction(sequence, transaction, score)
            rows.append(row)
            if row.flagged:
                events.append(AlertEvent(f"{self.replay_id}:{sequence}", self.replay_id, sequence, transaction.tx_id,
                                         transaction.elapsed_second, score))
        # Only publish the batch once every row has passed validation.
        self.recent.extend(rows)
        self.review.extend(events)
        self.processed = stop
        self.flagged += len(events)
        if stop == self.total:
            self.status = "Complete"
            self.next_tick = None
        return tuple(events)


@st.cache_resource(show_spinner=False, max_entries=1)
def _metadata(partition, timeline, versions, first, last):
    frame = pd.read_parquet(partition, columns=[ID, TIME, AMOUNT])
    order = pd.read_parquet(timeline, columns=[ID, TIME]).iloc[-ROWS:].reset_index(drop=True)
    if (len(frame) != ROWS or not frame[ID].is_unique or frame[[ID, TIME]].isna().any().any()
            or not frame[[ID, TIME]].equals(order) or not frame[TIME].is_monotonic_increasing
            or frame[TIME].iloc[0] != first or frame[TIME].iloc[-1] != last):
        raise DataUnavailable("Replay rows do not match the frozen feature chronology.")
    return frame


def metadata(paths):
    """Open the verified order and original amount without reading outcomes or scores."""
    partition = store.source_partition(paths, "test")
    timeline = paths.retrospective / "covariates" / "development_metadata.parquet"
    audit = store.score_manifest(paths)
    manifest = store.source_manifest(paths)
    if audit["v1_manifest_sha256"] != store.V1_SHA:
        raise DataUnavailable("Replay requires the original holdout lineage.")
    window = manifest["partitions"]["test"]
    versions = (store.signature(partition), store.signature(timeline))
    return _metadata(partition, timeline, versions, window["first_second"], window["last_second"])


def score_batch(paths, start, stop):
    if not 0 <= start < stop <= ROWS or stop - start > max(SPEEDS):
        raise ValueError("The replay batch must be a bounded holdout slice.")
    frame = metadata(paths).iloc[start:stop]
    scores = explanations.explainer(paths).score(start, stop)
    if scores.shape != (stop - start,) or not np.isfinite(scores).all():
        raise DataUnavailable("The frozen model did not return one finite score per transaction.")
    rows = (Transaction(int(tx_id), int(second), float(amount) if pd.notna(amount) else None)
            for tx_id, second, amount in frame.itertuples(index=False, name=None))
    return tuple(zip(rows, scores, strict=True))


def pause_session():
    if (state := st.session_state.get(REPLAY_KEY)) is not None:
        state.pause()
