"""Bounded session alert history and delivery policy for historical replay."""

from collections import Counter, deque
from dataclasses import dataclass
from datetime import datetime, timezone
from time import monotonic
from uuid import uuid4

from . import alert_mail
from .store import THRESHOLD

DRY_RUN, DISABLED, REAL = "Dry run", "Disabled", "Real delivery"
HISTORY_LIMIT = 80
SESSION_LIMIT = 5
MIN_INTERVAL = 30


@dataclass(slots=True)
class Notification:
    notice: alert_mail.MailNotice
    created: datetime
    mode: str
    status: str
    detail: str
    updated: datetime | None = None


class AlertSession:
    """Only new ordered events for the current replay may become notifications."""

    def __init__(self):
        self.mode = DRY_RUN
        self.replay_id = None
        self.last_sequence = 0
        self.history = deque(maxlen=HISTORY_LIMIT)
        self.counts = Counter()
        self.created = self.attempts = 0
        self.last_attempt = None
        self.pending = {}
        self.last_assessment = 0

    def bind(self, replay_id):
        """Reset event watermark and remove the current replay's send authorization."""
        if replay_id != self.replay_id:
            replacing = self.replay_id is not None
            self.replay_id = replay_id
            self.last_sequence = 0
            if replacing and self.mode == REAL:
                self.dry_run()

    def dry_run(self):
        self.mode = DRY_RUN
        self.cancel_pending()

    def disable(self):
        self.mode = DISABLED
        self.cancel_pending()

    def activate(self, config):
        if config is None:
            return False
        self.mode = REAL
        return True

    def cancel_pending(self):
        for task, _ in self.pending.values():
            task.cancel()
        self.refresh()

    def collect(self, events, *, config=None, now=None):
        for event in events:
            if event.replay_id != self.replay_id or event.sequence <= self.last_sequence:
                continue
            self.last_sequence = event.sequence
            if (event.source == "fraudlens.v3.historical_replay" and event.threshold == THRESHOLD
                    and event.threshold <= event.score <= 1):
                reference = f"FL-{uuid4().hex[:16].upper()}"
                self._record(alert_mail.MailNotice(reference, event.score, THRESHOLD), config, now)

    def collect_assessment(self, sequence, reference, assessment, model, *, config=None, now=None):
        """Consume only a new submitted assessment. A widget rerun cannot resend it."""
        if sequence <= self.last_assessment:
            return
        self.last_assessment = sequence
        if assessment.flagged:
            notice = alert_mail.MailNotice(reference, assessment.score, assessment.threshold,
                                           model=model, source="Manual assessment")
            self._record(notice, config, now)

    def test(self, *, config=None, now=None, threshold=THRESHOLD, model=alert_mail.MODEL, source="Historical replay"):
        reference = f"TEST-{uuid4().hex[:16].upper()}"
        self._record(alert_mail.MailNotice(reference, None, threshold, is_test=True, model=model, source=source), config, now)

    def _new(self, notice, status, detail):
        record = Notification(notice, datetime.now(timezone.utc), self.mode, status, detail)
        self.created += 1
        self.counts[status] += 1
        self.history.append(record)
        return record

    def _record(self, notice, config, now):
        if self.mode == DISABLED:
            self._new(notice, "Disabled", "Notification suppressed by the session setting.")
            return
        if self.mode == DRY_RUN:
            self._new(notice, "Dry run", "Preview only. No SMTP connection.")
            return
        if config is None:
            self.dry_run()
            self._new(notice, "Dry run", "SMTP configuration unavailable. Activate real delivery again after setup.")
            return
        if self.attempts >= SESSION_LIMIT:
            self._new(notice, "Session limit", "Real-send attempt quota reached.")
            return
        now = monotonic() if now is None else now
        if self.last_attempt is not None and now - self.last_attempt < MIN_INTERVAL:
            self._new(notice, "Rate limited", f"One real attempt every {MIN_INTERVAL} seconds.")
            return
        task = alert_mail.submit(config, notice)
        if task is None:
            self._new(notice, "Transport busy", "SMTP workers are busy. This notification will not retry.")
            return
        self.attempts += 1
        self.last_attempt = now
        record = self._new(notice, "Sending", "Submitted to the SMTP worker.")
        self.pending[notice.reference] = task, record
        self.refresh()

    def refresh(self):
        for reference, (task, record) in tuple(self.pending.items()):
            if not task.done():
                continue
            del self.pending[reference]
            self.counts["Sending"] -= 1
            try:
                if task.cancelled():
                    status, detail = "Cancelled", "Delivery stopped before the SMTP worker started."
                elif error := task.result():
                    status, detail = "Failed", error
                else:
                    status, detail = "SMTP accepted", "The relay accepted the message. Inbox delivery depends on the provider."
            except Exception:
                status, detail = "Failed", "The delivery worker failed. Acceptance could not be confirmed."
            record.status, record.detail = status, detail
            record.updated = datetime.now(timezone.utc)
            self.counts[status] += 1
