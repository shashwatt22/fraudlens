"""Minimal email contents and bounded, encrypted SMTP transport."""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from email.errors import HeaderParseError
from email.headerregistry import Address
from email.message import EmailMessage
from email.utils import formatdate
from html import escape
import os
import re
import smtplib
import ssl
from threading import BoundedSemaphore

MODEL = "FraudLens V3 · depth-10 XGBoost"
REVIEW = "Pending analyst review"
INVALID_CONFIG = "Set the SMTP host, port, TLS mode, sender, credentials and one authorized recipient."
SMTP_TIMEOUT = 10
_WORKERS = 2
_EXECUTOR = ThreadPoolExecutor(max_workers=_WORKERS, thread_name_prefix="fraudlens-smtp")
_SLOTS = BoundedSemaphore(_WORKERS)


@dataclass(frozen=True, slots=True)
class SMTPConfig:
    host: str
    port: int
    tls: str
    sender: str = field(repr=False)
    recipient: str = field(repr=False)
    username: str = field(repr=False)
    password: str = field(repr=False)


def _mailbox(value):
    # SMTPUTF8 is not required, and exactly one bare mailbox is permitted.
    if not value.isascii() or any(c.isspace() for c in value):
        raise ValueError(INVALID_CONFIG)
    address = Address(addr_spec=value)
    if not address.username or not address.domain or address.addr_spec != value:
        raise ValueError(INVALID_CONFIG)
    return address.addr_spec


def configuration(env=None):
    env = os.environ if env is None else env
    if env.get("FRAUDLENS_SMTP_ENABLED") != "1":
        return None, "Server SMTP opt-in is off."
    prefix = "FRAUDLENS_SMTP_"
    try:
        host = env[prefix + "HOST"].strip()
        port = int(env[prefix + "PORT"])
        tls = env[prefix + "TLS"].strip().lower()
        sender = _mailbox(env[prefix + "FROM"])
        recipient = _mailbox(env[prefix + "TO"])
        username, password = env[prefix + "USERNAME"], env[prefix + "PASSWORD"]
        if (not re.fullmatch(r"[A-Za-z0-9.-]+", host) or not 1 <= port <= 65535 or tls not in ("ssl", "starttls")
                or not username or any(ch in username for ch in "\r\n") or not password):
            raise ValueError(INVALID_CONFIG)
    except (KeyError, ValueError, HeaderParseError):
        return None, INVALID_CONFIG
    return SMTPConfig(host, port, tls, sender, recipient, username, password), None


@dataclass(frozen=True, slots=True)
class MailNotice:
    """Only these fields can pass from the local replay to the email renderer."""

    reference: str
    score: float | None
    threshold: float
    is_test: bool = False
    model: str = MODEL
    source: str = "Historical replay"

    @property
    def review_status(self):
        return "Delivery test · no case created" if self.is_test else REVIEW


def _content(notice):
    return (
        ("Case reference", notice.reference),
        ("Risk score", f"{notice.score:.4%}" if notice.score is not None else "Not applicable · delivery test"),
        ("Decision threshold", f"{notice.threshold:.10f}"),
        ("Model", notice.model),
        ("Review status", notice.review_status),
    )


def message(config, notice):
    title = "Delivery test" if notice.is_test else f"{notice.source} review alert"
    intro = ("This test confirms that the configured SMTP relay can accept a FraudLens email."
             if notice.is_test else "The model flagged a transaction for analyst review.")
    note = f"{notice.source} · Use the independent case reference in the local FraudLens app."
    fields = _content(notice)
    email = EmailMessage()
    email["Subject"] = f"FraudLens | {title} | {notice.reference}"
    email["From"] = config.sender
    email["To"] = config.recipient
    email["Date"] = formatdate(usegmt=True)
    email["Message-ID"] = f"<{notice.reference.lower()}@fraudlens.invalid>"
    email.set_content(f"FraudLens | {title}\n\n{intro}\n\n" + "\n".join(f"{key}: {value}" for key, value in fields)
                      + f"\n\n{note}\n")
    rows = "".join(f'<tr><th scope="row" style="text-align:left;padding:10px;color:#3a5269">{escape(key)}</th>'
                   f'<td style="padding:10px;color:#112940">{escape(value)}</td></tr>' for key, value in fields)
    email.add_alternative(
        '<!doctype html><html lang="en"><head><meta charset="utf-8"></head>'
        '<body style="font-family:Arial,sans-serif;color:#112940;background:#f4f7fa;padding:24px">'
        '<main style="max-width:580px;margin:auto;background:#fff;border:1px solid #dce6ef;border-radius:12px;overflow:hidden">'
        '<div style="background:#112940;color:#fff;padding:24px"><div style="color:#a9ded3;font-size:13px">FraudLens V3</div>'
        f'<h1 style="font-size:24px;margin:8px 0 0">{escape(title)}</h1></div><div style="padding:24px">'
        f'<p>{escape(intro)}</p><table style="width:100%;border-collapse:collapse;background:#edf5f4">{rows}</table>'
        f'<p style="font-size:13px;color:#3a5269;margin:20px 0 0">{escape(note)}</p></div></main></body></html>',
        subtype="html",
    )
    return email


def deliver(config, notice):
    """Return a safe outcome. An accepted message is never retried on QUIT failure."""
    smtp = None
    try:
        context = ssl.create_default_context()
        if config.tls == "ssl":
            smtp = smtplib.SMTP_SSL(config.host, config.port, timeout=SMTP_TIMEOUT, context=context)
        else:
            smtp = smtplib.SMTP(config.host, config.port, timeout=SMTP_TIMEOUT)
            smtp.ehlo()
            smtp.starttls(context=context)
            smtp.ehlo()
        smtp.login(config.username, config.password)
        refused = smtp.send_message(message(config, notice), from_addr=config.sender, to_addrs=[config.recipient])
        return "Recipient declined by the SMTP server." if refused else None
    except smtplib.SMTPAuthenticationError:
        return "SMTP authentication failed."
    except (smtplib.SMTPRecipientsRefused, smtplib.SMTPSenderRefused):
        return "Sender or recipient declined by the SMTP server."
    except (ssl.SSLError, smtplib.SMTPNotSupportedError):
        return "Secure SMTP negotiation failed."
    except (TimeoutError, smtplib.SMTPServerDisconnected):
        return "SMTP disconnected or timed out. Acceptance could not be confirmed."
    except (OSError, smtplib.SMTPException, ValueError):
        return "SMTP delivery failed. Check the authorized account and relay settings."
    finally:
        if smtp is not None:
            try:
                # The send_message response is the acceptance boundary.
                smtp.close()
            except OSError:
                pass


def submit(config, notice):
    """Reserve a worker slot before submitting. Return immediately to Streamlit."""
    if not _SLOTS.acquire(blocking=False):
        return None
    try:
        task = _EXECUTOR.submit(deliver, config, notice)
    except RuntimeError:
        _SLOTS.release()
        return None
    task.add_done_callback(lambda _: _SLOTS.release())
    return task
