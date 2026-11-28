"""Shared train/inference inputs for the optional-context manual challenger."""

from . import manual_model as manual

MODEL = "FraudLens Manual v2 · XGBoost"
EXTRA_FIELDS = ("DeviceType", "R_emaildomain")
EXTRA_LABELS = ("Device type", "Recipient email provider")
EXTRA_OPTIONS = (
    {"desktop": "Desktop or laptop", "mobile": "Mobile", manual.UNKNOWN: "Unknown"},
    manual.OPTIONS[-1],
)
FIELDS = (*manual.FIELDS, *EXTRA_FIELDS)
LABELS = (*manual.LABELS, *EXTRA_LABELS)


def normalize(raw):
    base = manual.normalize(raw)
    device, email = (raw[col] for col in EXTRA_FIELDS)
    base[EXTRA_FIELDS[0]] = device.where(device.isin(EXTRA_OPTIONS[0]), manual.UNKNOWN)
    base[EXTRA_FIELDS[1]] = email.map(manual.PROVIDERS).fillna(manual.OTHER).where(email.notna(), manual.UNKNOWN)
    return base


def validate(amount, network, funding, provider, device=manual.UNKNOWN, recipient=manual.UNKNOWN):
    base = manual.validate(amount, network, funding, provider)
    for key, value, options in zip(EXTRA_FIELDS, (device, recipient), EXTRA_OPTIONS, strict=True):
        if not isinstance(value, str) or value not in options:
            raise ValueError("Choose a supported value or Unknown for optional context.")
        base[key] = value
    return base


def selected(v3):
    """Keep v1 usable where the separately released v2 artifacts are absent."""
    promoted = v3 / "manual-challenger"
    return (promoted, True) if promoted.exists() else (v3 / "manual-assessment", False)
