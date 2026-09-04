# UAE Residence Document Reader

A self-contained project for one workflow only: a UAE resident presenting an
**Emirates ID**, a **passport** page and a **UAE driving licence**. There is no
GCC route, no tourist route and no country selector in the interface.

The extraction engine (`car_rental_document_reader/`) is a byte-identical copy
of the shared reader, so any fix made there can be copied across unchanged. Only
the interface and the launcher are specific to this project.

## Setup

Python 3.12 is the reference runtime.

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -e '.[web,local-ocr]'
```

The first run downloads the PP-OCRv5 models into `~/.paddleocr` and `~/.paddlex`,
which needs an internet connection once. Every run after that is offline.

## Run

```bash
.venv/bin/python run_uae_residence.py
```

The interface binds to `127.0.0.1:7871`, creates no public share URL and writes
its outputs to private temporary files. Set `UAE_RESIDENCE_PORT` to use another
port.

## What it reads

| Section | Fields |
| --- | --- |
| Personal Information | First name, last name, gender, date of birth, nationality |
| Emirates ID | Number, issue date, expiry date |
| Passport | Number, issued by, issue date, expiry date |
| UAE Driving Licence | Number, issued by, issue date, expiry date |

Each field carries a status: `Verified`, `High confidence`, `Edited`,
`Needs review`, `Conflicting` or `Not found`. Expiry dates are checked against
today and marked expired, expiring within 30 days, or valid.

The passport is optional. When no passport page is uploaded, that section is
marked *Not provided* and is left out of the review check.

## How the passport is read

The shared engine restricts the UAE Resident route to the Emirates ID and the
licence, and rejects a page whose issuing state is not the UAE — which is what a
resident's foreign passport is. Rather than change the engine, this project
reads in two steps:

1. The bundle is read as a UAE Resident. This produces the Emirates ID, the
   licence and the personal fields.
2. Any page that came back `UNKNOWN`, or was rejected for a foreign issuer, is
   re-read through the engine's own passport route. Only the `passport.*` fields
   are merged back, and the false rejection warning and error for that page are
   cleared.

Step 2 does not run when there is no such page, so a bundle of Emirates ID plus
licence costs nothing extra.

## Output

**Confirm and Continue** produces a flat JSON payload:

```
customer_type, first_name, last_name, gender, date_of_birth, nationality,
emirates_id_number, emirates_id_issue_date, emirates_id_expiry_date,
passport_number, passport_issued_by_country, passport_issue_date,
passport_expiry_date, driving_licence_number, driving_licence_issued_by,
driving_licence_issue_date, driving_licence_expiry_date,
passport_presented, field_status, manual_review_required, confirmed_by_user
```

Dates are displayed as `DD-MM-YYYY` and stored as `YYYY-MM-DD`. Identifiers stay
strings so a leading zero survives the round trip. A separate processing report
records document quality, warnings and per-field status counts.

## Limits

This tool is not identity verification, forgery detection, face matching,
biometric or liveness checking, sanctions screening, or government-database
verification. It reconciles independent evidence — MRZ checks, document layout,
barcode data, dates and cross-document identity fields — and sends unresolved
conflicts to manual review. OCR confidence alone is never treated as
verification.

Real customer documents should stay outside any repository.
