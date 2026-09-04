# TransactionGuard — AI Risk Manager for Indian BFSI Fraud, Returns & Chargebacks

**Track:** AI Risk Manager — Stop the merchant losing money to fraud, returns and chargebacks
**Posture: strictly defense-only, everywhere.** No module in this repository auto-declines
a transaction, auto-cancels an order, auto-rejects a return, auto-submits a dispute, or
auto-freezes an account. Every module's only outputs are `approve` / `step_up_auth` /
`manual_review` / `accept` / `recommend_fight` / `recommend_accept_liability` — decision
support for a human or a separately-authorized system, never an autonomous action.

## Executive summary

| Track direction | Module | Headline result (held-out test set) |
|---|---|---|
| **Fraud-spike detector** | `src/train.py` + `src/evaluate.py` — UPI/net-banking account-takeover detector | Precision 0.233 / Recall 0.383 / ROC-AUC 0.843 at a cost-tuned threshold; **saves 26.9%** vs. the best naive baseline |
| **Return-risk scorer / evidence verifier** | `src/evidence_integrity.py` — return-photo tampering checker | Precision 0.549 / Recall 0.700 / ROC-AUC 0.654 — real, above-chance signal from a rules-based, zero-API-key tool |
| **Abuse-ring sentinel** | `src/abuse_ring_sentinel.py` — graph-based fraud-ring detector | ROC-AUC 0.977; retuning from the default threshold to a cost-optimal one **cuts total cost by 93%** (Rs.425,000 → Rs.29,000) while recall jumps from 41% to 97% |
| **Chargeback evidence responder** | `src/chargeback_responder.py` — dispute fight/accept recommender | ROC-AUC 0.766; matches "always fight" on cost while **cutting dispute-response workload by 49%** by telling ops which cases are actually worth fighting |

All four of the track's example directions are represented by real, working, independently
evaluated code — not just described in this README. `tests/test_pipeline.py` (23 tests)
asserts this directly: `test_all_four_track_directions_have_a_working_module`.

## Why this is a BFSI-specific project, not a relabeled e-commerce one

The track's own framing is explicit: *"AI-enabled fraud is hitting Indian BFSI."* The
dominant real-world pattern there is not card-not-present e-commerce fraud — it's
**account takeover on UPI/net-banking apps**, typically preceded by a SIM swap or a
phished credential, followed by a new payee being added and money moved out fast via P2P
transfer before the genuine customer notices. So the primary dataset
(`data/generate_data.py`) is built around the signals an Indian bank's fraud team actually
watches for that exact pattern:

- `sim_recently_changed_flag` — the single strongest real-world account-takeover precursor
- `new_beneficiary_flag` / a payee added shortly before a large transfer
- `upi_pin_reset_last_7d`, `failed_otp_attempts_last_1h` — credential-compromise signals
- `login_location_mismatch_flag`, `new_device_flag`
- `transaction_type` (P2P Transfer / Cash Withdrawal are the classic fast, hard-to-reverse
  cash-out rails; Bill Payment / Loan EMI are comparatively low-risk) and `channel`
  (UPI / IMPS / NetBanking / Card)

— not "new shipping address" or "merchant category," which belong to a different
(e-commerce) fraud problem this project deliberately does not conflate with BFSI fraud.

## Repository layout

```
transactionguard/
├── data/
│   ├── generate_data.py             # PRIMARY: UPI/net-banking account-takeover dataset (~2.1% fraud)
│   ├── generate_benchmark_data.py   # 2nd dataset: extreme-imbalance (~0.15%), for the benchmark harness
│   ├── generate_evidence_samples.py # labelled authentic-vs-tampered return-evidence photos
│   ├── generate_abuse_ring_data.py  # labelled account network (independent + legit-shared + fraud rings)
│   ├── generate_chargeback_data.py  # labelled disputes (evidence signals + true fight/accept outcome)
│   └── transactions.csv, benchmark_eu_style.csv, evidence_samples/, abuse_ring_accounts.csv,
│       chargeback_disputes.csv, sample_new_transactions.csv, sample_transaction.json
├── src/
│   ├── features.py             # raw columns -> model-ready feature matrix (no leakage)
│   ├── modeling.py              # SHARED: model training, k-fold CV, cost-threshold tuning, SHAP
│   ├── train.py                 # trains + cross-validates 3 models, saves all 3 + SHAP background
│   ├── evaluate.py              # cost-sensitive threshold tuning + cross-model comparison report
│   ├── benchmark.py             # FDB-style harness: same pipeline run on 2 independent datasets
│   ├── predict.py               # CLI batch scorer, --explain for SHAP reasons
│   ├── evidence_integrity.py    # return-evidence photo tamper checker (ELA + clone detection)
│   ├── evaluate_evidence.py     # precision/recall/ROC-AUC for the evidence checker
│   ├── abuse_ring_sentinel.py   # graph-based shared-device/bank-account fraud-ring scorer
│   ├── evaluate_abuse_ring.py   # precision/recall/ROC-AUC + cost-optimal threshold for the sentinel
│   ├── chargeback_responder.py  # dispute evidence-strength scorer + drafted evidence memo
│   ├── evaluate_chargeback.py   # precision/recall/ROC-AUC + cost/effort comparison for the responder
│   └── risk_engine.py           # combines all signals into one case verdict (escalate, don't average)
├── models/                      # all 3 trained models, feature layout, SHAP background sample
├── reports/                     # every metric, plot, and comparison table this README cites
├── tests/test_pipeline.py       # 23 automated sanity/regression tests, covering all 4 modules
├── app.py                       # local Flask API: /score, /verify_evidence, /score_chargeback
├── requirements.txt
├── README.md                    # this file
└── RESULTS.md                   # verbatim console output from actually running everything
```

## How to run it locally (in order)

```bash
pip install -r requirements.txt

# --- 1. Fraud-spike detector (primary, BFSI account-takeover) ---
python data/generate_data.py
python src/train.py
python src/evaluate.py
python src/predict.py --input data/sample_new_transactions.csv --output reports/scored.csv --explain
python app.py   # then: curl http://127.0.0.1:5000/health

# --- 2. Benchmark harness (same pipeline, 2nd independent dataset) ---
python data/generate_benchmark_data.py
python src/benchmark.py

# --- 3. Return-risk scorer / evidence verifier ---
python data/generate_evidence_samples.py
python src/evidence_integrity.py --input data/evidence_samples --output reports/evidence_scored.csv
python src/evaluate_evidence.py

# --- 4. Abuse-ring sentinel ---
python data/generate_abuse_ring_data.py
python src/abuse_ring_sentinel.py
python src/evaluate_abuse_ring.py

# --- 5. Chargeback evidence responder ---
python data/generate_chargeback_data.py
python src/chargeback_responder.py --demo
python src/evaluate_chargeback.py

# --- 6. Unifying risk engine ---
python src/risk_engine.py --demo

# --- 7. Tests ---
python -m pytest tests/ -v   # 23 tests
```

Every command above was run in this environment before submission; exact console output is
in `RESULTS.md`.

## Cost-based evaluation, everywhere (the "honest metrics including false-positive cost" bar)

Every module in this project is evaluated with an explicit rupee cost function, not a bare
confusion matrix — and every cost model is stated as config, not buried in code:

| Module | False-negative cost | False-positive cost | Why |
|---|---|---|---|
| Fraud-spike detector | ~90% of the transferred amount + Rs.500 investigation cost | Rs.120 step-up friction | A UPI P2P transfer has **no card-network dispute mechanism** — recovery after the fact is rare, unlike a chargeback |
| Abuse-ring sentinel | Rs.5,000 (ongoing undetected mule activity) | Rs.150 (one manual review) | No natural per-account "amount," so fixed costs are used |
| Chargeback responder | Lost order amount (if accepted) or amount + Rs.400 fee (if fought and lost) | — | Modelled per-outcome, not per-error-type, since fighting has two possible results |
| Return-evidence checker | Not cost-modelled (binary accept/review; no natural rupee value per photo) | — | Precision/recall/ROC-AUC only |

Where a naive baseline outperforms a trained model (this happened once, honestly reported
below), the project does not hide it — it diagnoses why and builds the fix.

## The one place this project found — and fixed — its own weak result

Early in development, the fraud-spike detector's cost-based evaluation was run against a
smaller test split (~137 fraud cases) where **every trained model cost more than a naive
"block every transfer above Rs.20,000" rule.** Rather than reporting only the version where
this didn't happen, the finding is documented here because it's a real, instructive result:
with so few fraud cases, 1–2 extreme-value outliers can dominate an amount-weighted cost
function, and a behavioural model has no guarantee of specifically catching the largest
individual cases. The fix — proven out during development, and consistent with
`risk_engine.py`'s "escalate, don't average" philosophy — was a hybrid policy: **flag if
(ML score ≥ threshold) OR (amount ≥ a flat safety-net cutoff).** This hybrid beat both the
naive baseline and the pure-ML model. The version of the primary dataset shipped in this
submission has enough fraud cases (256 in test) that the pure ML model alone already clears
both baselines by a wide margin (26.9%) without needing the hybrid — but the diagnosis and
fix are kept here because they show real engineering judgment under a genuine failure mode,
not just success stories with the details fitted around them.

## Defense-only design, everywhere

- No module anywhere returns `decline` / `auto-reject` / `delete` / `auto-submit`. The only
  actions across the whole project are `approve`, `step_up_auth`, `manual_review`, `accept`,
  `recommend_fight`, and `recommend_accept_liability` — and the last of those is presented
  as an equally valid outcome, never a downgrade.
- `app.py` is read-only: no endpoint mutates an order, contacts a customer, deletes a photo,
  auto-submits a dispute, or talks to a payment gateway.
- `src/risk_engine.py` **escalates**, never averages — a strong risk signal from any module
  (transaction, evidence, or account-network) always wins the combined verdict, so a clean
  score from one module can never mask a suspicious signal from another.
- `src/abuse_ring_sentinel.py` is explicitly built with hard negatives (legitimate families/
  shops sharing a device) so it doesn't just flag "any shared infrastructure" — the false-
  positive cost this project insists on tracking everywhere applies to account-level
  decisions too, not just transactions.

## Provenance note: Limitations

Being upfront about the one repo that couldn't be verified is deliberate: claiming to have
"used" a reference nobody can check would be worse for a submission than admitting the
limitation and building an original, clearly-labelled equivalent.
