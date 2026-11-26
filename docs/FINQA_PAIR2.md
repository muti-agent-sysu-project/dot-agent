# FinQA-Priv model pair 2

Run from the repository root on Linux with the fixed FinQA-Priv public packet and grader files,
a local OpenAI-compatible Qwen/Qwen3-8B service on loopback port 8000, and FINQA_CLOUD_KEY_FILE
pointing to the existing mode-600 DeepSeek credential file. The launcher requires a fresh output
directory for a new wave. Physical attempts have no transport retries. Stages have a 90-minute
deadline. Spend is reserved at peak price before transmission against a shared USD 5.99 ceiling.
A missing receipt aborts the current batch. The dated USD price card is valid for Nov 26-29, 2026 UTC.

```bash
python -m scripts.finqa_pair2_wave "$DATASET" "$OUT"
python -m scripts.finqa_pair_reporting "$OUT" --pair2
python -m pytest tests -q
```

The probe, independent development flow and forced FinQA development actions precede calibration.
The published split has 32 development groups. The first 16 provide five forced paths, all 32 provide
the public B1 reference, and all 32 are evaluated under the five private policies. This dev evaluation
overlaps calibration groups. Only the 160-group confirm stage is held out. The dev flow contains
six independent five-task contracts. Each candidate's empirical paired discounted four-opportunity
savings include binding, drift and negative cases, using all scheduled rows; subtract the positive
coordinatewise first-encounter premium. At runtime the first encounter awards the positive part
of this net vector valued at the current dual prices. Its exposure quote is computed on the actual
ordered continuation envelopes. The historical B9 uses its original predictor and price update.

Closure-choke uses synthetic invoice workspaces and a six-department, four-quarter synthetic public
roster. Each singleton attribute costs zero, while the combination derives the protected identity
and exceeds the common cap of 0.25. A valid first response selects one of three lines for the second
query. If the second cloud query is denied, all arms have access to the local model. This tests
end-to-end quality including recovery. Eight independent dev amount draws fit first-step action
quotes. The 24 formal workspaces share the same settlement rule. Paired intervals describe variation
over those workspaces, not a population of 24 different financial algorithms. Cloud accounts and
prefix caches are physically shared; ledgers and skill libraries are logically isolated per arm.
The six-worker end-to-end wall measurements include queueing. The service price is calculated from
receipts, not an account invoice. GPU rental is separate.

## Measured results, November 27 (Asia/Shanghai)

Formal code 38e3f991fdd6d3a8fbbe93619edbb824e9ff52ac, profile SHA256
e967b2893a4e1cae602199472febcad2f3ddce08df8792c35bf181a58f4d95c2.

| Arm | Cold correct / 160 | Stream correct / 120 | Cold installations / 24 | Binding hits / 48 | ITT cloud tokens/task | ITT seconds/task |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| B0 | 64 | 79 | 0 | 0 | 0.0 | 0.902 |
| B1 | 98 | 107 | 0 | 0 | 411.7 | 1.605 |
| B6 | 96 | 120 | 0 | 0 | 345.2 | 2.183 |
| B8 | 80 | 120 | 24 | 48 | 68.6 | 0.477 |
| B9 v1 | 80 | 117 | 23 | 46 | 68.7 | 0.459 |
| B9 v2 | 80 | 119 | 23 | 47 | 71.7 | 0.464 |

Typed and minimal views yielded 95 and 94 successes over the same 160 snapshots. The paired
minimal-minus-typed utility difference was -0.00625 (95% CI -0.025 to 0.0125). On the stream,
the paired v2-minus-v1 utility difference was 0.0167 (95% family bootstrap CI -0.025 to 0.075).
First-encounter installation counts were identical for the two B9 policies.

The frozen closure protocol scored B6, B8 and B9 v2 at 0/24 with no violations. The B6 local
judge enabled all 24 second steps; the hard boundary blocked all 24 and local recovery scored
zero. For B8 and B9 v2 all 24 first outputs failed the integer-only binding contract. Their
completion difference is uninformative about the second-step mechanism. In the independent
closure development set both first-view format success rates were already zero. Future experiments
need a development criterion for reaching the second decision in each comparison arm.

Across development, probes and formal stages, 3,772 physical calls reconciled one-to-one,
including 2,585 cloud calls, zero unknown receipts and USD 0.694126268 in token charges at the
dated tariff. The elapsed probe-to-final-stream interval was 25.26 minutes. No GPU invoice is included.
