# AI use log (Assignment 2)

Tool: Claude (Anthropic), used through Claude Code with access to the assignment folder.
Date: 22 November 2026.

## What the AI did

- Read the brief, rubric and data, and checked the course AI rule (own first draft of the report
  before any AI editing; disclose AI use).
- Wrote and ran the analysis code (`analysis.py` / `analysis.ipynb`): data checks and cleaning, PV and
  price forecast models, the LP battery optimiser, the 48 h rolling-horizon scheduler, the sensitivities and figures.
- Produced a structured outline with the key numbers (`A2_写作提纲_outline.md`). It did **not** write the
  report prose. The report text is written by the student.

## What the student did

- Supplied the data, brief and rubric, and set the scope.
- Downloaded the official AEMO 2024 VIC1 price files (the AI environment could not reach AEMO).
- **To complete before submission:** checked the assumptions (tariff adders, battery model, capex, rebate),
  re-ran the code, verified the numbers below by hand, and wrote the report.

## Original request (excerpt)

> 原文: “根据要求帮我完成assignment 2，弄个初稿出来”
> Translation: “Complete Assignment 2 according to the requirements and produce a first draft.”

AI response on scope: because the course outline requires the student's own first draft, the AI limited
itself to the analysis, figures, outline and this log.

## AI errors found and corrected during the work (evidence of checking)

| # | Issue | How it was found | Fix |
|---|---|---|---|
| 1 | The time-zone check first used the single hour of peak irradiance (13:00), which cannot separate AEST from daylight-saving time. | Reviewing the logic against solar-noon physics | Replaced with the monthly irradiance-weighted centroid. It stays at 12:30–13:00 all year with no 1-hour jump, so the data has no DST. |
| 2 | The run crashed when saving the summary (tuple dictionary keys in JSON). | Traceback | Keys converted to strings. |
| 3 | Parallel runs used `multiprocessing.Pool`, which fails in a Jupyter notebook on macOS (the "spawn" start method). | Code review for portability to the student's Mac | Added `pmap()` with the "fork" start method and a serial fallback (`N_JOBS=1`). |
| 4 | The first forecast-based scheduler lost most of the wholesale value. | Comparing the rolling-horizon result with perfect foresight (Fig 8) | Diagnosed by price band (`value_by_price_band.csv`). Added a 5-minute real-time override that uses the last settled price, a refinement allowed by the brief's information rules. |
| 5 | The PV model looked "better" in 2024 (R² 0.996) than in training (0.965). | Sanity check of the metrics | The 2024 target is the commercial *forecast*, itself a smooth model of the same weather. The 2024 error therefore measures agreement between models, not real-world accuracy. This is stated as a limitation, and a persistence forecast is added as a weather-free stress test. |
| 6 | No 2024 spot price was in the CSV, and AEMO could not be reached from the AI environment. | Data audit | Used a clearly labelled placeholder for development only. The final numbers use the official AEMO files supplied by the student (see `summary.json -> price_source`). |
| 7 | “Grid charging” was first reported from the optimiser's c_grid variable, which showed 1,577 kWh of grid charging under the flat tariff. That is impossible economically. | Sanity check against theory: with a flat tariff, grid charging never pays. | For an AC-coupled battery the split between c_pv and c_grid is arbitrary, because both have the same efficiency. The metric now counts charging only while the site is importing. Flat tariff: 0 kWh, ToU: 129 kWh. |

## Automated checks

`results/validation_log.txt` lists the assertions that passed in the executed run. They cover a regular 5-min grid,
no DST, gap filling changing mean load by less than 1%, energy balance closure for every scenario, SoC
within limits, and rule-based self-consumption matching the LP optimum.

## Author clarification and Codex report revision on 22 November 2026

The earlier statement that report prose was written by the student is superseded by the author's
clarification: the author provided the outline and ideas, and OpenAI Codex generated the initial
report prose. Claude's recorded contribution was analysis code, figures and an outline.

The author confirmed rerunning the analysis locally. Codex then revised the report using the
supplied review comments, added the AI disclosure, reorganised sections and figures, checked
official sources and performed an independent recalculation of the saved interval settlement.
The recalculation agrees within AUD 0.02, allowing for four-decimal audit rounding; no fresh
optimisation run was performed during revision.

Codex identified the baseline wholesale-fee inconsistency and reconciled report bills by adding
AUD 300 to wholesale NoBESS and SelfCons in both years. Original analysis outputs are unchanged.
The report discloses modelling limits including simultaneous charge/discharge, training targets
that cross into 2024 and rolling terminal-energy treatment. These were not fixed in analysis.py.

Codex also prepared the repository documentation and organised the public repository under
the single Assignment2_Analysis_Package folder. The report records the billing reconciliation.
