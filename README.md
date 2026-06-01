# Presupposed vs Asserted Falsehoods

### Measuring Premise Sycophancy in GPT-4o, Claude Sonnet 4.5, and Gemini 2.0 Flash

> Presupposition framing roughly doubles the rate at which frontier models accept and elaborate false premises on nuanced historical claims.

This repository contains the complete data collection, scoring, and analysis pipeline for the paper. All 1,793 scored responses and the stimulus sets are included for full reproducibility.

---

## Key Finding

When a false factual premise is **presupposed** inside a question rather than **asserted** as a declarative statement, frontier LLMs accept it at roughly double the rate:

| Condition | Acceptance Rate |
|-----------|----------------|
| Presupposed-false | **24.7%** |
| Asserted-false | 12.2% |
| | **p < .001** |

The effect is driven by the syntactic packaging of the claim, not its content — each item uses the same false claim in both conditions, differing only in grammatical framing.

### By Model

| Model | Presupposed | Asserted | p |
|-------|-------------|----------|---|
| GPT-4o | 37.5% | 16.7% | < .001 |
| Gemini 2.0 Flash | 30.8% | 20.0% | .075 |
| Claude Sonnet 4.5 | 5.8% | 0.0% | .021 |

### By Presupposition Trigger Type

| Trigger | Gap (pp) |
|---------|----------|
| Definite descriptions | +27.3 |
| Wh-presuppositions | +13.6 |
| Factive/manner constructions | +3.9 |
| | H = 15.131, **p < .001** |

### Secondary Finding

Even when models eventually correct a false premise, presupposed prompts cause them to **lead with the false content first** 66% of the time vs 54% for asserted prompts (p = .003). A correction buried after a confident wrong opener still misleads readers.

---

## Theoretical Framework

The study bridges two literatures that have proceeded largely in parallel:

- **AI alignment:** Sycophancy is a documented artifact of RLHF — models trained on human preference data systematically accommodate user-supplied falsehoods (Perez et al., 2023; Sharma et al., 2024).
- **Linguistic pragmatics:** Presupposed content is processed as accepted background, not as a new claim requiring evaluation (Strawson, 1950; Karttunen, 1973; Stalnaker, 1974). Lewis (1979) formalized "accommodation" — cooperative listeners add presupposed content to the common ground rather than challenge it.

No prior study had tested whether Lewis's accommodation framework predicts LLM sycophancy behavior. This study addresses that gap.

---

## Experimental Design

**Two sub-studies**, each with 40 history-domain items:

- **Study 1** — Well-known myths (e.g., "Viking helmets had horns"). Three conditions: presupposed-false, asserted-false, presupposed-true control. *Result:* Near-ceiling correction regardless of framing (floor effect). Models reliably catch obvious myths.

- **Study 2** — Nuanced/contested historical claims (e.g., causes of the American Civil War). Two conditions: presupposed-false, asserted-false. *Result:* Significant presupposition effect. The effect only emerges when claims are complex enough to exceed the models' confident correction threshold.

**Scoring:** 4-point ordinal sycophancy rubric (0 = explicit correction → 3 = elaborative acceptance). Primary judge: Claude Sonnet 4.6 (held out from study sample), blinded to model identity. Cross-validation: GPT-4o on 199-response subset (80.4% exact agreement, 100% within one point, Spearman ρ = .774).

**Parameters:** Temperature = 0 | Max tokens = 1,200 | 3 runs per prompt | API gateway: OpenRouter | Collection window: April 2026

---

## Repository Structure

```
├── items.json                    # Study 1 stimulus set (40 items × 3 conditions)
├── items_nuanced.json            # Study 2 stimulus set (40 items × 2 conditions)
├── collect.py                    # Study 1 data collection (OpenRouter API)
├── collect_nuanced.py            # Study 2 data collection
├── responses.csv                 # Study 1 raw responses (1,080 expected)
├── responses_nuanced.csv         # Study 2 raw responses (720 expected)
├── judge.py                      # Primary LLM judge (Claude Sonnet 4.6)
├── score.py                      # Manual blinded scoring interface
├── scored_responses.csv          # Study 1 scored data
├── scored_nuanced.csv            # Study 2 scored data
├── cross_validate_judge.py       # Cross-validation judge (GPT-4o)
├── cross_validated_scores.csv    # Cross-validation results (n=199)
├── compute_kappa.py              # Inter-rater agreement computation
└── analyze.py                    # Full statistical analysis + visualizations
```

### Pipeline

```
Stimulus design (items.json)
        │
        ▼
Data collection (collect.py)          3 models × 40 items × 3 conditions × 3 runs
        │
        ▼
Blinded LLM scoring (judge.py)       Claude Sonnet 4.6, model identity stripped
        │
        ▼
Cross-validation (cross_validate_judge.py)    GPT-4o on stratified 199-response subset
        │
        ▼
Statistical analysis (analyze.py)    Chi-square, Kruskal-Wallis, Wilcoxon,
                                     Mann-Whitney U, correction timing analysis
```

---

## Running

```bash
pip install numpy pandas scipy matplotlib seaborn requests

export OPENROUTER_API_KEY=your_key_here

python collect.py                # Study 1 (~1,080 API calls)
python collect_nuanced.py        # Study 2 (~720 API calls)
python judge.py                  # Score all responses
python cross_validate_judge.py   # Cross-validate with GPT-4o
python analyze.py                # Generate results + figures
```

The collection scripts support resume — they skip already-collected rows if interrupted.

---

## Citation

```
Hardcastle, A. (2026). Presupposed vs Asserted Falsehoods: Measuring Premise
Sycophancy in GPT-4o, Claude Sonnet 4.5, and Gemini 2.0 Flash. Singapore
American School, Quest: AT English.
```

## License

MIT

## Author

**Auric Hardcastle** — [LinkedIn](https://linkedin.com/in/auric-hardcastle) · [GitHub](https://github.com/AuricHardcastle)
