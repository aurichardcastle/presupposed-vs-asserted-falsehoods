# Presupposed vs Asserted Falsehoods: Measuring Premise Sycophancy in GPT-4o, Claude Sonnet 4.5, and Gemini 2.0 Flash

Code and data for the paper investigating whether frontier LLMs exhibit higher rates of sycophantic acceptance when a false factual premise is syntactically presupposed within a question than when the same premise is stated as a declarative assertion.

**Key finding:** Presupposition framing roughly doubles false-premise acceptance (24.7% vs 12.2%, p < .001), with the effect largest for GPT-4o (37.5% vs 16.7%) and smallest for Claude Sonnet 4.5 (5.8% vs 0.0%).

## Overview

The study bridges three literatures:
- **AI alignment:** Sycophancy as a documented artifact of RLHF training (Perez et al., 2023; Sharma et al., 2024)
- **Linguistics:** Presupposition and accommodation theory (Strawson, 1950; Karttunen, 1973; Stalnaker, 1974; Lewis, 1979)
- **Automated evaluation:** LLM-as-judge methodology (Zheng et al., 2023)

## Design

Two sub-studies, each with 40 history-domain items:

- **Study 1:** Well-known myths (e.g., "Viking helmets had horns"). Three conditions: presupposed-false (A), asserted-false (B), presupposed-true control (C). Result: near-ceiling correction across all models regardless of framing (floor effect).
- **Study 2:** Nuanced/contested historical claims. Two conditions: presupposed-false (A), asserted-false (B). Result: significant presupposition effect (p < .001).

Each prompt sent to 3 models x 3 runs = 1,800 expected responses (1,793 scored). Scored on a 4-point ordinal sycophancy rubric by a blinded LLM judge (Claude Sonnet 4.6) with cross-validation by GPT-4o (80.4% exact agreement, Spearman rho = .774).

## Pipeline

```
items.json / items_nuanced.json    Stimulus sets (40 items each)
         |
   collect.py / collect_nuanced.py  API calls via OpenRouter (3 models x 3 runs)
         |
   responses.csv / responses_nuanced.csv
         |
     score.py / judge.py            Blinded LLM judge scoring (4-point rubric)
         |
   scored_responses.csv / scored_nuanced.csv
         |
   cross_validate_judge.py          Second judge (GPT-4o) on 199-response subset
         |
   cross_validated_scores.csv
         |
     analyze.py                     Statistical analysis (chi-square, Kruskal-Wallis,
                                    Wilcoxon, Mann-Whitney U, correction timing)
```

## Parameters

| Parameter | Value |
|-----------|-------|
| Models | GPT-4o, Claude Sonnet 4.5, Gemini 2.0 Flash |
| Temperature | 0 |
| Max tokens | 1,200 |
| Runs per prompt | 3 |
| API gateway | OpenRouter |
| Primary judge | Claude Sonnet 4.6 (held out from study sample) |
| Cross-validation judge | GPT-4o |
| Collection window | April 2026 |

## Results Summary

| Metric | Value |
|--------|-------|
| Overall acceptance (presupposed) | 24.7% |
| Overall acceptance (asserted) | 12.2% |
| Chi-square | 17.855, p < .001 |
| Definite descriptions gap | +27.3 pp |
| Wh-presuppositions gap | +13.6 pp |
| Factive/manner gap | +3.9 pp |
| Trigger type (Kruskal-Wallis) | H = 15.131, p < .001 |
| False-content-first rate | 66% vs 54%, p = .003 |

## Running

```bash
# Set your OpenRouter API key
export OPENROUTER_API_KEY=your_key_here

# Collect responses (Study 1)
python collect.py

# Collect responses (Study 2)
python collect_nuanced.py

# Score responses
python score.py

# Cross-validate
python cross_validate_judge.py

# Analyze
python analyze.py
```

## Citation

Hardcastle, A. (2026). Presupposed vs Asserted Falsehoods: Measuring Premise Sycophancy in GPT-4o, Claude Sonnet 4.5, and Gemini 2.0 Flash. Singapore American School, Quest: AT English.

## Author

Auric Hardcastle — [LinkedIn](https://linkedin.com/in/auric-hardcastle)
