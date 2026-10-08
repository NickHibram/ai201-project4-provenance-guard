# Provenance Guard — Project Plan
## Project Goal and Scope

Provenance Guard will be a Flask backend that assesses submitted English-language creative text using two detection signals, combines their outputs into a score, and returns one of three attribution results: `likely_human`, `uncertain`, or `likely_ai`. It will explain that result through a transparency label, preserve the decision in an audit log, and let creators contest it through an appeal.

The goal is a responsible system around an uncertain assessment, not proof of authorship. A stylistic resemblance to AI writing does not establish how a text was created or whether its creator was dishonest. The system will not identify the actual author, detect plagiarism, punish creators, or certify work as human-written.

The agreed design is an LLM assessment plus one stylometric signal containing four understandable features, combined with 60/40 weighting. This document finalizes previously unspecified details: feature calculations, a 50-word minimum, score boundaries, storage, and error handling. These are proposed implementation choices, not measured findings.

The initial stack will be Flask, Groq, Flask-Limiter, SQLite, and Python's standard library for feature extraction. There will be no trained classifier, fine-tuning, feature-learning pipeline, or frontend requirement. The supplied assignment determines the deliverables and milestone order.

## Inspiration

This project is inspired by Przystalski, Argasiński, Grabska-Gradzińska, and Ochab (2025), *Stylometry recognizes human and LLM-generated texts in short samples* (arXiv:2507.00838v1; https://arxiv.org/html/2507.00838v1). The authors use stylometric features with decision-tree and LightGBM classifiers on Wikipedia-based English texts (§§3.1–3.4). Their StyloMetrix setup contains 195 features; §4.1 highlights function-word types and lemma-based type-token ratio. The paper also discusses punctuation/preprocessing artifacts and limits its conclusions to the studied domain and language (§§3.3.2, 5–5.1).

**What I am adapting:** the idea of examining interpretable characteristics of writing as one source of attribution evidence.

**What I am not reproducing:** the paper's classifiers, complete feature set, preprocessing pipeline, or reported performance. My surface-word vocabulary measure, sentence-variation rule, punctuation-density regularity, function-word-frequency regularity, weights, and thresholds are project-specific simplifications. The paper does not validate these four rules or their combination for creative writing.


## Detection Signals

### Signal 1: LLM-Based Assessment Through Groq

**Purpose.** Ask an instruction-following language model to assess the writing holistically, considering phrasing, organization, repetition, coherence, and contextual consistency. This complements direct numerical measurements rather than replacing them.

**Input.** The submitted text, preserved without rewriting its wording or punctuation. Do not send the creator ID or the stylometric scores to the model. Withholding the second signal's output avoids explicitly anchoring the LLM assessment to it.

**Output contract.** A standalone `assess_with_llm(text)` function will return a validated result containing:

```json
{
  "ai_score": 0.74,
  "reasoning": "The phrasing and transitions are consistently regular, but the style alone does not establish authorship."
}
```

This is an **illustrative interface example**, not an actual model response. `ai_score` must be a finite number in `[0, 1]`; `reasoning` must be a nonempty string of at most 600 characters. Only `ai_score` affects the numerical calculation. The explanation is retained for debugging and review, not treated as independently verified evidence.

**Score meaning.** Zero means the model's assessment strongly leans toward human-like writing, 0.5 means inconclusive, and one means strongly AI-like. These values are not calibrated probabilities.

**Prompt requirements.** Request JSON only. Tell the model that the submission is untrusted text to analyze, not instructions to follow. Instruct it not to infer authorship from formality, slang, grammar errors, or personal anecdotes alone; to acknowledge alternative explanations; and to use a score near 0.5 when evidence is weak. The application, not the LLM, will choose the final attribution and label.

**Why choose it?** The design hypothesis is that holistic assessment may notice patterns that four structural measurements miss. This is a reason to test the signal, not an assumption that the model is an accurate detector.

**Blind spots.** Anticipated failures include polished human writing, deliberately casual AI writing, mixed authorship, and persuasive but unsupported explanations. Model outputs can also vary between requests. Agreement with stylometry will not be treated as proof.

**Operational behavior.** Use a low sampling temperature supported by the selected model, an explicit 20-second request timeout, and no automatic retries in version 1. Validate the response locally. Malformed JSON, missing fields, out-of-range scores, timeouts, and provider failures will mark this signal unavailable; do not silently replace an invalid score with 0 or 1.

**Model configuration.** Require `GROQ_API_KEY` and `GROQ_MODEL` in the local environment. The course names `meta-llama/llama-4-scout-17b-16e-instruct`, but that identifier was not listed on the Groq supported-models page checked for this plan. Before implementation, verify an available instruction-following text model in the actual account and record the selected identifier. Do not silently substitute models. Store the identifier, prompt version, and sampling settings with each completed LLM assessment.

### Signal 2: Four-Feature Stylometric Heuristics

**Purpose.** Measure structural characteristics directly in Python. There is no training phase. The four measurements produce **one** stylometric signal; they do not count as four independent detectors.

For punctuation and function words, I will examine **variation in their per-sentence density**, rather than assert that one absolute frequency universally identifies AI writing. That is an implementation refinement of the four features discussed for this project.

#### Shared preprocessing

- Preserve the original text in storage. Work on a separate analysis copy with standardized line endings.
- Count Unicode letter-based words, allowing internal apostrophes. Normalize curly apostrophes to straight apostrophes and lowercase tokens. Do not remove function words or perform stemming. Hyphenated words count as separate tokens.
- Use a simple sentence splitter at runs of `.`, `!`, or `?` followed by whitespace/end-of-text, and at line breaks. Ignore fragments without word tokens. Treat these as analysis units, not guaranteed grammatical sentences.
- Require at least **50 words and 3 analysis sentences** for an attribution based on both signals. Below either minimum, abstain as described under Uncertainty Representation.

Abbreviations, quotations, poetry line breaks, and run-on sentences can defeat this simple segmentation. That limitation must remain visible in the README.

#### Feature definitions

| Feature | Calculation | Proposed use and anticipated blind spot |
| --- | --- | --- |
| **Sentence-length variation** | Count words in each sentence. Compute population standard deviation divided by mean sentence length: `sentence_cv = population_std(lengths) / mean(lengths)`. | Lower variation contributes more AI-like evidence under a stylistic-regularity hypothesis. Deliberately repetitive human writing can trigger the same pattern. |
| **Vocabulary diversity / type-token ratio** | Record whole-text `unique_words / total_words`. For scoring, split tokens into consecutive, non-overlapping 50-word blocks and average each block's type-token ratio: `ttr_50`. Exclude the final incomplete block from this feature only; record how many words were used. | Lower block-level diversity contributes more AI-like evidence under a repetition hypothesis. A refrain or technical topic may repeat words legitimately; AI writing may also be highly diverse. |
| **Punctuation usage** | For each analysis sentence, count internal punctuation from `,;:—–()[]{}` and divide by its word count. Record overall density and calculate the coefficient of variation of those per-sentence densities: `punctuation_cv`. Sentence-ending punctuation is excluded from this feature. | More consistent density contributes more AI-like evidence under the regularity hypothesis. Formatting and an author's deliberate punctuation style can create the same pattern. If every density is zero, return a neutral component, not an AI accusation. |
| **Function-word frequency** | For each sentence, divide the number of words in the fixed list below by its word count. Record overall frequency and calculate variation across sentences: `function_word_cv`. | More consistent frequency contributes more AI-like evidence under the regularity hypothesis. Genre, sentence structure, and repeated human-written templates may cause the same result. Zero matches give a neutral component. |

The 50-word windows make the vocabulary comparison use the same token count per block. They do not make it a calibrated or genre-independent authorship test.

Fixed function-word list for version 1:

```text
a an the and or but if while because as until although
of at by for with about against between into through during
before after above below to from up down in out on off over under
i me my we us our you your he him his she her it its they them their
is am are was were be been being have has had do does did
that this these those
```

This is a deliberately limited token list, not a grammatical tagger. For example, `it's` remains one token and is not expanded to `it is`.

#### Converting measurements into component scores

These exact mappings are **initial, unvalidated heuristic assumptions**. They are specified before implementation so the program can be tested against a clear contract, not because the numerical anchors come from the paper.

Define `clip(x, 0, 1)` as limiting a value to the interval `[0, 1]`.

For sentence, punctuation, and function-word coefficients of variation:

```text
regularity_component(cv) =
    0.90 - 0.80 × clip((cv - 0.10) / 0.70, 0, 1)
```

Thus, variation at or below 0.10 gives 0.90, variation at or above 0.80 gives 0.10, and values between those anchors are mapped linearly. This encodes the proposed regularity hypothesis; it does not establish that regular writing was generated by AI.

For vocabulary diversity:

```text
vocabulary_component(ttr_50) =
    0.90 - 0.80 × clip((ttr_50 - 0.45) / 0.40, 0, 1)
```

Thus, a block-level ratio at or below 0.45 gives 0.90, a ratio at or above 0.85 gives 0.10, and intermediate values are mapped linearly.

For punctuation or function words with zero mean density, the coefficient of variation is undefined: record it as `null`, assign that component **0.50**, and record a neutral-component reason. Do not divide by zero, invent a variation value, or redistribute its weight.

Combine the four component scores equally:

```text
stylometric_score = (
    sentence_component
    + vocabulary_component
    + punctuation_component
    + function_word_component
) / 4
```

Equal weighting keeps the initial rule transparent and avoids pretending one component's reliability has been measured more precisely than another's. Preserve raw measurements, component scores, neutral-component reasons, and the final stylometric score in the audit data.

#### Why these are two distinct signals

The Groq signal is a holistic model judgment. Stylometry is a deterministic calculation over text structure. They use different methods, but they are **not assumed to be statistically independent**: both may respond to the same stylistic regularity, and the four heuristic components can also be correlated. Agreement therefore does not establish correctness or multiply the strength of the evidence.

### Combining the Signals

For an eligible submission with two valid signal outputs:

```text
raw_combined_score = 0.60 × llm_score + 0.40 × stylometric_score
```

The LLM receives slightly more weight because the design uses it for broader contextual assessment; the simple structural signal remains substantial. **The 60/40 split is a transparent starting policy, not an empirically optimized reliability estimate.**

Do not implement a learned classifier or adjust weights automatically. Any manual change to a feature rule, weight, or threshold must first be documented here, assigned a new configuration version, and followed by another test pass.

## Uncertainty Representation

### Meaning of the score

For compatibility with the assignment, the response field will be named **`confidence`**. Its documented meaning is a **directional AI-evidence index**, not the probability that the attribution is correct:

| Value | Interpretation |
| --- | --- |
| Near `0.00` | The system's available signals lean toward human-like writing. |
| Near `0.50` | Evidence is weak, mixed, or insufficient for an attribution. |
| Near `1.00` | The system's available signals lean toward AI-like writing. |

A score of **0.60** means the combined signals lean somewhat AI-like but do not meet the AI-label threshold. It does **not** mean a 60% probability of AI authorship. Likewise, a low score can support the human label; it is not automatically low confidence in that label.

Return `confidence_semantics: "ai_evidence_index_not_probability"` so the API contract makes this distinction explicit. The reader-facing label will use words, not a percentage presented as factual authorship probability.

### Attribution thresholds

Use the unrounded score for decisions. Boundaries have no gaps:

| Score range, when both signals are valid and minimum text requirements are met | Attribution | Label category |
| --- | --- | --- |
| `0.00 <= score <= 0.30` | `likely_human` | High-confidence human |
| `0.30 < score < 0.80` | `uncertain` | Uncertain |
| `0.80 <= score <= 1.00` | `likely_ai` | High-confidence AI |

“High-confidence” describes a category under this prototype's scoring policy, **not a measured accuracy guarantee**. The stronger threshold toward AI and broad uncertain interval are intended to reduce unsupported accusations, but their actual false-positive behavior must be evaluated.

A weighted average can conceal disagreement. Preserve both scores for review; do not describe the average as proof of agreement. Version 1 will not add a separate disagreement veto beyond these thresholds.

### Abstention and failure rules

For fewer than 50 words or fewer than 3 analysis sentences, skip the external LLM request, retain any computable structural measurements, and return `uncertain`. Set `confidence = 0.50`, `raw_combined_score = null`, and `analysis_status = "insufficient_text"`. Individual unavailable signal scores remain `null`.

If the LLM or stylometric signal fails on otherwise eligible text, return `uncertain`, `confidence = 0.50`, `raw_combined_score = null`, and `analysis_status = "signal_unavailable"`. Preserve any valid signal output and the failure reason. Do not silently turn the remaining signal into a 100%-weighted detector.

Here, 0.50 is an explicit **abstention marker**, not an invented measurement. Include `uncertainty_reasons`, such as `too_few_words`, `too_few_sentences`, `llm_unavailable`, or `stylometry_unavailable`. For successful two-signal analysis, set `analysis_status = "complete"` and `confidence = raw_combined_score`.

### Illustrative scoring checks — not experimental results

| LLM score | Stylometric score | Combined score | Expected attribution |
| --- | --- | --- | --- |
| `0.10` | `0.20` | `0.140` | `likely_human` |
| `0.74` | `0.55` | `0.664` | `uncertain` |
| `0.90` | `0.65` | `0.800` | `likely_ai` |
| `0.99` | `0.89` | `0.950` | `likely_ai` |
| `0.99` | `0.30` | `0.714` | `uncertain` |

These are arithmetic fixtures for testing scoring logic. They must not be presented in the README as observed detector performance.

### Calibration and evaluation approach

This prototype will normalize measurements into heuristic scores; it will **not claim statistical probability calibration**. Four example submissions can demonstrate variation and reveal problems, but cannot establish that scores correspond to true authorship probabilities.

Evaluate the required known-origin human and AI examples plus two borderline cases. Add poetry, short text, and formal writing. Record both signals, the combined score, the output label, and any surprising results. Pay particular attention to human-authored examples labeled AI and to how often the system abstains.

Some illustrative paragraphs in the course handout may fall below the chosen minimum length. Keep those as abstention tests, and use longer known-origin examples with at least 50 words and 3 sentence units to test the full pipeline. Do not change the minimum merely to force an expected label.

Use separate examples for final demonstration after any rule adjustment. Preserve failures rather than selecting only flattering examples. Any later probability-calibration claim would require an appropriate separate evaluation design; that is outside this implementation's scope.

## Transparency Label Design

The following are the **exact three label strings** to implement and copy verbatim into `README.md`:

| Variant | Exact text |
| --- | --- |
| **High-confidence AI** | "Likely AI-generated. This text shows strong indicators of AI generation, but automated detection can be wrong. The creator may appeal this result." |
| **High-confidence human** | "Likely human-written. This text shows strong indicators of human authorship, though automated analysis cannot verify authorship with certainty." |
| **Uncertain** | "Uncertain. Our analysis does not provide enough evidence to confidently determine whether this text is human-written or AI-generated." |

The label function will map the final attribution to one of these constants. Apply abstention rules before selecting the label. Do not allow the LLM to generate a different public label on each request.

The labels will not claim “verified human,” accuse a creator of deception, or attach sanctions. The README will explain that “strong indicators” means strong under the current heuristic policy, not independently verified creation history.

For appealed content, preserve the original decision and label in the audit record. Show the current status separately with the exact notice: **"Under review. The creator has appealed this assessment; it has not been resolved."** This is a review-status notice, not a fourth attribution variant.

Before finalizing implementation, show the three labels to someone unfamiliar with the project and ask what each means, whether any sounds like proof, and how they would challenge a result. Record their actual feedback; do not invent a user test.

## Appeals Workflow

### Who can appeal and what they provide

A creator may contest any of the three attribution results. `POST /appeal` will accept:

```json
{
  "content_id": "existing-content-id",
  "creator_id": "test-user-1",
  "creator_reasoning": "I wrote this myself. The repeated phrases are an intentional stylistic choice, and I can describe my drafting process."
}
```

`creator_reasoning` must be a nonempty string after trimming, with a maximum of 2,000 characters. The creator ID must match the ID stored with the submission.

**Identity limitation:** this local prototype uses self-reported IDs, not authenticated accounts. Matching IDs prevents accidental mismatches but does not prevent impersonation. Real ownership enforcement is a deployment requirement, not something this prototype will claim to provide.

### Processing and state change

Validate the request, look up the content, check the claimed creator ID, and check for an existing active appeal. In one SQLite transaction, create the appeal, change the current content status from `classified` to **`under_review`**, and append an appeal event to the audit log.

Return a confirmation containing `appeal_id`, `content_id`, `status`, and the review-status notice. Preserve the original text, attribution, individual scores, combined score, and label. An appeal must not erase or replace the original decision.

The request field is `creator_reasoning`; the stored/audit field will be `appeal_reasoning`, matching the evidence expected in the assignment.

No automated reclassification, reviewer-resolution endpoint, file-evidence uploads, or promised review deadline will be implemented. The system records a request for review; it must not imply that a real staffed review service exists.

### Information available for review

Storage will associate the original text and creator ID with the original decision timestamp, both signal outputs, all four raw stylometric measurements and component scores, score configuration, model/prompt identifiers, label, appeal reasoning, appeal timestamp, and current status. The local `GET /log` output will surface decision and appeal events; a dedicated review dashboard is outside scope.

### Appeal edge cases

| Case | Required behavior |
| --- | --- |
| Content ID does not exist | Return `404`; create no appeal and change no status. |
| An active appeal already exists | Return `409` with the existing appeal ID and current status; do not create a second active appeal. |
| Creator ID does not match | Return `403`; do not reveal another creator's text or reasoning. This remains only a claimed-ID check. |
| Missing, blank, wrongly typed, or overlength reasoning | Return `400`; preserve existing records unchanged. |
| Human or uncertain classification is contested | Accept an otherwise valid appeal; appeals are not restricted to AI labels. |
| Saving the appeal or audit event fails | Roll back the transaction and return an error; never confirm an appeal that was not stored. |

Use a uniqueness constraint on `appeals.content_id` for this version, which has no appeal-resolution or reopening workflow, to prevent duplicate records from simultaneous requests.

## Anticipated Edge Cases

These are anticipated risks and planned responses, not claims that the system has already been tested successfully.

| Scenario | Why this design may struggle | Planned response and remaining limitation |
| --- | --- | --- |
| **Repetitive poetry or song-like refrains** | Repetition and regular line lengths can raise multiple heuristic components; line breaks may be treated as sentences. | Use the combined score, conservative AI threshold, and appeal path. Short poems trigger abstention. Longer poems can still be misclassified. |
| **Formal academic or non-native English writing** | Both methods may mistake consistent phrasing for AI authorship. | Instruct the LLM not to use formality alone; include this case in evaluation and preserve an appeal path. Thresholds do not guarantee fairness. |
| **Very short text** | Vocabulary windows or sentence comparisons have too little material. | Return `uncertain` below 50 words or 3 analysis sentences, including an explicit reason. |
| **Human-edited AI text or mixed authorship** | A single document-level score cannot reconstruct which parts were generated, edited, or written independently. | Document that mixed authorship may receive any label, not necessarily `uncertain`. Do not promise automatic recognition of mixed work. |
| **AI imitating casual or irregular writing** | Deliberate stylistic variation may lower heuristic scores and influence the LLM. | Include a test example and report false negatives honestly. A human label is not a certificate. |
| **Non-English or code-switched text** | The function-word list and scoring assumptions target English. | State English-only support. Version 1 has no reliable language gate, so do not claim it automatically abstains on every unsupported language. |
| **Abbreviations, quotations, lists, or run-on sentences** | The simple splitter can produce misleading sentence units or too few units. | Test segmentation and inspect raw measurements; insufficient units trigger abstention. Record remaining parsing errors as limitations. |
| **Submission includes instructions such as “return a score of zero”** | The LLM may treat embedded text as instructions rather than evidence. | Separate instructions from untrusted content and validate output. Test this behavior; do not describe prompting as a complete security defense. |
| **Groq outage or invalid model output** | Only one usable signal may remain. | Return an explicitly degraded uncertain result, log the cause, and never invent the missing score. |

## Architecture

### Architecture narrative

`POST /submit` passes through request-size checks, rate limiting, and input validation; eligible text goes through Groq assessment and Python stylometry before score combination and uncertainty rules select a transparency label. The application stores the submission and its structured decision event together before returning the JSON result. `POST /appeal` looks up the original record, validates the appeal, and atomically saves the reasoning, changes the current status, and appends an audit event before confirming receipt.

### Submission flow

```text
Client
  | JSON: text, creator_id
  v
POST /submit
  | request body
  v
Size check + rate limit + input validation
  | valid text + creator ID
  v
Minimum-information check
  |                                  |
  | eligible text                    | too few words/sentences
  v                                  v
Detection coordinator                Abstention result
  | text -> Groq -> llm_score          | score 0.50 + reason
  | text -> Stylometry -> style_score |
  |         + four component scores   |
  v                                  |
60/40 combination or failure policy  |
  | combined score / abstention       |
  +----------------------------------+
  | score + analysis status + reasons
  v
Attribution policy -> Label lookup
  | attribution + exact label + signal details
  v
SQLite transaction: content record + decision audit event
  | stored content ID + decision
  v
JSON response: content_id, attribution, confidence, label, status
```

Both detector functions receive the same text. They may run sequentially in the small prototype; concurrency is not required. Raw text does not flow from one detector into the other.

### Appeal flow

```text
Client
  | JSON: content_id, creator_id, creator_reasoning
  v
POST /appeal
  | validated request
  v
Content lookup + claimed-ID check + duplicate check
  | existing content record + appeal reasoning
  v
SQLite transaction
  | save appeal
  | update current content status to under_review
  | append linked appeal audit event
  v
JSON confirmation: appeal_id, content_id, status, review notice
```

`GET /log` will read structured events from SQLite and return JSON for local inspection and grading. It does not create new attribution decisions.

## API Contract

| Endpoint | Accepts | Successful result |
| --- | --- | --- |
| `POST /submit` | JSON with `text` and `creator_id` | `200` with unique content ID, attribution, confidence semantics, exact label, analysis status, signal scores, uncertainty reasons, and current status. |
| `POST /appeal` | JSON with `content_id`, `creator_id`, and `creator_reasoning` | `200` with appeal ID, content ID, `under_review`, and confirmation notice. |
| `GET /log` | Optional integer `limit`, default 20, allowed 1–100 | `200` with `{"entries": [...]}`, newest event first. |

For submissions, require a nonempty string `creator_id` of at most 100 characters and nonblank text containing at least one word token. Set a 20,000-character text maximum and a 64 KiB request-body limit. Invalid fields return `400`, non-JSON requests return `415`, and oversized submissions return `413`; never silently truncate text.

Example **planned** successful response:

```json
{
  "content_id": "generated-uuid",
  "attribution": "uncertain",
  "confidence": 0.664,
  "confidence_semantics": "ai_evidence_index_not_probability",
  "raw_combined_score": 0.664,
  "label": "Uncertain. Our analysis does not provide enough evidence to confidently determine whether this text is human-written or AI-generated.",
  "status": "classified",
  "analysis_status": "complete",
  "signals": {
    "llm_score": 0.74,
    "stylometric_score": 0.55
  },
  "uncertainty_reasons": ["score_in_uncertain_range"]
}
```

`status` tracks the workflow (`classified` or `under_review`); `attribution` tracks the assessment; `analysis_status` distinguishes complete analysis from abstention. Do not conflate these fields. Return JSON error bodies consistently and do not expose secrets or stack traces.

## Storage, Audit Logging, and Rate Limiting

### Storage and audit structure

Use SQLite with three tables: `content`, `appeals`, and `audit_events`. Content stores the submitted text and original decision plus current workflow status. Appeals stores the linked reasoning and timestamp. Audit events store an append-only history; the current status may change without rewriting historical events.

A decision event will contain a unique event ID, UTC timestamp, content ID, creator ID, event type, attribution, reported confidence, raw combined score, label, both signal scores/statuses, raw stylometric measurements, four component scores, uncertainty reasons, model/prompt metadata, and score configuration version. An appeal event will link to the same content and original decision, recording the original attribution and confidence, its appeal ID, `appeal_reasoning`, and resulting `under_review` status.

Persist the content record and decision event atomically before reporting submission success. If storage fails, return `500` rather than claiming the assessment was saved. Appeals use the corresponding atomic transaction described above.

At least three real structured events will be generated for the README demonstration: two submissions and one appeal linked to one of them. The README will show the actual output, with any necessary redaction clearly marked. Illustrative planning examples are not substitutes for this evidence.

### Rate-limit policy

Use Flask-Limiter on `POST /submit` with the committed configuration:

```text
SUBMISSION_RATE_LIMIT = "3 per 5 minutes;100 per day"
RATE_LIMIT_STORAGE_URI = "memory://"
Rate-limit key = remote IP address
```

Three submissions in five minutes let a creator compare a draft with two revisions, then make the limit easy to demonstrate on a fourth attempt in the same window. The 100-per-day cap allows repeated drafting over a day while limiting automated bulk submissions. These are local application policies, not claims about Groq's account quotas. Validate and rate-limit before calling the provider, and count invalid attempts toward the request allowance. The browser will show a retry popup on a `429` response.

Excess requests will receive `429` in JSON with a retry indication. The local in-memory limiter will not be described as durable or appropriate for a distributed deployment: restarts reset local counters, multiple processes do not share them, and shared-IP users can affect one another. Do not trust arbitrary forwarded-IP headers in this local setup.

The README must contain the chosen limits, their rationale, and captured output demonstrating actual `429` responses. Test with a clean quota and more than three rapid attempts; separate application-limit behavior from provider failures. A documented detector stub may isolate the rate-limit test, but must not be passed off as live detector evidence.

### Privacy and prototype boundaries

Bind the demo to localhost. The unauthenticated log endpoint is for local grading visibility, not public deployment. Keep `.env`, the runtime database, and private submissions out of Git; commit only approved test fixtures and documented sample output. Explain that eligible text is sent to Groq for assessment. Never log API keys.

Before real deployment, authentication, actual ownership enforcement, access-controlled review, durable shared rate limiting, and explicit text-retention/deletion policies would be required. They are not completed features of this class prototype.

## Testing and Acceptance Criteria

| Test area | Planned checks and required evidence |
| --- | --- |
| **Signal interfaces** | Test each signal independently. Confirm valid ranges, strict response parsing, and preservation of raw measurements. |
| **Feature calculations** | Hand-check token counts, repeated-word diversity, equal-length sentences, missing internal punctuation, zero function-word matches, and three-sentence minimum behavior. |
| **Score boundaries** | Test `0.30`, just above `0.30`, just below `0.80`, and `0.80`; verify exact label selection without rounding gaps. Check both mapping endpoints and monotonic interpolation. |
| **Minimum information** | Test 49 versus 50 words and 2 versus 3 sentence units. Confirm abstention uses 0.50 with an explicit reason and null unavailable scores. |
| **Known-origin examples** | Submit at least one documented AI-generated text, one documented human-written text, and two borderline cases. Keep human authorship evidence separate from assumptions based on style. |
| **Borderline content** | Include formal human writing and edited AI writing. Add poetry and an AI-generated casual style; inspect disagreements and report mistakes. |
| **Reachable labels** | Demonstrate all three labels through actual submissions, alongside deterministic scoring tests. A stub proves routing, not detection. Do not hardcode particular text to manufacture a label. |
| **Appeals** | Confirm reasoning is stored, status becomes `under_review`, the original decision remains intact, and unknown IDs, duplicate appeals, mismatched IDs, and invalid reasoning are handled. |
| **Rate limiting** | Capture successful requests and subsequent `429` responses under the documented local test conditions. |
| **Logging and persistence** | Confirm at least three structured events including an appeal, a working `GET /log`, and persistence of content and audit history after restarting the app. |
| **Failure behavior** | Simulate provider failure, malformed model output, and storage failure. Verify no fabricated scores, false success messages, or leaked secrets. |

In the final README, show two actual submissions with noticeably different scores, including a high-score/high-confidence case and an uncertain case. Explain the score semantics, limitations, and any rule revisions. Do not call test intentions “results” until they have been run.

## AI Tool Plan

AI assistance will be driven by this specification. Generated code must be reviewed against the named interfaces, thresholds, and failure behavior. Keep notes on the request, output, and actual changes made for the README's AI usage reflection.

### M3 — Submission Endpoint and First Signal

**Provide to the AI tool:** Project Goal and Scope; Signal 1; Architecture; API Contract; and Storage/Audit requirements.

**Request:** A Flask skeleton with `POST /submit`, a standalone Groq assessment function, unique content IDs, initial SQLite decision logging, and `GET /log`. Start with a hardcoded route response only to verify request handling; then wire in the independently tested first signal. During this milestone, confidence and label may be clearly identified placeholders, as permitted by the assignment.

**Verify:** Required fields and invalid-input responses; the LLM function's return schema; score-range validation; safe handling of embedded instructions; unique IDs; stored log entries; and correspondence between the returned content ID and its audit event. Test the function directly before relying on the endpoint.

### M4 — Second Signal and Confidence Scoring

**Provide to the AI tool:** All Detection Signals sections; Uncertainty Representation; Architecture; and the relevant acceptance tests.

**Request:** Pure-Python extraction of the four specified features, exact component mappings, equal-weight stylometric aggregation, 60/40 combination, minimum-information/failure rules, and extended audit details. Explicitly prohibit adding a trained classifier or changing thresholds silently.

**Verify:** Hand-calculated measurements, neutral components, missing-signal handling, the illustrative arithmetic fixtures, and exact boundary behavior. Submit at least the four required real examples; compare both signals and investigate surprising outputs without assuming intuition proves ground truth.

### M5 — Transparency and Production Layer

**Provide to the AI tool:** Transparency Label Design; Appeals Workflow; Architecture; API Contract; Storage/Audit/Rate Limiting; and the associated tests.

**Request:** Exact label constants and lookup logic, the `POST /appeal` endpoint, transactional status/audit updates, duplicate handling, Flask-Limiter configuration, and structured JSON responses.

**Verify:** All three exact labels and abstention paths; preservation of the original decision after an appeal; stored reasoning and `under_review`; duplicate/invalid appeal behavior; actual `429` responses; and at least three inspectable audit events covering submissions and an appeal.

### Documenting AI use honestly

For at least two actual interactions, record what I asked the AI to generate, what it returned, and what I changed or overrode. Do not invent mistakes, corrections, or tests to satisfy the reflection requirement. These implementation notes will become the README's AI usage section.

## Milestone Sequence and Final Deliverables

Complete this specification and its diagram before implementation. Then proceed through M3's first working signal and logging, M4's four-feature signal and scoring, M5's labels/appeals/rate limits, and M6's README plus short recorded walkthrough.

The final README will include setup and model configuration, the architecture, both signals and their rationale, exact formulas and thresholds, all three verbatim labels, actual score examples, appeals behavior, the rate-limit configuration and `429` evidence, at least three audit entries, known limitations, a spec reflection, and specific AI usage examples. The walkthrough will show the system working and explain selected design decisions; it does not replace written evidence.

The spec reflection will distinguish what this plan helped implement from any documented changes made after testing. The separate course grading page was not included in the supplied materials and should be checked before submission. No stretch feature is included in version 1; update this plan before starting one.
