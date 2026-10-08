# Provenance Guard

[Watch the project demo](https://drive.google.com/file/d/1VzftuMg7gHpOoHkYw12CXrO7YTCDe4Oc/view?usp=sharing)

Provenance Guard is a local Flask prototype for English creative text. It combines a Groq language-model assessment and deterministic stylometry, returns a directional AI-evidence index and one of three attribution categories, and records decisions and appeals in SQLite. It cannot prove authorship. The technical contract is [planning.md](planning.md).

## Install and run

Use Python 3.12 or a compatible recent Python 3:

```sh
python3 -m venv .venv
./.venv/bin/python -m pip install -r requirements.txt
cp .env.example .env
```

Set your own `GROQ_API_KEY` and an account-available `GROQ_MODEL` in `.env`. The live check below explicitly used `openai/gpt-oss-20b`; the app never substitutes a different model automatically. Keep the key private. Eligible submitted text is sent to Groq, without the creator ID or stylometric scores.

```sh
./.venv/bin/python app.py
./.venv/bin/python -m unittest discover -s tests -v
```

The app binds to `localhost:7860`. Open `http://localhost:7860/` in a browser for the local interface: enter a creator ID and passage, view the assessment and the four stylometric measurements/component scores, appeal it, and inspect recent audit events. The result page shows the equal-weight stylometric average and the 60/40 combined-score calculation using six-decimal display values; scoring and classification use unrounded values. The browser page calls the same API routes documented below. App creation initializes the SQLite schema. The default database is `data/provenance_guard.sqlite3`; `DATABASE_PATH` can override it. `.env` and runtime databases are ignored by Git. The unit suite needs no network. The optional live integration check is `PYTHONPATH=. ./.venv/bin/python tests/manual_live.py`; it disables the app rate limit for its batch, makes real Groq calls, and writes six bundled submissions and an appeal. The regular app retains the 3-per-5-minutes limit. To add the two speech cases, set `MLK_SPEECH_PATH` and `MALCOLM_SPEECH_PATH` to local UTF-8 excerpt files outside the repository. Their transcripts are not bundled or committed.

The route in [app.py](app.py) validates requests and coordinates the two signals. [llm_signal.py](llm_signal.py) owns the provider call and response validation; [stylometry.py](stylometry.py) computes the four fixed features; [scoring.py](scoring.py) combines scores and applies thresholds; [database.py](database.py) commits content, appeals, and append-only audit events transactionally. [config.py](config.py) centralizes policy constants and environment settings. [templates/index.html](templates/index.html), [static/app.css](static/app.css), and [static/app.js](static/app.js) provide the local browser interface without changing API response contracts.

## API and workflow

`POST /submit` accepts JSON with `text` and `creator_id`. Text must be a nonblank string containing a word token and no more than 20,000 characters; the request body limit is 64 KiB. Creator IDs must be nonblank strings of at most 100 characters. Invalid fields return JSON `400`, non-JSON `415`, and oversized bodies `413`. A successful submission returns `200` with `content_id`, `attribution`, `confidence`, `confidence_semantics`, `raw_combined_score`, `label`, `status`, `analysis_status`, `signals`, and `uncertainty_reasons`, plus local signal details.

This quick manual request tests abstention without a Groq call:

```sh
curl -sS -X POST http://localhost:7860/submit \
  -H 'Content-Type: application/json' \
  -d '{"text":"A brief sentence.","creator_id":"local-demo"}'
```

It returns `uncertain`, `confidence: 0.5`, `raw_combined_score: null`, and `analysis_status: "insufficient_text"`. For both signals, submit at least 50 words and three analysis sentences; `tests/manual_live.py` contains eligible example requests.

`POST /appeal` accepts an existing `content_id`, its claimed `creator_id`, and nonblank `creator_reasoning` of at most 2,000 characters:

```sh
curl -sS -X POST http://localhost:7860/appeal \
  -H 'Content-Type: application/json' \
  -d '{"content_id":"REPLACE_WITH_CONTENT_ID","creator_id":"local-demo","creator_reasoning":"Please review my drafting process."}'
```

It stores the supplied reasoning as `appeal_reasoning`, returns `appeal_id`, `content_id`, `under_review`, and the exact notice below. In one transaction it inserts the appeal, updates the current workflow status, and appends a linked audit event. The browser then opens an “Appeal recorded” popup with the appeal ID and leaves a confirmation on the page. Recent audit entries show attribution, confidence, and the creator ID for both decisions and appeals, plus the appeal ID and reasoning for appeal events. Older appeal events that lack attribution and confidence in their stored payload receive those fields from the preserved content decision when `/log` is read; the old event payload is not rewritten. The original text, scores, attribution, and label remain intact. Any attribution can be appealed. Unknown content returns `404`, creator mismatch `403`, duplicate active appeal `409`, invalid reasoning `400`, and storage failure `500` with rollback. Claimed IDs are not authentication, and an appeal does not rerun either detector or promise staffed review.

`GET /log` returns `{"entries": [...]}` in newest-first order. `limit` defaults to 20 and accepts integers 1–100:

```sh
curl -sS 'http://localhost:7860/log?limit=20'
```

This unauthenticated log is for local grading visibility, not public deployment.

## Detection and scoring

The Groq signal receives only the original submitted text. A system prompt treats that text as untrusted data, requests JSON with a finite `ai_score` in `[0,1]` and nonblank `reasoning` of at most 600 characters, asks for weak evidence near 0.5, and avoids inferring authorship from formality, slang, errors, or anecdotes alone. It runs at temperature `0.1`, a 20-second timeout, and zero automatic retries. Invalid output or a provider error marks the signal unavailable. The model never chooses the public category or label. This holistic signal may catch contextual patterns that four counts miss, but its explanations are not verified facts.

Stylometry preserves the original text in storage and analyzes a copy with standardized line endings. It counts Unicode letter words, permits internal apostrophes, normalizes curly apostrophes, lowercases, and splits hyphenated words. Analysis sentences split at runs of `.`, `!`, or `?` followed by whitespace/end, and at line breaks; wordless fragments are ignored.

| Feature | Raw measurement | Component mapping |
| --- | --- | --- |
| Sentence-length variation | Population standard deviation of sentence word counts divided by their mean (`sentence_cv`) | `regularity_component(sentence_cv)` |
| Vocabulary diversity | Whole-text type-token ratio logged; mean type-token ratio of consecutive complete 50-word blocks (`ttr_50`) scored. An incomplete final block is excluded from this feature only. | `0.90 - 0.80 * clip((ttr_50 - 0.45) / 0.40, 0, 1)` |
| Punctuation usage | Per-sentence density and variation of internal `,;:—–()[]{}`; sentence-ending marks excluded | `regularity_component(punctuation_cv)` |
| Function words | Per-sentence rate and variation using the exact fixed list in [stylometry.py](stylometry.py) | `regularity_component(function_word_cv)` |

`regularity_component(cv) = 0.90 - 0.80 * clip((cv - 0.10) / 0.70, 0, 1)`. Zero mean punctuation or function-word density gives a `null` variation, a neutral `0.50` component, and a recorded reason. The four components have equal weight: `stylometric_score = (sentence + vocabulary + punctuation + function_word) / 4`. These rules are unvalidated hypotheses, not learned classifier features.

With eligible text and both valid signals, `raw_combined_score = 0.60 * llm_score + 0.40 * stylometric_score`. The unrounded score gives `likely_human` for `0.00 <= score <= 0.30`, `uncertain` for `0.30 < score < 0.80`, and `likely_ai` for `0.80 <= score <= 1.00`. The field `confidence` means `ai_evidence_index_not_probability`; `0.60` is **not** a 60% probability of AI authorship. Below 50 words or three sentences, Groq is skipped and the result is `uncertain`, `confidence: 0.50`, `raw_combined_score: null`, `analysis_status: "insufficient_text"`. An unavailable signal on otherwise eligible text gives the same abstention values with `analysis_status: "signal_unavailable"`. A surviving signal is never reweighted to 100%. Reasons identify insufficient information, unavailable signals, or a complete score in the uncertain range.

These are the exact three public labels, chosen from application constants:

> Likely AI-generated. This text shows strong indicators of AI generation, but automated detection can be wrong. The creator may appeal this result.

> Likely human-written. This text shows strong indicators of human authorship, though automated analysis cannot verify authorship with certainty.

> Uncertain. Our analysis does not provide enough evidence to confidently determine whether this text is human-written or AI-generated.

“Strong indicators” means strong under this heuristic policy, not verified creation history. Appeal status is a separate notice, not a fourth attribution label:

> Under review. The creator has appealed this assessment; it has not been resolved.

## Observed evidence

On 2026-10-07, the eight complete examples below were submitted through Groq model `openai/gpt-oss-20b`. The AI prose, poetry, and code-switched text were generated by Codex for this evaluation. The human passages are from Charlotte Perkins Gilman's 1892 [*The Yellow Wallpaper*](https://www.gutenberg.org/cache/epub/1952/pg1952-images.html), a creator-submitted passage identified as Martin Luther King Jr.'s [“I Have a Dream”](https://kinginstitute.stanford.edu/i-have-dream), and a 160-word early paragraph from a transcript of Malcolm X's [“The Ballot or the Bullet”](https://www.speeches-usa.com/Transcripts/malcolm_x-ballot.html). The MLK text was already in the local database; the Malcolm X passage was fetched and submitted once without adding its text to the repository. The mixed-source passage concatenates the Gilman excerpt and Codex's repetitive prose; it is not a human edit of AI text. An earlier Gettysburg test remains in the append-only audit history but is omitted from this comparison at the user's request. The first script run appealed the casual AI result. These are observed outputs, not accuracy estimates:

| Origin and input | Words / sentences | Groq | Stylometry | Combined index | Attribution |
| --- | ---: | ---: | ---: | ---: | --- |
| AI, repetitive prose | 81 / 3 | 0.600 | 0.875 | 0.710 | `uncertain` |
| Human, Gilman excerpt | 73 / 5 | 0.600 | 0.3574 | 0.5030 | `uncertain` |
| AI, casual prose | 63 / 4 | 0.350 | 0.4875 | 0.4050 | `uncertain` |
| Mixed, Gilman plus AI prose | 154 / 8 | 0.600 | 0.4473 | 0.5389 | `uncertain` |
| AI, repetitive poetry | 72 / 10 | 0.550 | 0.7635 | 0.6354 | `uncertain` |
| Human, creator-submitted MLK “I Have a Dream” passage | 420 / 40 | 0.600 | 0.3234 | 0.4894 | `uncertain` |
| Human, Malcolm X “The Ballot or the Bullet” excerpt | 160 / 6 | 0.300 | 0.3819 | 0.3328 | `uncertain` |
| AI, English/Spanish code-switching | 66 / 4 | 0.400 | 0.7150 | 0.5260 | `uncertain` |

The highest index in this comparison was 0.710. Both speech passages were classified `uncertain`; their known human provenance was not inferred from their scores. The AI-origin submissions received varied scores, yet all remained uncertain. The Groq explanation for Gilman's text called its phrasing potentially AI-like despite the documented human source. The scoring rules were not adjusted to make the results look more accurate. Stubbed end-to-end policy tests reached all three categories and appealed each one; they establish routing behavior only, not live detector performance.

This is an excerpt of three structured events returned by `GET /log?limit=3`, newest first, in a fresh temporary database after two short submissions and an appeal. The short inputs abstained without a Groq call, so all three confidence values are the 0.50 abstention marker. The full decision events also contain creator IDs, exact labels, signal details, measurements, and configuration metadata. The excerpt omits those fields for readability; IDs, timestamps, and scores are unchanged:

```json
{"event_id":"d34ca740-bcac-445a-8ace-fb97b73598c2","timestamp":"2026-10-08T02:33:23.765617Z","event_type":"appeal","content_id":"a1b24ce7-21d0-4897-9ad9-67e5554208c3","attribution":"uncertain","confidence":0.5,"appeal_id":"0f423f62-b66c-4b93-85ef-88efe8353e76","original_decision_event_id":"06563082-149e-44c9-801d-d20dd2e37033","appeal_reasoning":"I can explain my drafting process.","status":"under_review"}
{"event_id":"060667b6-5454-4c79-bc54-1339be59dd4f","timestamp":"2026-10-08T02:33:23.764580Z","event_type":"decision","content_id":"a9b85c15-74b5-4e5c-bc52-46467b1763f6","attribution":"uncertain","confidence":0.5,"status":"classified"}
{"event_id":"06563082-149e-44c9-801d-d20dd2e37033","timestamp":"2026-10-08T02:33:23.763114Z","event_type":"decision","content_id":"a1b24ce7-21d0-4897-9ad9-67e5554208c3","attribution":"uncertain","confidence":0.5,"status":"classified"}
```

Flask-Limiter applies `3 per 5 minutes;100 per day` to `POST /submit`, keyed by remote IP with `memory://` storage. Three submissions in five minutes let a creator compare a draft with two revisions while spacing out further checks; a fourth request in that window is a practical demo of the limit. The daily cap allows substantial writing and editing through a day while discouraging automated bulk submissions. Invalid attempts count. A clean-quota local test sent three invalid-field submissions and then an eligible fourth submission. No Groq call was made. Captured output:

```text
first_three_statuses: [400, 400, 400]
fourth_status: 429
body: {"error": "Submission rate limit exceeded"}
Retry-After: 300
```

Reproduce with `./.venv/bin/python -m unittest discover -s tests -p 'test_m5.py' -v`. Flask-Limiter computes `Retry-After` for the limit reached; a separate test checks the longer daily-limit interval. The browser shows a “Too many submissions” popup with the retry message on `429`; the error also remains under the form. The in-memory counter resets on restart, does not span processes, and groups users behind a shared IP.

## Limitations

Repetitive human poetry can raise several regularity components; short poetry abstains. Formal human writing may look regular to both signals. Mixed authorship and human-edited AI text cannot be reconstructed from one document-level score. Casual AI text may score lower, as observed. The assumptions and function-word list target English; there is no reliable language gate for non-English or code-switched input. Abbreviations, quotations, lists, poetry line breaks, and run-on sentences can distort analysis sentences. Separating embedded instructions from the system prompt reduces, but does not eliminate, prompt-injection risk. Provider outages and malformed replies cause explicit abstention. The two signals can respond to the same style, so agreement is not proof.

## Spec Reflection

**One way `planning.md` helped during implementation:** It fixed the four stylometric feature definitions, the neutral-component behavior, the 60/40 weights, and the exact thresholds before testing. When live AI-written repetitive prose scored 0.710 and received `uncertain`, I kept the specified 0.80 AI threshold instead of changing it to make the demonstration look more decisive. Boundary tests and the audit fields follow those planned rules.

**One divergence from the spec, and why:** The plan described a backend and did not require a frontend. The first browser-facing result at `/` was only a `GET` response containing a JSON list of endpoints. After I pointed out that this was not a usable UI, the implementation added a local page for submission, results, appeals, and recent audit events. This extension makes the prototype usable at `localhost:7860` without changing the three API contracts or any detection rule. A browser-driven check exercised submission and appeal through that page; it used short text and made no Groq request.

## AI Usage

**Instance 1 — replacing the JSON index with a UI**

- *What I gave the AI:* The existing Flask backend at `localhost:7860` and my feedback that opening `/` showed only `{"endpoints":["POST /submit","POST /appeal","GET /log"],"service":"Provenance Guard"}` instead of a UI.
- *What it produced:* The initial output was a `GET /` JSON route list. After the correction, it generated a browser page with a text-submission form, attribution and signal display, appeal form, and recent audit view.
- *What I changed or directed:* I directed it to redo the root page as a usable UI. I kept the backend scoring and label rules from the plan, then checked the page in Chrome and exercised a short-text submission and appeal through the browser.

**Instance 2 — implementing the Groq signal**

- *What I gave the AI:* The plan's standalone `assess_with_llm(text)` contract, prompt constraints, configured-model requirement, and failure rules.
- *What it produced:* A Groq prompt separated from submitted text, strict JSON parsing, and a validated score/reasoning result with model metadata.
- *What I changed or directed:* I required provider failures to be explicit and scores never to be fabricated. During implementation review, a failing test exposed Groq client construction outside the signal-error path; the code was changed so that failure marks the signal unavailable.

**Instance 3 — checking stylometric abstention**

- *What I gave the AI:* The exact four feature calculations, component mappings, 50-word and three-sentence minimum, and uncertainty rules from the plan.
- *What it produced:* Pure-Python measurements, four component scores, an equal-weight stylometric score, and submission logic for combining both signals.
- *What I changed or directed:* I required Groq to be skipped below either minimum. A regression test exposed that a stylometry exception could hide the sentence count and allow an unnecessary Groq call; the implementation then shared analysis-unit preprocessing and checked minimum information before the external request.

## Remaining work

Still pending from the plan: feedback from someone unfamiliar with the three labels; a known-origin human-edited AI example; a live `likely_ai` result; and the short recorded M6 walkthrough. Unit tests cover the documented abbreviation/quotation splitter behavior. None of the pending items is represented as completed. Public deployment would also require authentication, ownership checks, access-controlled logging/review, durable shared rate limiting, and a text-retention policy.
