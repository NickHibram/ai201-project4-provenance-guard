const byId = (id) => document.getElementById(id);
const submissionForm = byId('submission-form');
const appealForm = byId('appeal-form');
let currentSubmission = null;

function status(element, message, isError = false) {
  element.textContent = message;
  element.classList.toggle('error', isError);
}

async function requestJson(path, options) {
  const response = await fetch(path, options);
  const body = await response.json().catch(() => ({}));
  if (!response.ok) {
    let message = body.error || `Request failed (${response.status})`;
    const retryAfter = response.headers.get('Retry-After');
    if (response.status === 429 && retryAfter) {
      message += ` Try again in about ${retryAfter} seconds.`;
    }
    const error = new Error(message);
    error.httpStatus = response.status;
    throw error;
  }
  return body;
}

function readableAttribution(value) {
  return {
    likely_human: 'Likely human-written',
    likely_ai: 'Likely AI-generated',
    uncertain: 'Uncertain'
  }[value] || value;
}

function scoreText(value) {
  return typeof value === 'number' ? value.toFixed(3) : 'Unavailable';
}

function precise(value) {
  return typeof value === 'number' ? value.toFixed(6) : 'Unavailable';
}

function rateList(values) {
  return Array.isArray(values) && values.length
    ? values.map(precise).join(', ')
    : 'Unavailable';
}

function renderStylometry(result) {
  const style = result.stylometry || {};
  const measurements = style.measurements || {};
  const components = style.components || {};
  const reasons = style.neutral_component_reasons || [];
  byId('style-sentence-score').textContent = precise(components.sentence);
  byId('style-vocabulary-score').textContent = precise(components.vocabulary);
  byId('style-punctuation-score').textContent = precise(components.punctuation);
  byId('style-function-score').textContent = precise(components.function_word);
  byId('style-sentence-raw').textContent = `Sentence lengths (words): ${(measurements.sentence_lengths || []).join(', ') || 'Unavailable'}; CV: ${precise(measurements.sentence_cv)}`;
  byId('style-vocabulary-raw').textContent = `50-word block TTR: ${precise(measurements.ttr_50)}; words used: ${measurements.ttr_words_used ?? 'Unavailable'}`;
  byId('style-punctuation-raw').textContent = `Density: ${precise(measurements.punctuation_density)}; sentence rates: ${rateList(measurements.punctuation_rates)}; CV: ${precise(measurements.punctuation_cv)}${reasons.includes('zero_punctuation_density') ? '; neutral 0.500000 because no internal punctuation was found' : ''}`;
  byId('style-function-raw').textContent = `Overall frequency: ${precise(measurements.function_word_frequency)}; sentence rates: ${rateList(measurements.function_word_rates)}; CV: ${precise(measurements.function_word_cv)}${reasons.includes('zero_function_word_frequency') ? '; neutral 0.500000 because no function words were found' : ''}`;
  byId('style-average').textContent = precise(style.stylometric_score);
  byId('combined-math').textContent = result.raw_combined_score === null
    ? `Combined score not calculated (${result.analysis_status.replaceAll('_', ' ')}). The displayed 0.500 index is an abstention value.`
    : `Combined AI-evidence index = 0.60 × Groq ${precise(result.signals.llm_score)} + 0.40 × stylometry ${precise(style.stylometric_score)} = ${precise(result.raw_combined_score)}.`;
}

function renderResult(result) {
  const panel = byId('result-panel');
  panel.hidden = false;
  const badge = byId('result-attribution');
  badge.textContent = readableAttribution(result.attribution);
  badge.className = `attribution-badge ${result.attribution}`;
  byId('result-analysis-status').textContent = result.analysis_status.replaceAll('_', ' ');
  byId('result-label').textContent = result.label;
  byId('result-index').textContent = scoreText(result.confidence);
  byId('result-index-note').textContent = result.analysis_status === 'complete'
    ? 'Directional index, not authorship probability'
    : '0.500 marks abstention; it is not a measured signal';
  byId('result-marker').style.left = `${Math.max(0, Math.min(1, result.confidence)) * 100}%`;
  byId('result-llm').textContent = scoreText(result.signals.llm_score);
  byId('result-style').textContent = scoreText(result.signals.stylometric_score);
  renderStylometry(result);
  byId('result-content-id').textContent = result.content_id;
  byId('result-reasons').textContent = result.uncertainty_reasons.length
    ? `Reasons: ${result.uncertainty_reasons.map((reason) => reason.replaceAll('_', ' ')).join(', ')}`
    : 'Both signals were available for this assessment.';
}

async function refreshLog() {
  const list = byId('audit-list');
  const message = byId('audit-status');
  status(message, 'Loading recent events…');
  try {
    const data = await requestJson('/log?limit=8');
    list.replaceChildren();
    if (!data.entries.length) {
      status(message, 'No audit events yet. Submit a passage to create one.');
      return;
    }
    status(message, '');
    for (const event of data.entries) {
      const item = document.createElement('li');
      item.className = 'audit-item';
      const type = document.createElement('span');
      type.className = 'audit-type';
      type.textContent = event.event_type;
      const detail = document.createElement('div');
      const title = document.createElement('strong');
      title.textContent = event.event_type === 'appeal'
        ? `Appeal received · under review · ${readableAttribution(event.attribution)} · ${scoreText(event.confidence)}`
        : `${readableAttribution(event.attribution)} · ${scoreText(event.confidence)}`;
      const id = document.createElement('small');
      id.textContent = `Content ${event.content_id}`;
      const creator = document.createElement('small');
      creator.textContent = `Creator ${event.creator_id}`;
      detail.append(title, creator, id);
      if (event.event_type === 'appeal') {
        const appealId = document.createElement('small');
        appealId.textContent = `Appeal ${event.appeal_id}`;
        const reasoning = document.createElement('small');
        reasoning.textContent = `Reason: ${event.appeal_reasoning}`;
        detail.append(appealId, reasoning);
      }
      const time = document.createElement('time');
      time.dateTime = event.timestamp;
      time.textContent = new Date(event.timestamp).toLocaleString();
      item.append(type, detail, time);
      list.append(item);
    }
  } catch (error) {
    status(message, error.message, true);
  }
}

byId('creative-text').addEventListener('input', (event) => {
  byId('char-count').textContent = `${event.target.value.length.toLocaleString()} / 20,000`;
});

submissionForm.addEventListener('submit', async (event) => {
  event.preventDefault();
  const button = byId('analyze-button');
  const creatorId = byId('creator-id').value;
  const text = byId('creative-text').value;
  currentSubmission = null;
  byId('result-panel').hidden = true;
  byId('appeal-panel').hidden = true;
  status(byId('submit-status'), 'Analyzing and saving the assessment…');
  button.disabled = true;
  try {
    const result = await requestJson('/submit', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({text, creator_id: creatorId})
    });
    currentSubmission = {contentId: result.content_id, creatorId};
    renderResult(result);
    appealForm.reset();
    appealForm.hidden = false;
    byId('appeal-confirmation').hidden = true;
    byId('appeal-button').disabled = false;
    status(byId('appeal-status'), '');
    byId('appeal-panel').hidden = false;
    status(byId('submit-status'), 'Assessment saved.');
    byId('result-panel').scrollIntoView({behavior: 'smooth', block: 'start'});
    await refreshLog();
  } catch (error) {
    status(byId('submit-status'), error.message, true);
    if (error.httpStatus === 429) {
      byId('rate-limit-popup-message').textContent = error.message;
      byId('rate-limit-popup').showModal();
    }
  } finally {
    button.disabled = false;
  }
});

appealForm.addEventListener('submit', async (event) => {
  event.preventDefault();
  if (!currentSubmission) return;
  const button = byId('appeal-button');
  status(byId('appeal-status'), 'Saving your appeal…');
  button.disabled = true;
  try {
    const result = await requestJson('/appeal', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({
        content_id: currentSubmission.contentId,
        creator_id: currentSubmission.creatorId,
        creator_reasoning: byId('appeal-reasoning').value
      })
    });
    status(byId('appeal-status'), result.notice);
    byId('result-analysis-status').textContent = 'under review';
    byId('appeal-confirmation-notice').textContent = result.notice;
    byId('appeal-confirmation-id').textContent = result.appeal_id;
    appealForm.hidden = true;
    byId('appeal-confirmation').hidden = false;
    byId('appeal-popup-notice').textContent = result.notice;
    byId('appeal-popup-id').textContent = result.appeal_id;
    byId('appeal-popup').showModal();
    await refreshLog();
  } catch (error) {
    status(byId('appeal-status'), error.message, true);
    button.disabled = false;
  }
});

byId('refresh-log').addEventListener('click', refreshLog);
byId('appeal-popup-close').addEventListener('click', () => byId('appeal-popup').close());
byId('rate-limit-popup-close').addEventListener('click', () => byId('rate-limit-popup').close());
refreshLog();
