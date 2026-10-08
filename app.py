"""Flask JSON API for the local Provenance Guard prototype."""

from uuid import uuid4
import sqlite3

from flask import Flask, jsonify, render_template, request
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from werkzeug.exceptions import RequestEntityTooLarge

import config
from database import get_log, initialize, save_appeal, save_decision
from llm_signal import SignalUnavailable, assess_with_llm
from scoring import attribution_for_score, combine_scores
from stylometry import analysis_units, analyze_text, tokenize


def error(message, status):
    return jsonify({'error': message}), status


def json_body():
    if not request.is_json:
        return None, error('JSON request required', 415)
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        return None, error('Invalid JSON object', 400)
    return data, None


def create_app(overrides=None):
    app = Flask(__name__)
    app.config.update(DATABASE_PATH=config.DATABASE_PATH, GROQ_MODEL=config.GROQ_MODEL,
                      MAX_CONTENT_LENGTH=config.MAX_REQUEST_BYTES,
                      RATELIMIT_STORAGE_URI=config.RATE_LIMIT_STORAGE_URI)
    if overrides:
        app.config.update(overrides)
    initialize(app.config['DATABASE_PATH'])
    limiter = Limiter(get_remote_address, app=app,
                      storage_uri=app.config['RATELIMIT_STORAGE_URI'],
                      headers_enabled=True)
    app._limiter = limiter

    @app.errorhandler(RequestEntityTooLarge)
    def too_large(_exc):
        return error('Request body exceeds 64 KiB', 413)

    @app.errorhandler(429)
    def rate_limited(_exc):
        return error('Submission rate limit exceeded', 429)

    @app.get('/')
    def index():
        return render_template('index.html')

    @app.post('/submit')
    @limiter.limit(config.SUBMISSION_RATE_LIMIT)
    def submit():
        data, problem = json_body()
        if problem:
            return problem
        original_text = data.get('text')
        creator_id = data.get('creator_id')
        if (not isinstance(original_text, str) or len(original_text) > config.MAX_TEXT_CHARS
                or not original_text.strip()):
            return error('text must be a nonblank string of at most 20000 characters', 400)
        if (not isinstance(creator_id, str) or not creator_id.strip()
                or len(creator_id) > config.MAX_CREATOR_ID_CHARS):
            return error('creator_id must be a nonblank string of at most 100 characters', 400)
        if not tokenize(original_text):
            return error('text must contain at least one word', 400)
        word_count = len(tokenize(original_text))
        sentence_count = len(analysis_units(original_text))
        too_few_words = word_count < config.MIN_WORDS
        too_few_sentences = sentence_count < config.MIN_SENTENCES
        insufficient = too_few_words or too_few_sentences
        try:
            style = analyze_text(original_text)
            style_status = 'available' if style['stylometric_score'] is not None else 'insufficient_text'
        except Exception:
            style = None
            style_status = 'unavailable'
        llm = None
        llm_status = 'skipped_insufficient_text' if insufficient else 'unavailable'
        llm_failure = None
        # No external request for submissions below either minimum.
        if not insufficient:
            try:
                llm = assess_with_llm(original_text, model=app.config['GROQ_MODEL'])
                llm_status = 'available'
            except SignalUnavailable as exc:
                llm_failure = str(exc)
        llm_score = llm['ai_score'] if llm else None
        style_score = style['stylometric_score'] if style else None
        reasons = []
        if too_few_words:
            reasons.append('too_few_words')
        if too_few_sentences:
            reasons.append('too_few_sentences')
        if insufficient:
            analysis_status = 'insufficient_text'
        elif llm_score is None or style_score is None:
            analysis_status = 'signal_unavailable'
            if llm_score is None:
                reasons.extend(['llm_unavailable', llm_failure or 'llm_unavailable'])
            if style_score is None:
                reasons.append('stylometry_unavailable')
        else:
            analysis_status = 'complete'
        if analysis_status == 'complete':
            combined = combine_scores(llm_score, style_score)
            attribution = attribution_for_score(combined)
            confidence = combined
            if attribution == 'uncertain':
                reasons.append('score_in_uncertain_range')
        else:
            combined = None
            attribution = 'uncertain'
            confidence = 0.5
        decision = {'attribution': attribution, 'confidence': confidence,
                    'confidence_semantics': config.CONFIDENCE_SEMANTICS,
                    'raw_combined_score': combined,
                    'label': config.LABELS[attribution],
                    'status': 'classified', 'analysis_status': analysis_status,
                    'signals': {'llm_score': llm_score, 'stylometric_score': style_score},
                    'llm': {'status': llm_status, 'reasoning': llm['reasoning'] if llm else None,
                            'model': llm['model'] if llm else app.config['GROQ_MODEL'],
                            'prompt_version': llm['prompt_version'] if llm else config.PROMPT_VERSION,
                            'temperature': config.LLM_TEMPERATURE,
                            'failure_reason': llm_failure},
                    'stylometry': {'status': style_status, **style} if style else {'status': style_status},
                    'uncertainty_reasons': reasons,
                    'score_config_version': config.SCORE_CONFIG_VERSION}
        content_id = str(uuid4())
        try:
            save_decision(app.config['DATABASE_PATH'], content_id, creator_id,
                          original_text, decision)
        except sqlite3.Error:
            return error('Could not save assessment', 500)
        return jsonify({'content_id': content_id, **decision})

    @app.post('/appeal')
    def appeal():
        data, problem = json_body()
        if problem:
            return problem
        content_id = data.get('content_id')
        creator_id = data.get('creator_id')
        reasoning = data.get('creator_reasoning')
        if not isinstance(content_id, str) or not content_id.strip():
            return error('content_id must be a nonblank string', 400)
        if (not isinstance(creator_id, str) or not creator_id.strip()
                or len(creator_id) > config.MAX_CREATOR_ID_CHARS):
            return error('creator_id must be a nonblank string of at most 100 characters', 400)
        if (not isinstance(reasoning, str) or not reasoning.strip()
                or len(reasoning) > config.MAX_REASONING_CHARS):
            return error('creator_reasoning must be a nonblank string of at most 2000 characters', 400)
        try:
            outcome, payload = save_appeal(app.config['DATABASE_PATH'], content_id,
                                           creator_id, reasoning.strip())
        except sqlite3.Error:
            return error('Could not save appeal', 500)
        if outcome == 'missing':
            return error('Content not found', 404)
        if outcome == 'forbidden':
            return error('Creator ID does not match', 403)
        if outcome == 'duplicate':
            return jsonify({'error': 'An active appeal already exists', **payload}), 409
        return jsonify({**payload, 'notice': config.APPEAL_NOTICE})

    @app.get('/log')
    def log():
        raw_limit = request.args.get('limit', '20')
        try:
            limit = int(raw_limit)
        except ValueError:
            return error('limit must be an integer from 1 to 100', 400)
        if not 1 <= limit <= 100 or str(limit) != raw_limit:
            return error('limit must be an integer from 1 to 100', 400)
        try:
            return jsonify({'entries': get_log(app.config['DATABASE_PATH'], limit)})
        except sqlite3.Error:
            return error('Could not read audit log', 500)

    return app


if __name__ == '__main__':
    create_app().run(host='127.0.0.1', port=config.LOCAL_PORT)
