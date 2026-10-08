import json
import os
import sqlite3
import tempfile
import unittest
from unittest.mock import patch

from app import create_app
from llm_signal import SignalUnavailable


ELIGIBLE = (('The rain crossed our window and settled over the garden, where every leaf held a silver edge. '
             'I waited beside the lamp and remembered how my grandmother described this street before the houses changed. '
             'By morning the clouds had moved away, and the path looked ordinary again, but I kept the memory of that quiet hour.'))


class AppealsAndAuditTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.db_path = os.path.join(self.temp.name, 'db.sqlite3')
        self.options = {'TESTING': True, 'DATABASE_PATH': self.db_path,
                        'GROQ_MODEL': 'test-model', 'RATELIMIT_ENABLED': False}
        self.app = create_app(self.options)
        self.client = self.app.test_client()

    def tearDown(self):
        self.temp.cleanup()

    def submit(self, creator='author'):
        with patch('app.assess_with_llm', return_value={
            'ai_score': 0.9, 'reasoning': 'Regular style, but origin is uncertain.',
            'model': 'test-model', 'prompt_version': 'v1', 'temperature': 0.1}):
            return self.client.post('/submit', json={'text': ELIGIBLE, 'creator_id': creator})

    def test_exact_labels_and_appeal_preserves_decision(self):
        result = self.submit()
        self.assertEqual(result.status_code, 200)
        self.assertIn(result.json['attribution'], ('likely_ai', 'uncertain', 'likely_human'))
        self.assertIn(result.json['label'], (
            'Likely AI-generated. This text shows strong indicators of AI generation, but automated detection can be wrong. The creator may appeal this result.',
            'Likely human-written. This text shows strong indicators of human authorship, though automated analysis cannot verify authorship with certainty.',
            'Uncertain. Our analysis does not provide enough evidence to confidently determine whether this text is human-written or AI-generated.'))
        content_id = result.json['content_id']
        appeal = self.client.post('/appeal', json={'content_id': content_id,
                               'creator_id': 'author', 'creator_reasoning': 'The repeated style was intentional.'})
        self.assertEqual(appeal.status_code, 200)
        self.assertEqual(appeal.json['status'], 'under_review')
        self.assertEqual(appeal.json['notice'],
                         'Under review. The creator has appealed this assessment; it has not been resolved.')
        with sqlite3.connect(self.db_path) as connection:
            stored = connection.execute('SELECT attribution, confidence, label, status FROM content WHERE content_id=?',
                                        (content_id,)).fetchone()
            reason = connection.execute('SELECT appeal_reasoning FROM appeals WHERE content_id=?',
                                        (content_id,)).fetchone()[0]
        self.assertEqual(stored, (result.json['attribution'], result.json['confidence'],
                                  result.json['label'], 'under_review'))
        self.assertEqual(reason, 'The repeated style was intentional.')
        events = self.client.get('/log').json['entries']
        self.assertEqual([event['event_type'] for event in events], ['appeal', 'decision'])
        self.assertEqual(events[0]['original_decision_event_id'], events[1]['event_id'])
        self.assertEqual(events[0]['appeal_reasoning'], reason)
        self.assertEqual(events[0]['attribution'], result.json['attribution'])
        self.assertEqual(events[0]['confidence'], result.json['confidence'])

    def test_appeal_failures_do_not_mutate_history(self):
        content_id = self.submit().json['content_id']
        self.assertEqual(self.client.post('/appeal', json={'content_id': 'missing', 'creator_id': 'author',
                                                          'creator_reasoning': 'Why?'}).status_code, 404)
        self.assertEqual(self.client.post('/appeal', json={'content_id': content_id, 'creator_id': 'stranger',
                                                          'creator_reasoning': 'Why?'}).status_code, 403)
        for reasoning in ('', ' ' * 3, 'x' * 2001, 42):
            self.assertEqual(self.client.post('/appeal', json={'content_id': content_id,
                             'creator_id': 'author', 'creator_reasoning': reasoning}).status_code, 400)
        valid = {'content_id': content_id, 'creator_id': 'author', 'creator_reasoning': 'I wrote this.'}
        first = self.client.post('/appeal', json=valid)
        self.assertEqual(first.status_code, 200)
        duplicate = self.client.post('/appeal', json=valid)
        self.assertEqual(duplicate.status_code, 409)
        self.assertEqual(duplicate.json['appeal_id'], first.json['appeal_id'])
        self.assertEqual(len(self.client.get('/log').json['entries']), 2)

    def test_older_appeal_event_displays_original_decision_fields(self):
        content_id = self.submit().json['content_id']
        self.client.post('/appeal', json={'content_id': content_id,
            'creator_id': 'author', 'creator_reasoning': 'Please review.'})
        with sqlite3.connect(self.db_path) as connection:
            row = connection.execute("SELECT event_id, payload_json FROM audit_events WHERE event_type='appeal'").fetchone()
            old_payload = json.loads(row[1])
            old_payload.pop('attribution')
            old_payload.pop('confidence')
            connection.execute('UPDATE audit_events SET payload_json=? WHERE event_id=?',
                               (json.dumps(old_payload), row[0]))
        appeal_event = self.client.get('/log?limit=1').json['entries'][0]
        decision_event = self.client.get('/log?limit=2').json['entries'][1]
        self.assertEqual(appeal_event['attribution'], decision_event['attribution'])
        self.assertEqual(appeal_event['confidence'], decision_event['confidence'])

    def test_persistence_after_new_app_instance(self):
        content_id = self.submit().json['content_id']
        new_client = create_app(self.options).test_client()
        self.assertEqual(new_client.get('/log').json['entries'][0]['content_id'], content_id)
        self.assertEqual(new_client.post('/appeal', json={'content_id': content_id,
                         'creator_id': 'author', 'creator_reasoning': 'Please review.'}).status_code, 200)

    def test_request_size_and_wordless_input(self):
        self.assertEqual(self.client.post('/submit', json={'text': '123 !!!', 'creator_id': 'a'}).status_code, 400)
        self.assertEqual(self.client.post('/submit', data=b'x' * 65537,
                                          content_type='application/json').status_code, 413)

    def test_appeal_audit_failure_rolls_back_reason_and_status(self):
        content_id = self.submit().json['content_id']
        with sqlite3.connect(self.db_path) as connection:
            connection.execute('''CREATE TRIGGER reject_appeal_audit BEFORE INSERT ON audit_events
                WHEN NEW.event_type='appeal' BEGIN SELECT RAISE(ABORT, 'blocked'); END''')
        response = self.client.post('/appeal', json={'content_id': content_id,
            'creator_id': 'author', 'creator_reasoning': 'Please review my draft.'})
        self.assertEqual(response.status_code, 500)
        with sqlite3.connect(self.db_path) as connection:
            status = connection.execute('SELECT status FROM content WHERE content_id=?', (content_id,)).fetchone()[0]
            appeals = connection.execute('SELECT COUNT(*) FROM appeals').fetchone()[0]
        self.assertEqual(status, 'classified')
        self.assertEqual(appeals, 0)

    def test_decision_audit_failure_rolls_back_content(self):
        with sqlite3.connect(self.db_path) as connection:
            connection.execute('''CREATE TRIGGER reject_decision_audit BEFORE INSERT ON audit_events
                WHEN NEW.event_type='decision' BEGIN SELECT RAISE(ABORT, 'blocked'); END''')
        response = self.submit()
        self.assertEqual(response.status_code, 500)
        with sqlite3.connect(self.db_path) as connection:
            content_count = connection.execute('SELECT COUNT(*) FROM content').fetchone()[0]
            event_count = connection.execute('SELECT COUNT(*) FROM audit_events').fetchone()[0]
        self.assertEqual((content_count, event_count), (0, 0))

    def test_decision_event_contains_signal_and_configuration_details(self):
        result = self.submit()
        event = self.client.get('/log?limit=1').json['entries'][0]
        self.assertEqual(event['content_id'], result.json['content_id'])
        self.assertEqual(event['creator_id'], 'author')
        self.assertEqual(event['event_type'], 'decision')
        self.assertEqual(event['confidence_semantics'], 'ai_evidence_index_not_probability')
        self.assertEqual(event['llm']['model'], 'test-model')
        self.assertEqual(event['llm']['prompt_version'], 'v1')
        self.assertEqual(event['score_config_version'], 'v1')
        self.assertEqual(set(event['stylometry']['components']),
                         {'sentence', 'vocabulary', 'punctuation', 'function_word'})
        self.assertEqual(event['stylometry']['measurements']['word_count'], 56)
        self.assertEqual(event['status'], 'classified')

    def test_all_three_labels_and_appeals_are_reachable_with_stubbed_signals(self):
        from stylometry import analyze_text
        from config import LABELS
        for score, attribution in ((0.1, 'likely_human'), (0.5, 'uncertain'),
                                   (0.9, 'likely_ai')):
            with self.subTest(attribution=attribution):
                style = analyze_text(ELIGIBLE)
                style['stylometric_score'] = score
                llm = {'ai_score': score, 'reasoning': 'Stubbed for policy testing.',
                       'model': 'test-model', 'prompt_version': 'v1', 'temperature': 0.1}
                with patch('app.analyze_text', return_value=style):
                    with patch('app.assess_with_llm', return_value=llm):
                        result = self.client.post('/submit', json={
                            'text': ELIGIBLE, 'creator_id': 'author'})
                self.assertEqual(result.status_code, 200)
                self.assertEqual(result.json['attribution'], attribution)
                self.assertEqual(result.json['label'], LABELS[attribution])
                appeal = self.client.post('/appeal', json={
                    'content_id': result.json['content_id'], 'creator_id': 'author',
                    'creator_reasoning': 'Please review the assessment.'})
                self.assertEqual(appeal.status_code, 200)


class RateLimitTests(unittest.TestCase):
    def test_fourth_submit_within_five_minutes_is_429_before_provider(self):
        with tempfile.TemporaryDirectory() as folder:
            app = create_app({'TESTING': True, 'DATABASE_PATH': os.path.join(folder, 'db.sqlite3'),
                              'GROQ_MODEL': 'test-model'})
            client = app.test_client()
            with patch('app.assess_with_llm') as external:
                for _ in range(3):
                    self.assertEqual(client.post('/submit', json={'text': '', 'creator_id': 'a'}).status_code, 400)
                blocked = client.post('/submit', json={'text': ELIGIBLE, 'creator_id': 'a'})
            self.assertEqual(blocked.status_code, 429)
            self.assertEqual(blocked.json['error'], 'Submission rate limit exceeded')
            self.assertGreaterEqual(int(blocked.headers['Retry-After']), 240)
            external.assert_not_called()

    def test_daily_limit_reports_daily_retry_interval(self):
        with tempfile.TemporaryDirectory() as folder:
            with patch('app.config.SUBMISSION_RATE_LIMIT', '1000 per minute;2 per day'):
                app = create_app({'TESTING': True,
                                  'DATABASE_PATH': os.path.join(folder, 'db.sqlite3')})
            client = app.test_client()
            for _ in range(2):
                self.assertEqual(client.post('/submit', json={
                    'text': '', 'creator_id': 'a'}).status_code, 400)
            blocked = client.post('/submit', json={'text': '', 'creator_id': 'a'})
            self.assertEqual(blocked.status_code, 429)
            self.assertGreater(int(blocked.headers['Retry-After']), 60)


if __name__ == '__main__':
    unittest.main()
