import os
import tempfile
import unittest
from unittest.mock import patch

from app import create_app
from llm_signal import SignalUnavailable
from scoring import combine_scores, attribution_for_score
from stylometry import analyze_text


class StylometryTests(unittest.TestCase):
    def test_unicode_words_apostrophes_and_sentence_lengths(self):
        result = analyze_text("It's naïve—don't stop. We won't stop! Blue-green skies stay.")
        self.assertEqual(result['measurements']['word_count'], 11)
        self.assertEqual(result['measurements']['sentence_count'], 3)
        self.assertEqual(result['measurements']['sentence_lengths'], [4, 3, 4])

    def test_equal_lengths_no_punctuation_and_zero_function_words(self):
        result = analyze_text('Zebra quartz maple. Violet copper ember. Amber orchid velvet.')
        self.assertEqual(result['measurements']['sentence_cv'], 0)
        self.assertIsNone(result['measurements']['punctuation_cv'])
        self.assertIsNone(result['measurements']['function_word_cv'])
        self.assertEqual(result['components']['punctuation'], 0.5)
        self.assertEqual(result['components']['function_word'], 0.5)
        self.assertIn('zero_punctuation_density', result['neutral_component_reasons'])

    def test_repeated_vocabulary_uses_only_full_fifty_word_blocks(self):
        text = ('echo ' * 50) + ('new ' * 5)
        result = analyze_text(text)
        self.assertEqual(result['measurements']['word_count'], 55)
        self.assertAlmostEqual(result['measurements']['whole_text_ttr'], 2 / 55)
        self.assertEqual(result['measurements']['ttr_50'], 0.02)
        self.assertEqual(result['measurements']['ttr_words_used'], 50)
        self.assertEqual(result['components']['vocabulary'], 0.9)

    def test_varied_lengths_punctuation_and_function_word_rates(self):
        result = analyze_text('The cat, the dog. Zebra. A little bird: in the garden.')
        self.assertGreater(result['measurements']['sentence_cv'], 0)
        self.assertGreater(result['measurements']['punctuation_cv'], 0)
        self.assertGreater(result['measurements']['function_word_cv'], 0)
        self.assertAlmostEqual(result['measurements']['punctuation_density'], 2 / 11)

    def test_anchors_and_interpolation(self):
        from stylometry import regularity_component, vocabulary_component
        self.assertAlmostEqual(regularity_component(0.1), 0.9)
        self.assertAlmostEqual(regularity_component(0.45), 0.5)
        self.assertAlmostEqual(regularity_component(0.8), 0.1)
        self.assertAlmostEqual(vocabulary_component(0.45), 0.9)
        self.assertAlmostEqual(vocabulary_component(0.65), 0.5)
        self.assertAlmostEqual(vocabulary_component(0.85), 0.1)

    def test_line_breaks_quotations_and_code_switched_tokens(self):
        from stylometry import analysis_units, tokenize
        text = 'Dr. Vale said "Wait!"\nPrimera línea—bright morning\nA list item follows.'
        # The abbreviation splits, while the quoted exclamation mark does not.
        self.assertEqual(len(analysis_units(text)), 4)
        self.assertIn('línea', tokenize(text))
        self.assertIn('bright', tokenize(text))


class ScoringTests(unittest.TestCase):
    def test_exact_thresholds(self):
        for score, expected in ((0.3, 'likely_human'), (0.3000001, 'uncertain'),
                                (0.7999999, 'uncertain'), (0.8, 'likely_ai')):
            with self.subTest(score=score):
                self.assertEqual(attribution_for_score(score), expected)

    def test_plan_arithmetic_fixtures(self):
        for llm, style, expected in ((0.10, 0.20, 0.140), (0.74, 0.55, 0.664),
                                     (0.90, 0.65, 0.800), (0.99, 0.89, 0.950),
                                     (0.99, 0.30, 0.714)):
            with self.subTest(llm=llm, style=style):
                self.assertAlmostEqual(combine_scores(llm, style), expected)


class SubmissionPolicyTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.app = create_app({'TESTING': True, 'DATABASE_PATH': os.path.join(self.temp.name, 'db.sqlite3'),
                               'GROQ_MODEL': 'test-model', 'RATELIMIT_ENABLED': False})
        self.client = self.app.test_client()

    def tearDown(self):
        self.temp.cleanup()

    def test_minimum_word_and_sentence_boundaries_skip_groq(self):
        cases = [('word ' * 47 + '. A. B.', 'too_few_words'),
                 ('word ' * 50 + '. More words.', 'too_few_sentences')]
        for text, reason in cases:
            with self.subTest(reason=reason), patch('app.assess_with_llm') as external:
                response = self.client.post('/submit', json={'text': text, 'creator_id': 'author'})
                self.assertEqual(response.json['analysis_status'], 'insufficient_text')
                self.assertEqual(response.json['confidence'], 0.5)
                self.assertIsNone(response.json['raw_combined_score'])
                self.assertIn(reason, response.json['uncertainty_reasons'])
                external.assert_not_called()

    def test_fifty_words_three_sentences_and_missing_llm(self):
        text = ('word ' * 17 + '. ') * 2 + ('word ' * 16 + '.')
        with patch('app.assess_with_llm', side_effect=SignalUnavailable('outage')):
            response = self.client.post('/submit', json={'text': text, 'creator_id': 'author'})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json['analysis_status'], 'signal_unavailable')
        self.assertIsNone(response.json['signals']['llm_score'])
        self.assertIsNotNone(response.json['signals']['stylometric_score'])
        self.assertEqual(response.json['confidence'], 0.5)

    def test_stylometry_failure_on_short_text_still_skips_groq(self):
        with patch('app.analyze_text', side_effect=RuntimeError('style failed')):
            with patch('app.assess_with_llm') as external:
                response = self.client.post('/submit', json={
                    'text': 'word ' * 49 + '. Another.', 'creator_id': 'author'})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json['analysis_status'], 'insufficient_text')
        self.assertIn('too_few_sentences', response.json['uncertainty_reasons'])
        external.assert_not_called()

    def test_fifty_word_run_on_abstains_for_too_few_sentences(self):
        with patch('app.assess_with_llm') as external:
            response = self.client.post('/submit', json={
                'text': 'word ' * 50, 'creator_id': 'author'})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json['analysis_status'], 'insufficient_text')
        self.assertIn('too_few_sentences', response.json['uncertainty_reasons'])
        external.assert_not_called()


if __name__ == '__main__':
    unittest.main()
