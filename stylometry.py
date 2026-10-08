"""Deterministic four-feature stylometric signal specified in planning.md."""

import math
import re
import statistics

import config


WORD_PATTERN = re.compile(r"[^\W\d_]+(?:'[^\W\d_]+)*", re.UNICODE)
SENTENCE_SPLIT = re.compile(r'[.!?]+(?=\s|$)\s*|\n+')
INTERNAL_PUNCTUATION = frozenset(',;:—–()[]{}')
FUNCTION_WORDS = frozenset('''a an the and or but if while because as until although
of at by for with about against between into through during
before after above below to from up down in out on off over under
i me my we us our you your he him his she her it its they them their
is am are was were be been being have has had do does did
that this these those'''.split())


def tokenize(text):
    return WORD_PATTERN.findall(text.replace('’', "'").replace('‘', "'").lower())


def analysis_units(text):
    analysis_copy = text.replace('\r\n', '\n').replace('\r', '\n')
    units = [(part, tokenize(part)) for part in SENTENCE_SPLIT.split(analysis_copy)]
    return [(part, tokens) for part, tokens in units if tokens]


def coefficient_of_variation(values):
    mean = statistics.fmean(values)
    return statistics.pstdev(values) / mean if mean else None


def clip(value):
    return max(0.0, min(1.0, value))


def regularity_component(cv):
    return 0.90 - 0.80 * clip((cv - 0.10) / 0.70)


def vocabulary_component(ttr_50):
    return 0.90 - 0.80 * clip((ttr_50 - 0.45) / 0.40)


def analyze_text(original_text):
    analysis_copy = original_text.replace('\r\n', '\n').replace('\r', '\n')
    words = tokenize(analysis_copy)
    units = analysis_units(analysis_copy)
    lengths = [len(tokens) for _, tokens in units]
    word_count = len(words)
    sentence_cv = coefficient_of_variation(lengths) if lengths else None
    block_count = word_count // 50
    ttr_50 = (statistics.fmean(len(set(words[i * 50:(i + 1) * 50])) / 50
                                for i in range(block_count)) if block_count else None)
    punctuation_counts = [sum(char in INTERNAL_PUNCTUATION for char in part)
                          for part, _ in units]
    punctuation_rates = [count / length for count, length in zip(punctuation_counts, lengths)]
    function_counts = [sum(token in FUNCTION_WORDS for token in tokens)
                       for _, tokens in units]
    function_rates = [count / length for count, length in zip(function_counts, lengths)]
    punctuation_cv = coefficient_of_variation(punctuation_rates) if punctuation_rates else None
    function_cv = coefficient_of_variation(function_rates) if function_rates else None
    measurements = {
        'word_count': word_count,
        'sentence_count': len(units),
        'sentence_lengths': lengths,
        'sentence_cv': sentence_cv,
        'whole_text_ttr': len(set(words)) / word_count if word_count else None,
        'ttr_50': ttr_50,
        'ttr_words_used': block_count * 50,
        'punctuation_density': sum(punctuation_counts) / word_count if word_count else None,
        'punctuation_rates': punctuation_rates,
        'punctuation_cv': punctuation_cv,
        'function_word_frequency': sum(function_counts) / word_count if word_count else None,
        'function_word_rates': function_rates,
        'function_word_cv': function_cv,
    }
    neutral_reasons = []
    if punctuation_cv is None and units:
        neutral_reasons.append('zero_punctuation_density')
    if function_cv is None and units:
        neutral_reasons.append('zero_function_word_frequency')
    components = {
        'sentence': regularity_component(sentence_cv) if sentence_cv is not None else None,
        'vocabulary': vocabulary_component(ttr_50) if ttr_50 is not None else None,
        'punctuation': (regularity_component(punctuation_cv)
                        if punctuation_cv is not None else (0.5 if units else None)),
        'function_word': (regularity_component(function_cv)
                          if function_cv is not None else (0.5 if units else None)),
    }
    score = (sum(components.values()) / 4
             if all(component is not None for component in components.values()) else None)
    if score is not None and not math.isfinite(score):
        raise ValueError('non-finite stylometric score')
    return {'measurements': measurements, 'components': components,
            'neutral_component_reasons': neutral_reasons, 'stylometric_score': score}
