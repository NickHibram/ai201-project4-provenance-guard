"""Fixed, unrounded combination and attribution rules."""

import config


def combine_scores(llm_score, stylometric_score):
    return config.LLM_WEIGHT * llm_score + config.STYLOMETRIC_WEIGHT * stylometric_score


def attribution_for_score(score):
    if not 0 <= score <= 1:
        raise ValueError('score must be in [0, 1]')
    if score <= config.LIKELY_HUMAN_MAX:
        return 'likely_human'
    if score < config.LIKELY_AI_MIN:
        return 'uncertain'
    return 'likely_ai'
