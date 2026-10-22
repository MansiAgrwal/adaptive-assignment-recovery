"""Exact illustrative counterexample, not a fitted model or novelty claim.

Two equally prevalent learner types have fixed learning probabilities.
All begin unmastered; there is no loss. Perfectly observed competence is used
to show that measurement error is not needed for the aggregation problem.
"""
from fractions import Fraction as F
from pathlib import Path
import json


def conditional_fast_weight(sequence):
    gains = {'A': (F(2, 5), F(1, 20)), 'B': (F(1, 4), F(1, 10))}
    fast = slow = F(1, 2)
    for activity in sequence:
        gf, gs = gains[activity]
        fast *= 1 - gf
        slow *= 1 - gs
    return fast / (fast + slow)


def record(sequence):
    w = conditional_fast_weight(sequence)
    ga = w * F(2, 5) + (1-w) * F(1, 20)
    gb = w * F(1, 4) + (1-w) * F(1, 10)
    return {'previous_activities': sequence or 'none',
            'condition': 'competence remains unmastered after all previous activities',
            'fast_type_fraction': float(w), 'next_A_gain': float(ga),
            'next_B_gain': float(gb), 'B_minus_A_gain': float(gb-ga)}


if __name__ == '__main__':
    rows = [record(x) for x in ['', 'A', 'AA', 'BB', 'AAAA']]
    assert rows[0]['next_A_gain'] == .225
    assert rows[2]['next_A_gain'] != rows[3]['next_A_gain']
    assert rows[0]['B_minus_A_gain'] < 0 < rows[4]['B_minus_A_gain']
    result = {'scope': __doc__,
              'input_A_gains_fast_slow': [.4, .05],
              'input_B_gains_fast_slow': [.25, .10],
              'results': rows,
              'conclusion': 'At the same time and competence state, different prior activity histories give different marginal next-activity gains. A constant homogeneous competence kernel does not generally represent the mixture. This example does not show that every mixture is non-Markov, or that a correctly specified heterogeneous model is unidentified.'}
    Path(__file__).with_suffix('.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
