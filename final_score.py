#!/usr/bin/env python3

import numpy as np


def baseline_points(n):
    return 2 * n


def standing_points(r, N):
    return 7 - 7 * np.log2((r-1) / (N-1) + 1)


def team_points(one_person):
    return int(one_person)


def final_score(standing1, standing2, baselines, team):
    return (standing1 + standing2 * 2) / 3 + baselines + team


if __name__ == '__main__':
    NUM_TEAMS = 56
    b = baseline_points(10)
    print('Your baseline score is {0}'.format(b))
    t = team_points(one_person=False)
    print('Your team score is {0}'.format(t))
    s1 = standing_points(5, NUM_TEAMS)
    print('Your standing score at the FIRST deadline is {0:.2f}'.format(s1))
    s2 = standing_points(15, NUM_TEAMS)
    print('Your standing score at the SECOND deadline is {0:.2f}'.format(s2))
    final = final_score(s1, s2, b, t)
    print('Your final score is {0:.2f}'.format(final))
