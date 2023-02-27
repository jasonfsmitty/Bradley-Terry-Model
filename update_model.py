#!/usr/bin/env python3
import argparse
import logging
import numpy as np
import os
import pandas as pd
import time
import json

from datetime import datetime
from collections import Counter

logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    level=logging.INFO)

DUMMY_PLAYER = 'DUMMY PLAYER'

#----------------------------------------------------------------------------
# Read AHF score data into a ledger. This is specific to the JSON format
# provided by the gamesheetstats.com API. To use this script with a different
# data source, you would replace this reader with one specific to your context.
class AhfScoreReader:
    def __init__(self):
        pass

    #----------------------------------------------------------------------------
    def read(self, filename):
        ledger = {
            'Date'     : [],
            'Player A' : [],
            'Player B' : [],
            'Wins A'   : [],
            'Wins B'   : [],
        }
        with open(filename) as f:
            jdata = json.load(f)
            for game in jdata:
                self.readGame(ledger, game['game'])
        return ledger

    #----------------------------------------------------------------------------
    def readGame(self, ledger, game):
        date       = datetime.strptime(game['date'], "%b %d, %Y").date()
        team1      = game['homeTeam']['name']
        team1score = game['finalScore']['homeGoals']
        team2      = game['visitorTeam']['name']
        team2score = game['finalScore']['visitorGoals']
        shootout   = any(map(lambda x: x['title'] == 'SO', game['scoresByPeriod']))

        if team1score == team2score:
            raise RuntimeError("Does not support ties")

        winner = team1 if team1score > team2score else team2
        loser  = team2 if team1score > team2score else team1

        ledger['Date'].append(str(date.isoformat()))
        ledger['Player A'].append(loser)
        ledger['Player B'].append(winner)
        ledger['Wins A'].append(0)
        ledger['Wins B'].append(1)


def extract_game_data(inputFile):
    df = pd.DataFrame(AhfScoreReader().read(inputFile))

    assert all(c in df.columns for c in ['Date', 'Player A', 'Player B', 'Wins A', 'Wins B']), \
        'Expecting columns Date, Player A, Player B, Wins A, Wins B'

    #df['Date'] = df['Date'].astype(datetime)
    df['Wins A'] = df['Wins A'].astype(int)
    df['Wins B'] = df['Wins B'].astype(int)

    return df

def add_dummy_games(game_data, alpha=1):
    ''' Regularizes the estimate by adding games against a dummy player.

        :param alpha: regularization parameter, number dummy wins/loses to add
    '''
    players = sorted(list(set(game_data['Player A']) | set(game_data['Player B'])))

    # Add dummy games
    dummy_data = [[datetime(2000, 1, 1), p, DUMMY_PLAYER, alpha, alpha] for p in players]
    df = pd.DataFrame(dummy_data, columns=game_data.columns)
    df = pd.concat([game_data, df])
    df

    return df

def compute_rank_scores(game_data, max_iters=200, error_tol=0.00001):
    ''' Computes Bradley-Terry using iterative algorithm

        See: https://en.wikipedia.org/wiki/Bradley%E2%80%93Terry_model
    '''
    # Do some aggregations for convenience
    # Total wins per player
    #winsA = game_data.groupby('Player A').agg(sum)['Wins A'].reset_index()
    winsA = game_data.groupby('Player A')[['Wins A']].agg(sum).reset_index()
    winsA = winsA[winsA['Wins A'] > 0]
    winsA.columns = ['Player', 'Wins']
    #winsB = game_data.groupby('Player B').agg(sum)['Wins B'].reset_index()
    winsB = game_data.groupby('Player B')[['Wins B']].agg(sum).reset_index()
    winsB = winsB[winsB['Wins B'] > 0]
    winsB.columns = ['Player', 'Wins']
    wins = pd.concat([winsA, winsB]).groupby('Player').agg(sum)['Wins']

    # Total games played between pairs
    num_games = Counter()
    for index, row in game_data.iterrows():
        key = tuple(sorted([row['Player A'], row['Player B']]))
        total = sum([row['Wins A'], row['Wins B']])
        num_games[key] += total

    # Iteratively update 'ranks' scores
    players = sorted(list(set(game_data['Player A']) | set(game_data['Player B'])))
    ranks = pd.Series(np.ones(len(players)) / len(players), index=players)
    for iters in range(max_iters):
        oldranks = ranks.copy()
        for player in ranks.index:
            denom = np.sum(np.fromiter((num_games[tuple(sorted([player, p]))]
                           / (ranks[p] + ranks[player])
                           for p in ranks.index if p != player), dtype=float))
            ranks[player] = 1.0 * wins[player] / denom

        ranks /= sum(ranks)

        if np.sum((ranks - oldranks).abs()) < error_tol:
            break

    if np.sum((ranks - oldranks).abs()) < error_tol:
        logging.debug(" * Converged after %d iterations.", iters)
    else:
        logging.debug(" * Max iterations reached (%d iters).", max_iters)

    del ranks[DUMMY_PLAYER]

    # Scale logarithm of score to be between 1 and 1000
    ranks = ranks.sort_values(ascending=False) \
                 .apply(lambda x: x * 10000.0 + 0.5) \
                 .astype(int) \
                 .clip(1)

    return ranks


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script to update Google Sheet with ranking model")
    parser.add_argument('--alpha', default=0.85, type=float, help='Regularization parameter')
    parser.add_argument('--debug', action='store_true', help="Enable debug logging")
    parser.add_argument('input', help="JSON file containing score data to analyze")
    args = parser.parse_args()

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    logging.debug("Extracting game data from '%s'...", args.input)
    game_data = extract_game_data(args.input)
    logging.debug(" * Found %d rows.", len(game_data))

    logging.debug("Adding dummy game for regularization (alpha={})...".format(args.alpha))
    game_data = add_dummy_games(game_data, args.alpha)

    logging.debug("Computing rank scores...")
    ranks = compute_rank_scores(game_data)

    # TODO: print
    for rank in ranks.items():
        print("{:<30} : {:>5}".format(rank[0], rank[1]))

