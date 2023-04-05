"""
@author: lxy
@email: linxy59@mail2.sysu.edu.cn
@date: 2021/9/26
@description: null
"""
from typing import Tuple

from toolbox.game.Player import Player


class DoublePlayerGameEnv:
    def __init__(self):
        self.players = [1, 2]  # player1 and player2
        self.current_player: int = 1

    def reset(self, start_player=0):
        self.current_player = self.players[start_player]  # start player
        # return state

    def step(self, action):
        # return state, reward, done, info
        pass

    def game_end(self) -> Tuple[bool, int]:
        # return bool, player_id
        # if player 1 win, then return True, 1
        # if no one win, then return False, -1
        return False, -1

    def render(self, player1, player2):
        pass

    def get_current_player(self):
        return self.current_player


class GameManager:
    """game manager"""

    def start_play(self, player1: Player, player2: Player, start_player=0, is_shown=True):
        """start a game between two players"""
        pass

    def start_self_play(self, player: Player, is_shown=False):
        """ start a self-play game using a MCTS player, reuse the search tree,
        and store the self-play data: (state, mcts_probs, z) for training
        """
        pass
