"""
@date: 2021/9/26
@description: null
"""


class Player:
    def __init__(self):
        self.player_index = None

    def get_player_index(self):
        # for multi player game
        return self.player_index

    def set_player_index(self, player_index):
        # for multi player game
        self.player_index = player_index

    def get_action(self, state, **kwargs):
        raise NotImplementedError

    def reset(self):
        # for self-play player
        pass
