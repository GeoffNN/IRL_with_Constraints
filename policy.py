class Policy:

    def __init__(self, rec_arr):
        self.recommendations = rec_arr

    def next_action(self, state):
        return self.recommendations(state)
