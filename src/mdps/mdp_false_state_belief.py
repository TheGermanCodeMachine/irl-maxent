import numpy as np
from itertools import product

#state 0 to length are the 'wait' states

# this encode the MDP described in: https://www.alignmentforum.org/s/4dHMdK5TLN6xcqtyc/p/cnC2RMWEGiGpJv8go#Time_inconsistency_and_Procrastination

class FalseStateBeliefMDP:
    # start state will be 2. Agent beliefs there is reward R*2 in 0 and R in 3, but state 0 actually doesnt exist.
    def __init__(self, size):
        self.size = size

        self.actions = [0, 1] #-1 => left; +1 => right

        self.n_states = size 
        self.n_actions = len(self.actions)

        self.p_transition = self._transition_prob_table()

    def state_index_to_point(self, state):
        """
        Convert a state index to the coordinate representing it.

        Args:
            state: Integer representing the state.

        Returns:
            The coordinate as tuple of integers representing the same state
            as the index.
        """
        return state

    def state_point_to_index(self, state):
        """
        Convert a state coordinate to the index representing it.

        Note:
            Does not check if coordinates lie outside of the world.

        Args:
            state: Tuple of integers representing the state.

        Returns:
            The index as integer representing the same state as the given
            coordinate.
        """
        return state

    def state_point_to_index_clipped(self, state):
        """
        Convert a state coordinate to the index representing it, while also
        handling coordinates that would lie outside of this world.

        Coordinates that are outside of the world will be clipped to the
        world, i.e. projected onto to the nearest coordinate that lies
        inside this world.

        Useful for handling transitions that could go over an edge.

        Args:
            state: The tuple of integers representing the state.

        Returns:
            The index as integer representing the same state as the given
            coordinate if the coordinate lies inside this world, or the
            index to the closest state that lies inside the world.
        """
        s = max(0, min(state, self.n_states-1))
        return self.state_point_to_index(s)

    def state_index_transition(self, s, a):
        """
        Perform action `a` at state `s` and return the intended next state.

        Does not take into account the transition probabilities. Instead it
        just returns the intended outcome of the given action taken at the
        given state, i.e. the outcome in case the action succeeds.

        Args:
            s: The state at which the action should be taken.
            a: The action that should be taken.

        Returns:
            The next state as implied by the given action and state.
        """
        s = self.state_index_to_point(s)
        if s < self.size:
            if a == 1:
                s += 1
            else:
                s -= 1
        return self.state_point_to_index_clipped(s)

    def _transition_prob_table(self):
        """
        Builds the internal probability transition table.

        Returns:
            The probability transition table of the form

                [state_from, state_to, action]

            containing all transition probabilities. The individual
            transition probabilities are defined by `self._transition_prob'.
        """
        table = np.zeros(shape=(self.n_states, self.n_states, self.n_actions))

        for s1 in range(self.n_states):
            for j, a in enumerate(self.actions):
                s2 = self.state_index_transition(s1, a)
                table[s1, s2, a] = 1

        return table

    def __repr__(self):
        return "GridWorld(size={})".format(self.size)


def state_features(world):
    """
    Return the feature matrix assigning each state with an individual
    feature (i.e. an identity matrix of size n_states * n_states).

    Rows represent individual states, columns the feature entries.

    Args:
        world: A GridWorld instance for which the feature-matrix should be
            computed.

    Returns:
        The coordinate-feature-matrix for the specified world.
    """
    return np.identity(world.n_states)