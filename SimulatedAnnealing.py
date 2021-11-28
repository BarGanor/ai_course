from q1 import *


class SimulatedAnnealing:
    def __init__(self, starting_board, goal_board):
        self.start_node = Node(None, starting_board)
        self.end_node = Node(None, goal_board)
        self.initial_temp = 100
        self.final_temp = .1
        self.alpha = 0.01

    def search(self):
        current_temp = self.initial_temp
        current_state = self.start_node
        solution = current_state

        while current_temp:
            pass

class HillClimbing:
    def __init__(self, starting_board, goal_board):
        self.start_node = Node(None, starting_board)
        self.end_node = Node(None, goal_board)

        self.start_node.h = self.start_node.f = self.start_node.get_heuristic(self.end_node)

        self.restarts = 0
        self.tries = 0
        self.max_tries = 300

    def search(self):

        while self.restarts < 5:
            current_node = self.start_node

            while self.tries < self.max_tries:
                agents = current_node.get_agents()
                agents_children = self.get_children(current_node, agents)
                highest_valued = self.get_highest_valued_successor(agents_children)
                current_node.print(True)
                if highest_valued.h <= current_node.h:
                    current_node = highest_valued

                if current_node == self.end_node:
                    return current_node.return_path()

    def get_children(self, current_node, agents):
        children = []
        for agent in agents:

            moves = agent.get_moves()
            for new_position in moves:
                new_node = Node(current_node, new_position)
                new_node.h = new_node.get_heuristic(self.end_node)
                children.append(new_node)

        return children

    def get_highest_valued_successor(self, moves):
        highest_valued = None
        for move in moves:
            if highest_valued is None or move.h < highest_valued.h:
                highest_valued = move

            else:
                pass
        return highest_valued
