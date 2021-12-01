import numpy as np
from scipy.spatial.distance import cdist

from AStar import AStar
import random

HillClimbing=None
LocalBeam = None


class BoardSearch:
    def __init__(self, starting_board, goal_board, search_method, detail_output):
        self.method_dict = {1: AStar, 2: HillClimbing, 3: SimulatedAnnealing, 4: LocalBeam}
        self.starting_board = np.array(starting_board)
        self.goal_board = np.array(goal_board)
        self.search_method = self.method_dict.get(search_method)
        self.detail_output = detail_output

    def find_path(self):

        if self.less_agents_then_needed():
            return 'No possible path'

        if self.search_method == AStar:
            a_star = AStar(starting_board=self.starting_board, goal_board=self.goal_board, cost=1)
            path = a_star.search()
        #
        # elif self.search_method == HillClimbing:
        #     hill_climbing = HillClimbing(starting_board=self.starting_board, goal_board=goal_board)
        #     path = hill_climbing.search()

        elif self.search_method == SimulatedAnnealing:
            simulated_annealing = SimulatedAnnealing(starting_board=self.starting_board, goal_board=goal_board)
            path = simulated_annealing.search()

        # elif self.search_method == LocalBeam:
        #     local_beam = LocalBeam(starting_board=self.starting_board, goal_board=self.goal_board)
        #     path = local_beam.search()

        else:
            path = None

        self.print_path(path)

    def print_path(self, path):
        if path is None:
            print('No path found')

        else:
            for i in range(len(path)):
                if i == 0:
                    print('Board 1 (starting position):')
                    path[i].print(detail_output=False)

                elif i == len(path) - 1:
                    print('Board ' + str(i + 1) + '(goal position):')
                    path[i].print(detail_output=self.detail_output, last=True)
                else:
                    print('Board ' + str(i + 1) + ':')
                    path[i].print(detail_output=self.detail_output)

    def less_agents_then_needed(self):
        return (self.starting_board == 2).sum() < (self.goal_board == 2).sum()


class Node:
    def __init__(self, parent=None, position=None):
        self.parent = parent

        if type(position) != np.ndarray:
            self.position = np.array(position)
        else:
            self.position = position

        self.g = 0
        self.h = 0
        self.f = 0

    def __eq__(self, other):
        if other is None:
            return False
        comparison = self.position == other.position

        return comparison.all()

    def get_agents(self):
        position = self.position
        agents = []

        for row in range(position.shape[0]):
            for col in range(position.shape[1]):

                # Get all agents best move
                if position[row][col] == 2:
                    agent = Agent(self, col=col, row=row)
                    agents.append(agent)
        return agents

    def print(self, detail_output=False, last=False):
        transform_dict = {0: ' ', 1: '@', 2: '*'}
        print('  1 2 3 4 5 6')
        for row in range(len(self.position)):
            string_ints = [transform_dict[num] for num in self.position[row]]

            print(str(row + 1) + ':' + ' '.join(string_ints))

        if self.h is not None and detail_output:
            print('Heuristic:' + str(self.h))

        if not last:
            print('-' * 5)

    def return_path(self):
        path = []
        current = self

        while current is not None:
            path.append(current)
            current = current.parent

        return path[::-1]

    def change_board(self, agent_row, agent_col, wanted_row, wanted_col):
        changed_board = self.position.copy()
        changed_board[agent_row][[agent_col]] = 0

        if (5 < wanted_row) or (wanted_row < 0) or (5 < wanted_col) or (wanted_col < 0) and self.h == 100:
            return changed_board

        elif changed_board[wanted_row][wanted_col] == 0:
            changed_board[wanted_row][wanted_col] = 2
            return changed_board

        else:
            return None

    def get_coordinates(self):
        coordinates = np.where(self.position == 2)
        return list(zip(coordinates[0], coordinates[1]))

    def get_heuristic(self, end_node):
        curr_coordinates = self.get_coordinates()
        goal_coordinates = end_node.get_coordinates()

        if len(curr_coordinates) == len(goal_coordinates):
            heuristic = self.get_min_sum_of_distance(curr_coordinates, goal_coordinates)
        else:
            heuristic = 100

        return heuristic


    @staticmethod
    def get_min_sum_of_distance(curr, goal):
        curr_clean = [x for x in curr if x not in goal]
        goal_clean = [x for x in goal if x not in curr]

        if len(curr_clean) == 0:
            return 0

        distances = cdist(np.array(curr_clean), np.array(goal_clean))

        min_dist = sum(min(distances, key=min))

        return float(min_dist)


class Agent:
    def __init__(self, node, col, row):
        self.node = node
        self.col = col
        self.row = row

    def get_moves(self):
        move_left = Move(self.node, self.col, self.row, [0, -1])
        move_right = Move(self.node, self.col, self.row, [0, 1])
        move_up = Move(self.node, self.col, self.row, [1, 0])
        move_down = Move(self.node, self.col, self.row, [-1, 0])

        moves = [move_left.get_move(),
                 move_right.get_move(),
                 move_up.get_move(),
                 move_down.get_move()
                 ]

        moves = [move for move in moves if move is not None]
        return moves


class Move(Agent):
    def __init__(self, node: Node, col, row, direction):
        super().__init__(node, col, row)
        self.node = node
        self.row = row
        self.col = col
        self.direction = direction

    def get_move(self):
        wanted_row = self.row + self.direction[0]
        wanted_col = self.col + self.direction[1]
        move = self.node.change_board(self.row, self.col, wanted_row=wanted_row, wanted_col=wanted_col)

        return move



import numpy as np
import random

class SimulatedAnnealing:
    def __init__(self, starting_board, goal_board):
        self.start_node = Node(None, starting_board)
        self.end_node = Node(None, goal_board)
        self.start_node.h = self.start_node.get_heuristic(self.end_node)
        self.initial_temp = 100
        self.final_temp = .1
        self.alpha = 0.01

    def search(self):
        current_state = self.start_node

        for t in range(100):

            if current_state == self.end_node:
                return current_state.return_path()

            current_temp = self.schedule(t)

            if current_temp == 0:
                return current_state.return_path()

            step_chosen = False
            steps_considered = []
            while not step_chosen:
                next_step = self.get_random_successor(current_state)
                delta_e = current_state.h - next_step.h

                if delta_e > 0:
                    current_state = next_step

                elif next_step not in steps_considered:
                    prob = np.exp(delta_e / current_temp)
                    current_state = random.choices(population=[current_state, next_step], weights=[1 - prob, prob], k=1)[0]

                if current_state == next_step:
                    step_chosen = True

                else:
                    steps_considered.append(next_step)


    def schedule(self, t):
        return min(pow((self.initial_temp - t), (1/10)), 1)

    def get_random_successor(self, current):
        agents = current.get_agents()
        moves = self.get_children(current_node=current, agents=agents)
        random_move = random.choice(moves)

        return random_move

    def get_children(self, current_node, agents):
        children = []
        for agent in agents:

            moves = agent.get_moves()
            for new_position in moves:
                new_node = Node(current_node, new_position)
                new_node.h = new_node.get_heuristic(self.end_node)
                children.append(new_node)
        return children


def find_path(starting_board, goal_board, search_method, detail_output):
    board_search = BoardSearch(starting_board, goal_board, search_method, detail_output)
    return board_search.find_path()


starting_board = [[2, 0, 2, 0, 2, 0],
                  [0, 0, 0, 2, 1, 2],
                  [1, 0, 0, 0, 0, 0],
                  [0, 0, 1, 0, 1, 0],
                  [2, 0, 0, 0, 0, 0],
                  [0, 1, 0, 0, 0, 0]]
goal_board = [[2, 0, 2, 0, 0, 0],
              [0, 0, 0, 2, 1, 2],
              [1, 0, 0, 0, 0, 2],
              [0, 0, 1, 0, 1, 0],
              [0, 0, 0, 0, 0, 0],
              [0, 1, 0, 0, 0, 0]]

find_path(starting_board, goal_board, 4, True)
