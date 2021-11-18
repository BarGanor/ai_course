import numpy as np
from scipy.spatial.distance import cdist


def get_min_sum_of_distance(curr, goal):
    curr_clean = [x for x in curr if x not in goal]
    goal_clean = [x for x in goal if x not in curr]

    if len(curr_clean) == 0:
        return 0

    distances = cdist(np.array(curr_clean), np.array(goal_clean))

    min_dist = sum(min(distances, key=min))

    return float(min_dist)


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

    def print(self, detail_output):

        transform_dict = {0: ' ', 1: '@', 2: '*'}
        print('  1 2 3 4 5 6')
        for row in range(len(self.position)):
            string_ints = [transform_dict[num] for num in self.position[row]]

            print(str(row + 1) + ':' + ' '.join(string_ints))

        if self.h is not None and detail_output:
            print('Heuristic:' + str(self.h))
        print('-' * 5)

    def return_path(self):
        path = []

        current = self

        while current is not None:
            path.append(current)
            current = current.parent

        # Return reversed so its start to end
        path = path[::-1]

        return path

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
            heuristic = get_min_sum_of_distance(curr_coordinates, goal_coordinates)
        else:
            heuristic = 100

        return heuristic


class Agent:
    def __init__(self, node, col, row):
        self.node = node
        self.col = col
        self.row = row

    def get_moves(self):
        moves = Moves(self.node, self.col, self.row)

        return moves.get_moves()


class Moves(Agent):
    def __init__(self, node: Node, col, row):
        super().__init__(node, col, row)
        self.node = node
        self.row = row
        self.col = col

    def move_left(self):
        try:
            move_left = self.node.change_board(self.row, self.col, wanted_row=self.row, wanted_col=self.col - 1)
            return move_left
        except:
            pass

    def move_right(self):
        try:
            move_right = self.node.change_board(self.row, self.col, wanted_row=self.row, wanted_col=self.col + 1)
            return move_right
        except:
            pass

    def move_up(self):
        try:
            move_up = self.node.change_board(self.row, self.col, wanted_row=self.row + 1, wanted_col=self.col)
            return move_up
        except:
            pass

    def move_down(self):
        try:
            move_down = self.node.change_board(self.row, self.col, wanted_row=self.row - 1, wanted_col=self.col)
            return move_down
        except:
            pass

    def get_moves(self):

        moves = [self.move_left(), self.move_right(), self.move_up(), self.move_down()]

        return [move for move in moves if move is not None]


class BoardSearch:
    def __init__(self, starting_board, goal_board, search_method, detail_output):
        self.method_dict = {1: AStar}
        self.starting_board = np.array(starting_board)
        self.goal_board = np.array(goal_board)
        self.search_method = self.method_dict.get(search_method)
        self.detail_output = detail_output

    def find_path(self):

        if self.less_agents_then_needed():
            return 'No Possible Path'

        if self.search_method == AStar:
            a_star = AStar(starting_board=starting_board, goal_board=goal_board, cost=1)
            path = a_star.search()
            if path is None:
                print('No path was found')

            else:
                print('Path Found')
                for step in path:
                    step.print(detail_output=self.detail_output)
                print('Number of steps: ' + str(len(path)))

    def less_agents_then_needed(self):
        return (self.starting_board == 2).sum() < (self.goal_board == 2).sum()


class AStar:
    def __init__(self, starting_board, goal_board, cost):
        self.start = starting_board
        self.end = goal_board
        self.cost = cost
        self.end_node = Node(None, goal_board)
        self.yet_to_visit = []
        self.visited = []
        self.tries = 0
        self.max_tries = 30
        self.max_tries_exceeded = self.tries > self.max_tries

    def search(self):
        cost = self.cost
        end_node = self.end_node
        yet_to_visit = self.yet_to_visit
        visited = self.visited

        start_node = Node(None, starting_board)
        start_node.h = start_node.f = start_node.get_heuristic(end_node)

        yet_to_visit.append(start_node)

        while len(yet_to_visit) > 0:
            self.tries += 1

            current_node = yet_to_visit[0]
            current_index = 0

            for index, item in enumerate(yet_to_visit):
                if item.f < current_node.f:
                    current_node = item
                    current_index = index
            yet_to_visit.pop(current_index)
            visited.append(current_node)

            if current_node == end_node:
                return current_node.return_path()

            current_node.print(True)

            # Get all children nodes
            agents = current_node.get_agents()
            children = []
            for agent in agents:

                moves = agent.get_moves()
                for new_position in moves:
                    new_node = Node(current_node, new_position)
                    children.append(new_node)

            # Add children to yet visited
            for child in children:
                # Child in on the visited list
                if len([visited_child for visited_child in visited if visited_child == child]) > 0:
                    continue

                child.g = current_node.g + cost
                child.h = child.get_heuristic(end_node)
                child.f = child.g + child.h

                # Child is already in the yet to visit list and g cost is already higher
                if len([yet_visited for yet_visited in yet_to_visit if child == yet_visited and child.g > yet_visited.g]):
                    continue


                yet_to_visit.append(child)


def find_path(starting_board, goal_board, search_method, detail_output):
    board_search = BoardSearch(starting_board, goal_board, search_method, detail_output)
    board_search.find_path()


starting_board = [[0, 1, 0, 0, 0, 0],
                  [0, 1, 0, 1, 1, 0],
                  [0, 1, 0, 1, 0, 0],
                  [0, 1, 0, 1, 0, 1],
                  [0, 1, 0, 1, 2, 2],
                  [0, 0, 0, 1, 2, 2]]
goal_board = [[2, 1, 0, 0, 0, 0],
              [0, 1, 0, 1, 1, 0],
              [2, 1, 2, 1, 0, 0],
              [0, 1, 0, 1, 0, 1],
              [0, 1, 0, 1, 0, 0],
              [0, 0, 0, 1, 0, 0]]

find_path(starting_board=starting_board, goal_board=goal_board, search_method=1, detail_output=None)
