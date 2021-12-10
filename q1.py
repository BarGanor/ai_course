import math

import numpy as np
from scipy.spatial.distance import cdist

import random

'''                             A* Algorithm                          '''


class AStar:
    def __init__(self, starting_board, goal_board, cost):
        self.cost = cost
        self.start_node = Node(None, starting_board)
        self.end_node = Node(None, goal_board)
        self.yet_to_visit = [self.start_node]
        self.visited = []

        self.tries = 0
        self.max_tries = 15000
        self.start_node.h = self.start_node.f = self.start_node.get_heuristic(self.end_node)

    def search(self):
        yet_to_visit = self.yet_to_visit

        while len(yet_to_visit) > 0:
            self.tries += 1

            current_node = yet_to_visit[0]
            current_index = 0

            current_node, current_index = self.pick_best_node(current_node, current_index)
            yet_to_visit.pop(current_index)
            self.visited.append(current_node)

            if current_node == self.end_node:
                return current_node.return_path()

            if self.check_max_tries():
                break

            agents = current_node.get_agents()  # Get all agents on board
            children = self.get_children(current_node, agents)  # Get all children_nodes
            self.add_children_yet_to_visit(children, current_node)  # Append children to yet to visit

    def check_max_tries(self):
        return self.tries > self.max_tries

    def pick_best_node(self, current_node, current_index):
        for index, item in enumerate(self.yet_to_visit):
            if item.f < current_node.f:
                current_node = item
                current_index = index

        return current_node, current_index

    @staticmethod
    def get_children(current_node, agents):
        children = []
        for agent in agents:

            moves = agent.get_moves()
            for new_position in moves:
                new_node = Node(current_node, new_position)
                children.append(new_node)

        return children

    def add_children_yet_to_visit(self, children, current_node):
        for child in children:
            # Child in on the visited list
            if len([visited_child for visited_child in self.visited if visited_child == child]) > 0:
                continue

            child.g = current_node.g + self.cost
            child.h = child.get_heuristic(self.end_node)
            child.f = child.g + child.h

            # Child is already in the yet to visit list and g cost is already higher
            if len([yet_visited for yet_visited in self.yet_to_visit if child == yet_visited and child.g >= yet_visited.g]) > 0:
                continue

            self.yet_to_visit.append(child)


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
            if self.restarts == 0:
                current_node = self.start_node

            else:
                agents = self.start_node.get_agents()
                random_agent = random.choice(agents)
                random_agent_move = random.choice(random_agent.get_moves())
                current_node = Node(parent=self.start_node, position=random_agent_move)

            while self.tries < self.max_tries:
                agents = current_node.get_agents()
                agents_children = self.get_children(current_node, agents)
                highest_valued = self.get_highest_valued_successor(agents_children)

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

    @staticmethod
    def get_highest_valued_successor(moves):
        highest_valued = None
        for move in moves:
            if highest_valued is None or move.h < highest_valued.h:
                highest_valued = move

            else:
                pass
        return highest_valued


class SimulatedAnnealing:
    def __init__(self, starting_board, goal_board):
        self.start_node = Node(None, starting_board)
        self.end_node = Node(None, goal_board)
        self.start_node.h = self.start_node.get_heuristic(self.end_node)
        self.initial_temp = 3
        self.alpha = 0.03

    def search(self):
        current_state = self.start_node
        current_state.details = {}
        current_temp = self.initial_temp

        current_actions_considered = []
        for t in range(1, 101):

            # Check if found path
            if current_state == self.end_node:
                return current_state.return_path()

            # If the temperature reached 0, return the path so far
            if current_temp == 0.03:
                return None

            # Pick a random step
            current_state_children = self.get_children(current_node=current_state)
            next_step = random.choice(current_state_children)

            # Calculate the difference between the current and the random next step
            delta_e = (current_state.h - next_step.h) / current_state.h

            # Get the probability to choose the next step and whether or not to take it
            accept_change, probability = self.accept_change(current_temp, delta_e)

            step_considered = self.get_step_considered(current_state, next_step, probability)
            current_actions_considered.append(step_considered)

            # If the difference is positive, make the change
            if delta_e > 0:
                current_state = next_step
                current_state.details['steps_considered'] = current_actions_considered
                current_actions_considered = []

            # If the difference is negative (and not deadly) but we accept it anyways, make the change
            elif accept_change and next_step.h != 100:
                current_state = next_step
                current_state.details['steps_considered'] = current_actions_considered

            current_temp = self.schedule(t)

    # Returns true if the random choice was made
    @staticmethod
    def accept_change(curr_temp, delta_e):
        acceptance_rate = math.exp(delta_e / curr_temp)

        if delta_e == 0:
            return random.choice([True, False]), 0.5
        elif delta_e > 0:
            return True, 1
        else:
            return random.choices(population=[True, False], k=1, weights=[acceptance_rate, 1 - acceptance_rate]), acceptance_rate

    # Create a dictionary containing the data needed on this step.
    @staticmethod
    def get_step_considered(current_step, next_step, probability):
        current_coord = set(current_step.get_coordinates())
        next_coord = set(next_step.get_coordinates())
        if len(current_coord) == len(next_coord):
            unique = (current_coord | next_coord) - (current_coord & next_coord)
            origin = unique & current_coord
            destination = unique & next_coord
        else:
            origin = current_coord - (current_coord & next_coord)
            destination = {None, None}

        return {'from': list(origin)[0], 'to': list(destination)[0], 'probability': probability}

    def schedule(self, t):
        return round(self.initial_temp - t * self.alpha, 2)

    # Get all children nodes of current node
    def get_children(self, current_node):
        agents = current_node.get_agents()
        children = []
        for agent in agents:

            moves = agent.get_moves()
            for new_position in moves:
                new_node = Node(current_node, new_position)
                new_node.h = new_node.get_heuristic(self.end_node)
                new_node.details = {}
                children.append(new_node)
        if len(children) == 0:
            print(current_node)
        return children


class LocalBeam:
    def __init__(self, start_board, end_board):
        self.start_node = Node(None, start_board)
        self.end_node = Node(None, end_board)
        self.start_node.h = self.start_node.f = self.start_node.get_heuristic(self.end_node)
        self.k = 3
        self.tries = 0
        self.visited = []

    def search(self):
        all_nodes_moves = self.get_all_nodes_moves([self.start_node])
        current_nodes = self.get_best_k(all_nodes_moves)

        while len(current_nodes) > 0 and self.tries < 100:
            if self.path_found(current_nodes):
                return [node.return_path() for node in current_nodes]

            all_nodes_moves = self.get_all_nodes_moves(current_nodes)
            current_nodes = self.get_best_k(all_nodes_moves)

    def get_all_nodes_moves(self, current):

        all_moves = []
        for node in current:
            agents = node.get_agents()
            moves = self.get_children(node, agents)
            all_moves.extend(moves)

        return all_moves

    def path_found(self, current):
        for node in current:
            if node == self.end_node:
                return True

        return False

    def get_best_k(self, moves):
        best_k_moves = []

        for move in moves:

            if len(best_k_moves) < self.k:
                best_k_moves.append(move)

            else:
                max_h_index, max_h_move = max(enumerate(best_k_moves), key=lambda x: x[1].h)
                if move.h < max_h_move.h and move not in best_k_moves:
                    best_k_moves[max_h_index] = move

        return best_k_moves

    def get_children(self, current_node, agents):
        children = []
        for agent in agents:

            moves = agent.get_moves()
            for new_position in moves:
                new_node = Node(current_node, new_position)
                new_node.h = new_node.get_heuristic(self.end_node)

                children.append(new_node)

        return children


class GeneticAlgorithm:
    def __init__(self, starting_board, goal_board):
        self.start_node = Node(None, starting_board)
        self.end_node = Node(None, goal_board)
        self.start_node.h = self.start_node.get_heuristic(self.end_node)
        self.tries = 0
        self.mutate_probability = 0.1

    def search(self):
        population = self.gen_initial_population()

        while self.tries < 30:  # Up to 30 generation at most
            new_population = []
            population = self.assign_weights(population)
            weights = [i.details['weight'] for i in population]

            for i in range(len(population)):
                valid_child = False
                while not valid_child:
                    random_father, random_mother = np.random.choice(population, size=2, p=weights)
                    child = self.reproduce(random_father, random_mother)
                    valid_child = self.is_valid_child(child, random_father, random_mother)

                    if valid_child and (random_father.h >= child.h or random_mother.h >= child.h):
                        new_population.append(child)

            finished, node = self.check_if_finished(new_population)
            if finished:
                return node, True
            population = new_population
            self.tries += 1
        min_h = min(enumerate(population), key=lambda x: x[1].h)[1]
        return min_h.return_path(), False

    # Check if the child is a valid move of any parent
    def is_valid_child(self, child, father, mother):
        is_valid = True
        valid_child_of_father = self.valid_move(child, father)
        valid_child_of_mother = self.valid_move(child, mother)

        # If both are valid, choose the one that minimizes h and save other parent.
        if valid_child_of_father and valid_child_of_mother:
            child.parent = min(father, mother, key=lambda x: x.h)
            child.details['other_parent'] = max(father, mother, key=lambda x: x.h)

        elif self.valid_move(child, father):
            child.parent = father
            child.details['other_parent'] = mother

        elif self.valid_move(child, mother):
            child.parent = mother
            child.details['other_parent'] = father

        else:
            is_valid = False

        return is_valid

    # Check if a move is valid
    @staticmethod
    def valid_move(curr, parent):
        curr_coord = set(curr.get_coordinates())
        parent_coord = set(parent.get_coordinates())

        unique_in_curr = list(curr_coord - (curr_coord & parent_coord))
        unique_in_parent = list(parent_coord - (curr_coord & parent_coord))

        # Child can't have more agents than parent and only one agent can change position in each move
        if (len(curr_coord) > len(parent_coord)) | (len(unique_in_curr) > 1):
            return False

        # If one agent moved, check if the move was legal
        elif len(unique_in_curr) == len(unique_in_parent) == 1:

            horizontal_move = (unique_in_parent[0][0] == unique_in_curr[0][0] + 1) | (unique_in_parent[0][0] == unique_in_curr[0][0] - 1)
            vertical_move = (unique_in_parent[0][1] == unique_in_curr[0][1] + 1) | (unique_in_parent[0][1] == unique_in_curr[0][1] - 1)

            return horizontal_move or vertical_move

        # Only agents on the edge can disappear
        elif (len(curr_coord) < len(parent_coord)) | (len(parent_coord) == 1):
            on_edge = (unique_in_parent[0][0] == 5) | (unique_in_parent[0][1] == 5)
            return on_edge

        # If child is the parent, move is valid
        elif curr_coord == parent_coord:
            return True

        else:
            return False

    # The weights (probability to pick each parent node) are the relative heuristic compared to others
    @staticmethod
    def assign_weights(population):
        list_of_h = [(101 - i.h) for i in population]
        sum_of_h = sum(list_of_h)
        list_of_weights = [i / sum_of_h for i in list_of_h]

        for i in range(len(population)):
            population[i].details['weight'] = list_of_weights[i]
        return population

    def check_if_finished(self, population):
        for node in population:
            if node == self.end_node:
                return True, node

            else:
                return False, None

    # Child creation- slices the parent nodes randomly and then concatenates them.
    def reproduce(self, father, mother):
        num_cols = father.position.shape[0]
        slice_num = random.choice(range(num_cols))

        # Slice each parent to create parts for child
        sliced_father = father.position[:, slice_num:]
        sliced_mother = mother.position[:, :slice_num]

        # Concatenate the parent parts to create a legal child.
        child = np.concatenate((sliced_mother, sliced_father), axis=1)
        child = Node(parent=[father, mother], position=child)

        if self.do_mutation():
            child = self.mutate(child)
        else:
            child.details = {'mutated': 'no'}
        child.h = child.get_heuristic(self.end_node)

        return child

    # Random choice by probability to decide id to do mutation
    def do_mutation(self):
        return np.random.choice([True, False], 1, p=[self.mutate_probability, 1 - self.mutate_probability])

    # The mutation changes the position of the child to a new legal position
    @staticmethod
    def mutate(child):
        child_agents = child.get_agents()
        agent_moves = [agent.get_moves() for agent in child_agents]
        moves = [Node(parent=child.parent, position=item) for sublist in agent_moves for item in sublist]
        child = random.choice(moves)

        child.details = {'mutated': 'yes'}

        return child

    # The initial population is composed of the top 10 available moves from the start board
    def gen_initial_population(self):
        chosen_moves = []
        agents = self.start_node.get_agents()
        for agent in agents:
            agent_moves = agent.get_moves()
            for move in agent_moves:
                move_node = Node(parent=self.start_node, position=move)
                move_node.h = move_node.get_heuristic(self.end_node)
                move_node.details = {'mutated': 'no'}
                chosen_moves.append(move_node)

        sorted_moves = sorted(chosen_moves, key=lambda x: x.h, reverse=True)
        return sorted_moves[:10]


"""
The Board Search class is the "search manager" of this task. 
Jobs - 
1. Initiate a search by the arguments given to it.
2. Print the result for each search type.
3. check if the board is possible 
"""


class BoardSearch:
    def __init__(self, starting_board, goal_board, search_method, detail_output):
        self.method_dict = {1: AStar, 2: HillClimbing, 3: SimulatedAnnealing, 4: LocalBeam, 5: GeneticAlgorithm}
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

        elif self.search_method == HillClimbing:
            hill_climbing = HillClimbing(starting_board=self.starting_board, goal_board=self.goal_board)
            path = hill_climbing.search()

        elif self.search_method == SimulatedAnnealing:
            simulated_annealing = SimulatedAnnealing(starting_board=self.starting_board, goal_board=self.goal_board)
            path = simulated_annealing.search()

        elif self.search_method == LocalBeam:
            local_beam = LocalBeam(start_board=self.starting_board, end_board=self.goal_board)
            path = local_beam.search()

        elif self.search_method == GeneticAlgorithm:
            genetic_algorithm = GeneticAlgorithm(starting_board=self.starting_board, goal_board=self.goal_board)
            path = genetic_algorithm.search()

        else:
            path = None

        self.print_path(path)

    def print_path(self, path):
        if path is None:
            print('No path found')

        elif self.search_method == SimulatedAnnealing:

            for step in range(len(path)):
                if step == 0:
                    print('Board 1 (starting position):')
                    print(path[step])

                elif step == len(path) - 1:
                    print('Board ' + str(step + 1) + '(goal position):')
                    print(path[step])
                else:
                    print('Board ' + str(step + 1) + ':')
                    print(path[step])

                    final_output = ''
                    for action in path[step].details['steps_considered']:
                        action_str = 'action:' + str(action['from']) + '->' + str(action['to']) + '; probability:' + str(action['probability']) + '\n'
                        final_output += action_str
                    print(final_output.rstrip('\n'))
                print('-' * 5)

        elif self.search_method == LocalBeam:
            print('Board 1 (starting position):')
            print(path[0][0])

            for step in range(1, len(path[0])):
                print('Board ' + str(step + 1) + 'a :')
                print(path[0][step])
                print('Board ' + str(step + 1) + 'b :')
                print(path[1][step])
                print('Board ' + str(step + 1) + 'c :')
                print(path[2][step])

        elif self.search_method == GeneticAlgorithm:

            if path[1]:
                for step in range(len(path)):
                    if step == 0:
                        print('Board 1 (starting position):')
                        path[step].print()

                    elif step == len(path) - 1:
                        print('Board ' + str(step + 1) + '(goal position):')
                        path[step].print(last=True)
                    else:
                        print('Board ' + str(step + 1) + ':')
                        path[step].print(detail_output=self.detail_output)

            genetic_creation = path[0][-1]
            mutated = str(genetic_creation.details['mutated'])
            first_parent_weight = str(genetic_creation.parent.details['weight'])
            second_parent_weight = str(genetic_creation.details['other_parent'].details['weight'])
            print('Starting board 1 (probability of selection from population::' + first_parent_weight +'):')
            print(genetic_creation.parent)
            print('Starting board 2 (probability of selection from population::' + second_parent_weight +'):')
            print(genetic_creation.details['other_parent'])
            print('Result board (mutation happened::' + mutated + '):')
            print(genetic_creation)

        else:
            for step in range(len(path)):
                if step == 0:
                    print('Board 1 (starting position):')
                    print(path[step])

                elif step == len(path) - 1:
                    print('Board ' + str(step + 1) + '(goal position):')
                    print(path[step])
                else:
                    print('Board ' + str(step + 1) + ':')
                    print(path[step])

    def less_agents_then_needed(self):
        return (self.starting_board == 2).sum() < (self.goal_board == 2).sum()


"""
The Node class represents a board in the search. 
Jobs - 
1. Get all agents on current board.
2. Print itself properly.
3. Make a change to itself (return a board with a single move).
4. Return its own path- Get all parent nodes (boards) and return a declining list of them. 
5. Return a list of tuples with coordinates (x,y) of all agents in itself.
6. Given a goal board, return a heuristic of itself. 
"""


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

        self.details = None

    def __eq__(self, other):
        if other is None:
            return False
        comparison = self.position == other.position
        return comparison.all()

    def __hash__(self):
        return 0

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

    def __str__(self):
        transform_dict = {0: ' ', 1: '@', 2: '*'}
        column_numbers = '  1 2 3 4 5 6'
        rows = ''
        for row in range(len(self.position)):
            string_ints = [transform_dict[num] for num in self.position[row]]
            rows += str(row + 1) + ':' + ' '.join(string_ints) + '\n'

        final = column_numbers + '\n' + rows
        return final

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

    # The heuristic measurement - return the min distance between two coordinate vectors
    @staticmethod
    def get_min_sum_of_distance(curr, goal):
        curr_clean = [x for x in curr if x not in goal]
        goal_clean = [x for x in goal if x not in curr]

        if len(curr_clean) == 0:
            return 0

        distances = cdist(np.array(curr_clean), np.array(goal_clean))
        min_dist = min(distances.sum(axis=1))

        return float(min_dist)


"""
The Agent class represents a agent of a certain board 
Jobs - 
1. Get all possible moves that it has (as nodes)
"""


class Agent:
    def __init__(self, node, col, row):
        self.node = node
        self.col = col
        self.row = row

    def get_moves(self): # Get all possible moves and return only valid (not none)
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


"""
The Move class represents an agents move on a certain board 
Jobs - 
1. Given a parent node and a direction, return a copy of the parent node after a move (changed board).
"""


class Move(Agent):
    def __init__(self, node: Node, col, row, direction):
        super().__init__(node, col, row)
        self.node = node
        self.row = row
        self.col = col
        self.direction = direction

    def get_move(self):  # Get new board after changed position
        wanted_row = self.row + self.direction[0]
        wanted_col = self.col + self.direction[1]
        move = self.node.change_board(self.row, self.col, wanted_row=wanted_row, wanted_col=wanted_col)

        return move


def find_path(starting_board, goal_board, search_method, detail_output):
    board_search = BoardSearch(starting_board, goal_board, search_method, detail_output)
    return board_search.find_path()

