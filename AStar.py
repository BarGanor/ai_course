# from q1 import Node


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
