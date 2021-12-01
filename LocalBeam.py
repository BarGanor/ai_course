



class LocalBeam:
    def __init__(self, starting_board, goal_board):
        self.start_node = Node(None, starting_board)
        self.end_node = Node(None, goal_board)
        self.start_node.h = self.start_node.f = self.start_node.get_heuristic(self.end_node)
        self.k = 3
        self.tries = 0
        self.visited = []

    def search(self):
        all_nodes_moves = self.get_all_nodes_moves([self.start_node])
        current_nodes = self.get_best_k(all_nodes_moves)

        while len(current_nodes) > 0 and self.tries < 100:

            path_found, path_node = self.path_found(current_nodes)

            if path_found:
                return path_node.return_path()

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
                return True, node

            else:
                return False, None

    def get_best_k(self, moves):
        best_k_moves = []

        for move in moves:

            if len(best_k_moves) < self.k:
                best_k_moves.append(move)

            else:
                max_h_index, max_h_move = max(enumerate(best_k_moves), key=lambda x: x[1].h)
                if move.h < max_h_move.h and move not in self.visited and move not in best_k_moves:
                    best_k_moves[max_h_index] = move
                    move.print(True)
                    self.visited.append(move)

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
