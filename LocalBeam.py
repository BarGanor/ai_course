#
#
# class LocalBeam:
#     def __init__(self, start_board, end_board):
#         self.start_node = Node(None, start_board)
#         self.end_node = Node(None, end_board)
#         self.start_node.h = self.start_node.f = self.start_node.get_heuristic(self.end_node)
#         self.k = 3
#         self.tries = 0
#         self.visited = []
#
#     def search(self):
#         all_nodes_moves = self.get_all_nodes_moves([self.start_node])
#         current_nodes = self.get_best_k(all_nodes_moves)
#
#         while len(current_nodes) > 0 and self.tries < 100:
#             if self.path_found(current_nodes):
#                 return [node.return_path() for node in current_nodes]
#
#             all_nodes_moves = self.get_all_nodes_moves(current_nodes)
#             current_nodes = self.get_best_k(all_nodes_moves)
#
#     def get_all_nodes_moves(self, current):
#
#         all_moves = []
#         for node in current:
#             agents = node.get_agents()
#             moves = self.get_children(node, agents)
#             all_moves.extend(moves)
#
#         return all_moves
#
#     def path_found(self, current):
#         for node in current:
#             if node == self.end_node:
#                 return True
#
#         return False
#
#     def get_best_k(self, moves):
#         best_k_moves = []
#
#         for move in moves:
#
#             if len(best_k_moves) < self.k:
#                 best_k_moves.append(move)
#
#             else:
#                 max_h_index, max_h_move = max(enumerate(best_k_moves), key=lambda x: x[1].h)
#                 if move.h < max_h_move.h and move not in best_k_moves:
#                     best_k_moves[max_h_index] = move
#
#         return best_k_moves
#
#     def get_children(self, current_node, agents):
#         children = []
#         for agent in agents:
#
#             moves = agent.get_moves()
#             for new_position in moves:
#                 new_node = Node(current_node, new_position)
#                 new_node.h = new_node.get_heuristic(self.end_node)
#
#                 children.append(new_node)
#
#         return children
#
