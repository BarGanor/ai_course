import random
#
#
# class HillClimbing:
#     def __init__(self, starting_board, goal_board):
#
#         self.start_node = Node(None, starting_board)
#         self.end_node = Node(None, goal_board)
#
#         self.start_node.h = self.start_node.f = self.start_node.get_heuristic(self.end_node)
#
#         self.restarts = 0
#         self.tries = 0
#         self.max_tries = 300
#
#     def search(self):
#
#         while self.restarts < 5:
#             if self.restarts == 0:
#                 current_node = self.start_node
#
#             else:
#                 agents = self.start_node.get_agents()
#                 random_agent = random.choice(agents)
#                 random_agent_move = random.choice(random_agent.get_moves())
#                 current_node = Node(parent=self.start_node, position=random_agent_move)
#
#             while self.tries < self.max_tries:
#                 agents = current_node.get_agents()
#                 agents_children = self.get_children(current_node, agents)
#                 highest_valued = self.get_highest_valued_successor(agents_children)
#
#                 if highest_valued.h <= current_node.h:
#                     current_node = highest_valued
#                     current_node.print(True)
#
#                 if current_node == self.end_node:
#                     return current_node.return_path()
#
#     def get_children(self, current_node, agents):
#         children = []
#         for agent in agents:
#
#             moves = agent.get_moves()
#             for new_position in moves:
#                 new_node = Node(current_node, new_position)
#                 new_node.h = new_node.get_heuristic(self.end_node)
#                 children.append(new_node)
#
#         return children
#
#     def get_highest_valued_successor(self, moves):
#         highest_valued = None
#         for move in moves:
#             if highest_valued is None or move.h < highest_valued.h:
#                 highest_valued = move
#
#             else:
#                 pass
#         return highest_valued
