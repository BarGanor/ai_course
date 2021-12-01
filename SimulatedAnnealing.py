# import numpy as np
# import random
#
# class SimulatedAnnealing:
#     def __init__(self, starting_board, goal_board):
#         self.start_node = Node(None, starting_board)
#         self.end_node = Node(None, goal_board)
#         self.start_node.h = self.start_node.get_heuristic(self.end_node)
#         self.initial_temp = 100
#         self.final_temp = .1
#         self.alpha = 0.01
#
#     def search(self):
#         current_state = self.start_node
#
#         for t in range(100):
#
#             if current_state == self.end_node:
#                 return current_state.return_path()
#
#             current_temp = self.schedule(t)
#
#             if current_temp == 0:
#                 return current_state.return_path()
#
#             step_chosen = False
#             steps_considered = []
#             while not step_chosen:
#                 next_step = self.get_random_successor(current_state)
#                 delta_e = current_state.h - next_step.h
#
#                 if delta_e > 0:
#                     current_state = next_step
#
#                 elif next_step not in steps_considered:
#                     prob = np.exp(delta_e / current_temp)
#                     current_state = random.choices(population=[current_state, next_step], weights=[1 - prob, prob], k=1)[0]
#
#                 if current_state == next_step:
#                     step_chosen = True
#
#                 else:
#                     steps_considered.append(next_step)
#
#
#     def schedule(self, t):
#         return min(pow((self.initial_temp - t), (1/10)), 1)
#
#     def get_random_successor(self, current):
#         agents = current.get_agents()
#         moves = self.get_children(current_node=current, agents=agents)
#         random_move = random.choice(moves)
#
#         return random_move
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
#         return children
