#
# class SimulatedAnnealing:
#     def __init__(self, starting_board, goal_board):
#         self.start_node = Node(None, starting_board)
#         self.end_node = Node(None, goal_board)
#         self.start_node.h = self.start_node.get_heuristic(self.end_node)
#         self.initial_temp = 100
#
#     def search(self):
#         current_state = self.start_node
#
#         for t in range(1, 101):
#
#             if current_state == self.end_node:
#                 print('eq')
#                 return current_state.return_path()
#
#             current_temp = self.schedule(t)
#             if current_temp == 0:
#                 current_state.print(True)
#                 return None
#
#             step_chosen, steps_tried = False, 0
#             steps_considered = []
#             steps_considered_coord = []
#             agents = current_state.get_agents()
#             current_state_moves = self.get_children(current_node=current_state, agents=agents)
#             possible_moves = len(current_state_moves)
#
#             print('*' * 15)
#             print('****** before *******')
#             current_state.print(True)
#             while not step_chosen:
#                 next_step = self.get_random_successor(current_state_moves)
#                 delta_e = (current_state.h - next_step.h) / current_state.h
#
#                 step_action = self.get_step_considered(current_state, next_step)
#
#                 steps_considered_coord.append(step_action)
#
#                 if delta_e > 0:
#                     prob = 1
#                     current_state = next_step
#                     step_action['probability'] = prob
#
#                 elif next_step not in steps_considered:
#                     prob = np.exp(delta_e * current_temp)
#                     current_state = random.choices(population=[current_state, next_step], weights=[1 - prob, prob], k=1)[0]
#
#                     step_action['probability'] = round(prob, 3)
#
#
#                 else:
#                     steps_considered_coord.remove(step_action)
#
#                 if next_step not in steps_considered:
#                     steps_considered.append(next_step)
#
#                 if current_state == next_step:
#                     current_state.details = steps_considered_coord
#                     step_chosen = True
#                     print('****** After *******')
#                     current_state.print(True)
#                     print('*' * 15)
#
#                 if len(steps_considered) == possible_moves:
#                     current_state.details = steps_considered_coord
#                     current_state = min(enumerate(steps_considered), key=lambda x: x[1].h)[1]
#                     step_chosen = True
#                     print('****** After *******')
#                     current_state.print(True)
#                     print('*' * 15)
#
#                 current_state_moves.remove(next_step)
#
#     def get_step_considered(self, current, next):
#         current_coord = set(current.get_coordinates())
#         next_coord = set(next.get_coordinates())
#         if len(current_coord) == len(next_coord):
#             unique = (current_coord | next_coord) - (current_coord & next_coord)
#             origin = unique & current_coord
#             destination = unique & next_coord
#         else:
#             origin = current_coord - (current_coord & next_coord)
#             destination = {None, None}
#
#         return {'from': list(origin)[0], 'to': list(destination)[0]}
#
#     def schedule(self, t):
#         # return min(pow((self.initial_temp - t), (1 / 10)), (1 / 20) * t + 5)
#         return t
#
#     def get_random_successor(self, moves):
#         random_move = random.choice(moves)
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
