"""
06/15/18
Missionaries and Cannibals.

Three missionaries and three cannibals are on one side of the river, 
along with a boat that can hold one or two people. Find a way to get
everyone to the other side without ever leaving a group of missionaries 
in one place outnumbered by the cannibals in that place (cannibals will eat the missionaries).

Breadth-First-Search solution

"""

import time, resource

class FronteirQueue:

	class FronteirQueueNode:
		def __init__(self, data, next=None):
			self.data = data
			self.next = next

	def __init__(self):
		self.head = None
		self.tail = None
		self.members = {}

	# returs false if node already in fronteir
	def add(self, node):
		if node.id not in self.members:
			self.members[node.id] = True

			if self.head is None:
				self.head = self.FronteirQueueNode(node)
				self.tail = self.head
			else:
				new = self.FronteirQueueNode(node)
				self.tail.next = new
				self.tail = new

			return True
		else:
			return False

	def next(self):
		if self.head is not None:
			oldhead = self.head
			self.head = self.head.next
			if self.head is None:
				self.tail = self.head
			return oldhead.data
		else:
			return None
		
class State:
	def __init__(self, data, action=None, parent=None):
		# data = [[missionaiers_left, cannibals_left, boat], [missionaier_right, cannibals_right, boat]]
		self.state = data
		self.action = action
		self.parent = parent
		self.id = str(self.state)
		self.actions = [[-1,-1,-1], [0,-1,-1], [-1,0,-1], [0,-2,-1], [-2,0,-1]]
	def __str__(self):
		return str(self.state)
	def is_valid(self):
		if self.state[0][0] > 0 and self.state[0][0] < self.state[0][1]:
			return False
		if self.state[1][0] > 0 and self.state[1][0] < self.state[1][1]:
			return False

		return True
		# return self.state[0][0] >= self.state[0][1] and self.state[1][0] >= self.state[1][1]
	def is_goal(self):
		return self.state[1] == [3,3,1]
	def move(self, state, action):
		# where is the boat?
		if state[0][2]:
			fr = state[0]
			to = state[1]
		else:
			fr = state[1]
			to = state[0]

		for i in range(len(action)):
			fr[i] += action[i]
			to[i] -= action[i]
			# detect invalid moves
			if fr[i] < 0 or to[i] < 0:
				return None

		return State(state, action, self)

	def expand(self):
		children = []
		for a in self.actions:
			clone = [self.state[i][:] for i in range(len(self.state))]
			child = self.move(clone, a)
			# print("{} | {}".format(child, child.is_valid()))
			if child is not None and child.is_valid():
				children.append(child)

		return children


class Game:
	def __init__(self):
		self.explored = {}

	def solve_bsf(self, state):
		fronteir = FronteirQueue()
		fronteir.add(state)

		S = fronteir.next()
		while S is not None:
			# print(S)
			if S.is_goal():
				return Solution(S) # found solution!
			self.explored[S.id] = True
			children = S.expand()
			for child in children:
				if child.id not in self.explored:
					fronteir.add(child)

			# explore next node
			S = fronteir.next()
		
		# no solution found
		return None

class Solution:
	def __init__(self, state):
		self.state = state
		self.path_to_goal = self.path_to_node()

	def path_to_node(self):
		self.search_depth = 1
		path = []
		n = self.state
		while n.parent is not None:
			path.append(n.action)
			n = n.parent
			self.search_depth += 1
		path.reverse()
		return path 

	def show(self):
		nodes = [self.state]
		actions = []
		
		n = self.state
		while n.parent is not None:
			actions.append(n.action)
			nodes.append(n.parent)
			n = n.parent

		actions.reverse()
		actions.append(None)
		nodes.reverse()

		print("   [[M, C, B], [M, C, B]] | Action ")
		for i in range(len(nodes)):
			print("{}. {} | {}".format(i, nodes[i], actions[i]))

if __name__ == "__main__":
	print("Missionaries and Cannibals Problem")
	print("----------------------------------")

	start_time = time.time()

	initial_state = State([[3,3,1], [0,0,0]])
	game = Game()
	solution = game.solve_bsf(initial_state)

	if solution is not None:
		solution.show()
	else:
		print("no solution found!")

	t = time.time() - start_time
	ram = (resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)/1000000
	print("\n----------------------------------\n")
	print("running_time: {0:.8f}".format(t))
	print("max_ram_usage: {0:.8f}".format(ram))
	print("nodes_expanded: {}".format(len(game.explored)))
	print("search_depth: {}".format(solution.search_depth))
	print("\n")


	