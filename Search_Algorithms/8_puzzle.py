"""
8_puzzle.py
@param method: search method to use (bfs/dfs/ast)
@param board: comma separated list of integers containing no spaces

Solutions:
- breadth-first-search (bfs)
- depth-first-search (dfs)
- a* with Manhattan distance heruistic (ast)
"""
import sys, time, resource
from math import sqrt,floor
from heapq import heappush, heappop

class FronteirQueue:

	class FronteirQueueNode:
		def __init__(self, data, next=None):
			self.data = data
			self.next = next

	def __init__(self):
		self.head = None
		self.tail = None
		self.node_ids = {}

	# returs false if node already in fronteir
	def add(self, node):
		if node.id not in self.node_ids:
			self.node_ids[node.id] = True

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

class FronteirStack:
	class FronteirStackNode:
		def __init__(self, data, next=None):
			self.data = data
			self.next = next

	def __init__(self):
		self.head = None
		self.node_ids = {}

	def add(self, node):
		if node.id not in self.node_ids:
			self.node_ids[node.id] = True

			if self.head is None:
				self.head = self.FronteirStackNode(node)
			else:
				new = self.FronteirStackNode(node, self.head)
				self.head = new

			return True
		else:
			return False

	def next(self):
		if self.head is not None:
			oldhead = self.head
			self.head = self.head.next
			return oldhead.data
		else:
			return None

class FronteirPriorityQueue:
	def __init__(self):
		self.heap = []
		self.nodes = {}
		self.counter = 0 # increment counter with every addition to mantain insertion order
	def add(self, node):
		if node.id in self.nodes:
			self.remove(node.id)
		
		priority = node.cost + node.manhattan_d()
		self.counter += 1
		entry = [priority, self.counter, node]
		self.nodes[node.id] = entry
		heappush(self.heap, entry)
	
	def remove(self, nodeid):
		entry = self.nodes.pop(nodeid)
		entry[-1] = None

	def next(self):
		while self.heap:
			priority, count, node = heappop(self.heap)
			if node is not None:
				del self.nodes[node.id]
				return node
		# raise KeyError('Fronteir (priority queue) is empty')
		return None


class Node:
	def __init__(self, data, action=None, cost=0, parent=None):
		self.size = int(sqrt(len(data)))
		self.state = [0] * len(data)
		self.cost = cost
		self.action = action
		self.parent = parent

		for n in range(len(data)):
			val = int(data[n])
			self.state[n] = val
			if val == 0:
				self.idx = n
				self.row = floor(self.idx/self.size)
				self.col = self.idx % self.size

		self.id = ",".join(map(str,self.state))
		self.goal_id = "0,1,2,3,4,5,6,7,8";

	def __str__(self):
		return "{} | cost: {} | action: {}".format(",".join(map(str,self.state)), self.cost, self.action)
		# return ",".join(map(str,self.state))

	def children(self):
		return [self.move_up(), self.move_down(), self.move_left(), self.move_right()]
	
	def swap(self, state, i, j):
		tmp = state[i]
		state[i] = state[j]
		state[j] = tmp
		return state

	def move_down(self):
		if self.row < self.size-1:
			newState = self.swap(self.state[:], self.idx, self.idx+self.size)
			return Node(newState, 'Down', self.cost+1, self)
		else:
			return None

	def move_up(self):
		if self.row > 0:
			newState = self.swap(self.state[:], self.idx, self.idx-self.size)
			return Node(newState, 'Up', self.cost+1, self)
		else:
			return None

	def move_right(self):
		if self.col < self.size-1:
			newState = self.swap(self.state[:], self.idx, self.idx+1)
			return Node(newState, 'Right', self.cost+1, self)
		else:
			return None

	def move_left(self):
		if self.col > 0:
			newState = self.swap(self.state[:], self.idx, self.idx-1)
			return Node(newState, 'Left', self.cost+1, self)
		else:
			return None

	def equals(self, other):
		for i in range(len(self.state)):
			if self.state[i] != other.state[i]:
				return False
		return True

	def is_goal(self):
		return self.id == self.goal_id;
		# for i in range(len(self.state)):
		# 	if self.state[i] != i:
		# 		return False
		# return True

	def manhattan_d(self):
		d = 0
		for i in range(len(self.state)):
			t = self.state[i]
			if t > 0 and t != i:
				# row_d = abs(floor(t/self.size) - floor(i/self.size))
				# col_d = abs(t % self.size - i % self.size)
				# d += (row_d + col_d)
				d += ( abs(t-i)//self.size + abs(t-i)%self.size )

		return d


class Board:
	def __init__(self, node):
		self.start = node
		self.explored = {}
		self.max_search_depth = 0
	
	def solve_bfs(self):
		if self.start.is_goal():
			return self.start

		self.fronteir = FronteirQueue()
		self.fronteir.add(self.start)
		

		n = self.fronteir.next()
		while n is not None:
			if n.is_goal(): return n
			self.explored[n.id] = True
			for child in n.children():
				if child is not None and child.id not in self.explored:
					added = self.fronteir.add(child)
					if added and child.cost > self.max_search_depth:
						self.max_search_depth = child.cost

			n = self.fronteir.next()

		# if no solution - return none
		return None

	def solve_dfs(self):
		self.fronteir = FronteirStack()
		self.fronteir.add(self.start)

		n = self.fronteir.next()
		while n is not None:
			
			if n.is_goal(): return n

			self.explored[n.id] = True
			children = n.children()
			for i in range(len(children)-1, -1, -1):
				child = children[i]
				if child is not None and child.id not in self.explored:
					# if child.cost > self.max_search_depth:
					# 	self.max_search_depth = child.cost
					# self.fronteir.add(child)
					added = self.fronteir.add(child)
					if added and child.cost > self.max_search_depth:
						self.max_search_depth = child.cost
			n = self.fronteir.next()

		# if no solution - return none
		return None

	# def solve_dfs(self, node):
	# 	if node.is_goal():
	# 		return node
	# 	else:
	# 		children = n.children()

	# 		for i in range(len(children)-1, -1, -1):
	# 			return self.solve_dfs(children[i])

	def solve_ast(self):
		self.fronteir = FronteirPriorityQueue()
		self.fronteir.add(self.start)

		n = self.fronteir.next()
		while n is not None:
			if n.is_goal(): return n
			self.explored[n.id] = True
			for child in n.children():
				if child is not None and child.id not in self.explored:
					if child.cost > self.max_search_depth:
						self.max_search_depth = child.cost

					self.fronteir.add(child)
			n = self.fronteir.next()

		# if no solution - return none
		return None


class Solution:
	def __init__(self, node, board):
		self.node = node
		
		self.path_to_goal = self.path_to_node()
		self.cost_of_path = node.cost
		self.search_depth = node.cost
		self.nodes_expanded = len(board.explored)
		self.max_search_depth = board.max_search_depth


	def path_to_node(self):
		self.depth = 1
		path = []
		n = self.node
		while n.parent is not None:
			path.append(n.action)
			n = n.parent
			self.depth += 1
		path.reverse()
		return path

	def show(self):
		print("path_to_goal: {}\n".format(s.path_to_goal))
		print("cost_of_path: {}\n".format(s.cost_of_path))
		print("nodes_expanded: {}\n".format(s.nodes_expanded))
		print("search_depth: {}\n".format(s.search_depth))
		print("max_search_depth: {}\n".format(s.max_search_depth))
		print("running_time: {0:.8f}\n".format(t))
		print("max_ram_usage: {0:.8f}\n".format(ram))

if __name__ == '__main__':

	print("8 Puzzle Problem")
	print("----------------------------------")

	start_time = time.time()

	if len(sys.argv) < 3:
		print("Usage: driver_3.py <method> <board>")
		exit()

	method = sys.argv[1]
	board = sys.argv[2].split(",")

	start = Node(board)
	# print("Start: ", start)
	

	B = Board(start)
	if method == "bfs":
		finish = B.solve_bfs()
	elif method == "dfs":
		finish = B.solve_dfs()
	elif method == "ast":
		finish = B.solve_ast()

	
	if finish is not None:
		s = Solution(finish, B)

		t = time.time() - start_time
		ram = (resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)/1000000

		# with open('output.txt', 'w') as f:
		# 	f.write("path_to_goal: {}\n".format(s.path_to_goal))
		# 	f.write("cost_of_path: {}\n".format(s.cost_of_path))
		# 	f.write("nodes_expanded: {}\n".format(s.nodes_expanded))
		# 	f.write("search_depth: {}\n".format(s.search_depth))
		# 	f.write("max_search_depth: {}\n".format(s.max_search_depth))
		# 	f.write("running_time: {0:.8f}\n".format(t))
		# 	f.write("max_ram_usage: {0:.8f}\n".format(ram))

		print("path_to_goal: {}".format(s.path_to_goal))
		print("cost_of_path: {}".format(s.cost_of_path))
		print("nodes_expanded: {}".format(s.nodes_expanded))
		print("search_depth: {}".format(s.search_depth))
		print("max_search_depth: {}".format(s.max_search_depth))
		print("running_time: {0:.8f}".format(t))
		print("max_ram_usage: {0:.8f}".format(ram))


	else:
		print ("No solution found")