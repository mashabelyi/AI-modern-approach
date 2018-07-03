from BaseAI_3 import BaseAI
from random import randint
from math import log2, pow
import time


# Time Limit Before Losing
timeLimit = 0.2
allowance = 0.04 #0.05

class GridHelper:
	
	def playerChildren(grid):
		availableMoves = grid.getAvailableMoves()
		children = []
		for move in availableMoves:
			tmp = grid.clone()
			tmp.move(move)
			children.append(tmp)
		return children

	def computerChildren(grid):
		availableCells = grid.getAvailableCells()
		children = []
		for cell in availableCells:
			for tile in [2,4]:
				tmp = grid.clone()
				tmp.insertTile(cell, tile)
				children.append(tmp)

		# Order by worst eval score?
		return children

	def eval(grid):
		numEmptyTiles = len(grid.getAvailableCells()) 
		# return numEmptyTiles
		
		# favor large numbers on the edges of the grid
		W = [[3,3,3,3],[3,-3,-3,3],[3,-3,-3,3],[3,3,3,3]] # Winning!!!
		
		monotonicity = 0
		# penalty = 0
		# bonus = 0 #for adjacent tiles
		smoothness = 0
		
		neighboors = [[-1,0], [1,0], [0,-1], [0,1]]
		for i in range(grid.size):
			for j in range(grid.size):
				tile = grid.getCellValue([i,j])

				if tile > 0:
					monotonicity += W[i][j] * log2(grid.getCellValue([i,j]))
					
					numTilesNearby = 0
					valueDifference = 0
					for n in neighboors:
						ii = i+n[0]
						jj = j+n[1] 
						if ii > -1 and ii < grid.size and jj > -1 and jj < grid.size:
							neighboor = grid.getCellValue([ii,jj])
							if neighboor > 0:
								numTilesNearby += 1
								valueDifference += abs(log2(neighboor) - log2(tile))

						if numTilesNearby > 0:
							smoothness += valueDifference/numTilesNearby



		maxTileScore = log2(grid.getMaxTile())

		maxScore_weight = 1
		smoothness_weight = -1
		emptyTiles_weight = max(2, maxTileScore) # 2
		monotonicity_weight = 1

		return maxTileScore*maxScore_weight + numEmptyTiles*emptyTiles_weight + monotonicity*monotonicity_weight + smoothness*smoothness_weight


class MinMaxSearch:
	"""
	Min-Max Search with Alpha/Beta pruning
	"""
	def getAction(self, grid):
		self.startTime = time.clock()
		self.outOfTime = False

		v = float("-inf")
		bestMove = None
		availableMoves = grid.getAvailableMoves()
		depth_limit = 4
		# self.depth = depth_limit
		# depth_limit = max(4, log2(grid.getMaxTile())-2)

		for move in availableMoves:
			child = grid.clone()
			child.move(move)
			val = self.minValue(child, float("-inf"), float("inf"), depth_limit)
			if val >= v:
				v = val
				bestMove = move

		
		return bestMove

	
	def checkTime(self, currTime):
		if currTime - self.startTime > timeLimit + allowance:
			self.outOfTime = True

	def ranOutOfTime(self):
		return (time.clock() - self.startTime) > (timeLimit + allowance)

	# Player's turn
	def maxValue(self, grid, alpha, beta, depth):
		self.checkTime(time.clock())
		
		if not grid.canMove() or self.outOfTime or depth==0:
			return GridHelper.eval(grid)
		else:
			v = float("-inf")
			children = GridHelper.playerChildren(grid)
			for child in children:
				v = max(v, self.minValue(child, alpha, beta, depth-1))
				if v >= beta: 
					return v
				alpha = max(alpha, v)

				if self.ranOutOfTime():
					break

			return v
	
	# Computer's turn
	def minValue(self, grid, alpha, beta, depth):
		self.checkTime(time.clock())
		
		# availableCells = grid.getAvailableCells()
		# if len(availableCells)==0 or self.outOfTime or depth==0:
		if not grid.canMove() or self.outOfTime or depth==0:
			return GridHelper.eval(grid)
		else:
			
			children = GridHelper.computerChildren(grid)

			v = float("inf")
			move = None
			for child in children:
				
				v = min(v, self.maxValue(child, alpha, beta, depth-1))
				if v <= alpha: 
					return v
				beta = min(beta, v)

				if self.ranOutOfTime():
					break

			return v

class PlayerAI(BaseAI):
	def getMove(self, grid):
		# h = GridHelper.eval(grid)
		# print(h)

		minmax = MinMaxSearch()
		move = minmax.getAction(grid)
		return move
		

		# moves = grid.getAvailableMoves()
		# return moves[randint(0, len(moves) - 1)] if moves else None
