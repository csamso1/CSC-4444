#Clayton Samson
#CSC 4444 HW 2 Code
#CS444455

from agents import *
from search import *

import copy

def main():
	print("Vacuume starting run:")
	clean_up()

	exit(0)

def clean_up():
	"""Initial and Goal States, as wella as a graph to search"""
	#Vacuume_Problem = NewGraphProblem('State5', ['State43', 'State44', 'State45', 'State46', 'State47', 'State48'], Vacuum_map)
	astar_search(Vacuume_Problem)

main()