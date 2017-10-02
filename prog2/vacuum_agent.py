#Clayton Samson
#CSC 4444 HW 2 Code
#CS444455

from agents import *
from search import *

import copy

def main():
	Vacuum_map = UndirectedGraph(dict(
	    State1 =dict(Right = ['State2', 7], Down =['State4', 7], Suck = ['State7', 5]),
	    State2 =dict(Right = ['State3', 7], Down =['State5', 7], Left = ['State1', 7] ,Suck = ['State14', 5]), 
	    State3 =dict (Down =['State6', 7], Left = ['State2', 7], Suck = ['State21', 5]),   
	    State4 =dict(Up =['State1', 7], Right = ['State5', 7]),   
	    State5 =dict (Up =['State2', 7], Left = ['State4', 7], Right = ['State6', 7]),   
	    State6 =dict (Up =['State3', 7], Left = ['State5', 7] ),
	    State7 =dict (Right =['State8', 5], Down = ['State10', 5] ),
	    State8 =dict (Left =['State7', 5], Down = ['State11', 5], Right = ['State9', 5] , Suck = ['State26', 3]),
	    State9 =dict (Down =['State12', 5], Suck = ['State33', 3],  Left = ['State8', 5]),    
	    State10 =dict (Up =['State7', 5],   Right = ['State11', 5]),
	    State11 =dict (Up =['State8', 5], Left = ['State10', 5],  Right = ['State12', 5]),
	    State12 =dict (Up =['State9', 5], Left = ['State11', 5]),
	    State13 =dict (Down =['State16', 5], Suck = ['State25', 3],  Right = ['State14', 5]),  
	    State14 =dict (Down =['State17', 5], Left = ['State13', 5],  Right = ['State15', 5]),  
	    State15 =dict (Down =['State18', 5], Suck = ['State39', 3],  Left = ['State14', 5]),  
	    State16 =dict (Up =['State13', 5],  Right = ['State17', 5]),  
	    State17 =dict (Up =['State14', 5], Left = ['State16', 5] , Right = ['State18', 5]),  
	    State18 =dict (Up =['State15', 5], Left = ['State17', 5] ),
	    State19 =dict (Down =['State22', 5], Suck = ['State31', 3],  Right = ['State20', 5]),  
	    State20 =dict (Down =['State23', 5], Suck = ['State38', 3], Left = ['State19', 5],  Right = ['State21', 5]),  
	    State21 =dict (Down =['State24', 5],  Left = ['State20', 5]),  
	    State22 =dict (Up =['State19', 5],   Right = ['State23', 5]),  
	    State23 =dict (Up =['State20', 5], Left = ['State22', 5],  Right = ['State24', 5]),  
	    State24 =dict (Up =['State21', 5],  Left = ['State23', 5]),  
	    State25 =dict (Down =['State28', 3],   Right = ['State26', 3]),  
	    State26 =dict (Up =['State23', 5], Left = ['State22', 5],  Right = ['State24', 5]),  
	    State27 =dict (Down =['State30', 3],  Left = ['State26', 3], Suck = ['State45', 1]), 
	    State28 =dict (Up =['State25', 3],   Right = ['State29', 3]),  
	    State29 =dict (Up =['State26', 3], Left = ['State28', 3],  Right = ['State30', 3]),  
	    State30 =dict (Up =['State27', 3],  Left = ['State29', 3]),  
	    State31 =dict (Down =['State34', 3],   Right = ['State32', 3]),  
	    State32 =dict (Down =['State35', 3], Left = ['State31', 3],  Right = ['State33', 3], Suck = ['State44', 1]),  
	    State33 =dict (Down =['State36', 3],  Left = ['State32', 3]),  
	    State34 =dict (Up =['State31', 3],   Right = ['State35', 3]),  
	    State35 =dict (Up =['State32', 3], Left = ['State34', 3],  Right = ['State36', 3]),  
	    State36 =dict (Up =['State33', 3],  Left = ['State35', 3]), 
	    State37 =dict (Down =['State40', 3],   Right = ['State38', 3], Suck = ['State43', 1]),  
	    State38 =dict (Down =['State41', 3], Left = ['State37', 3],  Right = ['State39', 3]),  
	    State39 =dict (Down =['State42', 3],  Left = ['State38', 3]),  
	    State40 =dict (Up =['State37', 3],   Right = ['State41', 3]),  
	    State41 =dict (Up =['State38', 3], Left = ['State40', 3],  Right = ['State42', 3]),  
	    State42 =dict (Up =['State39', 3],  Left = ['State41', 3]), 
	    State43 =dict (Down =['State46', 1],  Right = ['State44', 1]) ,
	    State44 =dict (Down =['State47', 1],  Left = ['State43', 1]) ,
	    State45 =dict (Down =['State48', 1],  Left = ['State44', 1]),
	    State46 =dict (Up =['State43', 1],  Right = ['State47', 1]),
	    State47 =dict (Up =['State44', 1],  Left = ['State46', 1]),
	    State48 =dict (Up =['State45', 1],  Left = ['State47', 1]) 
    ))

	Vacuum_map.percepts = dict(
	    State1=[1, 1, 1, 1], State2 = [1, 1, 1, 2], State3 = [1, 1, 1, 3],
	    State4=[1, 1, 1, 4], State5 = [1, 1, 1, 5], State6= [1, 1, 1, 6],
	    State7=[0, 1, 1, 1], State8 = [0, 1, 1, 2], State9 = [0, 1, 1, 3],
	    State10=[0, 1, 1, 4], State11 = [0, 1, 1, 5], State12= [0, 1, 1, 6],
	    State13 =[1, 0, 1, 1], State14 = [1, 0, 1, 2], State15 = [ 1, 0, 1, 3],
	    State16=[ 1, 0, 1, 4], State17 = [ 1, 0, 1, 5], State18= [1,0, 1, 6],
	    State19=[1, 1, 0, 1], State20 = [1, 1, 0, 2], State21 = [1, 1, 0, 3],
	    State22=[1, 1, 0, 4], State23 = [1, 1, 0, 5], State24= [1, 1, 0, 6],
	    State25=[0, 0, 1, 1], State26 = [0, 0, 1, 2], State27 = [0, 0, 1, 3],
	    State28=[0, 0, 1, 4], State29 = [0, 0, 1, 5], State30= [0, 0, 1, 6],
	    State31=[0, 1, 0, 1], State32 = [0, 1, 0, 2], State33 = [0, 1, 0, 3],
	    State34=[0, 1, 0, 4], State35 = [0, 1, 0, 5], State36= [0, 1, 0, 6],
	    State37=[1, 0, 0, 1], State38 = [1, 0, 0, 2], State39 = [ 1, 0, 0, 3],
	    State40=[ 1, 0, 0, 4], State41 = [ 1, 0, 0, 5], State42= [1,0, 0, 6],
	    State43 =[0, 0, 0, 1], State44 = [0, 0, 0, 2], State45 = [ 0, 0, 0, 3],
	    State46 =[ 0, 0, 0, 4], State47 = [ 0, 0, 0, 5], State48 = [0, 0, 0, 6]
	    )

	Vacuum_Problem = NewGraphProblem('State5', ['State43', 'State44', 'State45', 'State46', 'State47', 'State48'], Vacuum_map)
	h=memoize(Vacuum_Problem.h, 'h')
	[iter, all_colors, new_node]=astar_search(Vacuum_Problem, lambda n:n.path_cost + h(n))

	new_path = new_node.path()
	print(new_path)
	len(new_path)
	print(new_node.state)
	print(new_node.path_cost, new_node.action)
	print(new_node.f)
	parent_node = new_node.parent
	print(parent_node.state)
	print(parent_node.path_cost, parent_node.action)
	print(parent_node.f)
	grad_parent = parent_node.parent
	print(grad_parent.state)
	print(grad_parent.path_cost, grad_parent.action)
	print(grad_parent.f)
	gg_p=grad_parent.parent
	print(gg_p.state)
	print(gg_p.path_cost, gg_p.action)
	print(gg_p.f)
	ggg_p = gg_p.parent
	print(ggg_p.state)
	print(ggg_p.path_cost, ggg_p.action)
	print(ggg_p.f)
	g4_p = ggg_p.parent
	print(g4_p.state)
	print(g4_p.path_cost, g4_p.action)
	print(g4_p.f)
	g5_p = g4_p.parent
	print(g5_p.state)
	print(g5_p.path_cost, g5_p.action)
	print(g5_p.f)
	g6_p=g5_p.parent
	print(g6_p.state)
	print(g6_p.path_cost, g6_p.action)
	print(g6_p.f)

	#astar_search(Vacuume_Problem)

if __name__ == "__main__":main()