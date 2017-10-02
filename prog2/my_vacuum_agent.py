import sys
sys.path.append('/mnt/c/Users/Clayton/Documents/GitHub/CSC-4444/prog2/aima-python-master')
from search import *
from utils import (
    is_in, argmin, argmax, argmax_random_tie, probability, weighted_sampler,
    memoize, print_table, open_data, Stack, FIFOQueue, PriorityQueue, name,
    distance
)

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

node_colors = dict()
initial_node_colors = dict(node_colors)

def best_first_graph_search(problem, f):
    """Search the nodes with the lowest f scores first.
    You specify the function f(node) that you want to minimize; for example,
    if f is a heuristic estimate to the goal, then we have greedy best
    first search; if f is node.depth then we have breadth-first search.
    There is a subtlety: the line "f = memoize(f, 'f')" means that the f
    values will be cached on the nodes as they are computed. So after doing
    a best first search you can examine the f values of the path returned."""
    
    # we use these two variables at the time of visualisations
    iterations = 0
    all_node_colors = []
    node_colors = dict(initial_node_colors)
    
    f = memoize(f, 'f')
    node = Node(problem.initial)
    
    node_colors[node.state] = "red"
    iterations += 1
    all_node_colors.append(dict(node_colors))
    
    if problem.goal_test(node.state):
        node_colors[node.state] = "green"
        iterations += 1
        all_node_colors.append(dict(node_colors))
        return(iterations, all_node_colors, node)
    
    frontier = PriorityQueue(min, f)
    frontier.append(node)
    
    node_colors[node.state] = "orange"
    iterations += 1
    all_node_colors.append(dict(node_colors))
    
    explored = set()
    while frontier:
        node = frontier.pop()
        
        node_colors[node.state] = "red"
        iterations += 1
        all_node_colors.append(dict(node_colors))
        
        if problem.goal_test(node.state):
            node_colors[node.state] = "green"
            iterations += 1
            all_node_colors.append(dict(node_colors))
            return(iterations, all_node_colors, node)
        
        explored.add(node.state)
        for child in node.expand(problem):
            if child.state not in explored and child not in frontier:
                frontier.append(child)
                node_colors[child.state] = "orange"
                iterations += 1
                all_node_colors.append(dict(node_colors))
            elif child in frontier:
                incumbent = frontier[child]
                if f(child) < f(incumbent):
                    del frontier[incumbent]
                    frontier.append(child)
                    node_colors[child.state] = "orange"
                    iterations += 1
                    all_node_colors.append(dict(node_colors))

        node_colors[node.state] = "gray"
        iterations += 1
        all_node_colors.append(dict(node_colors))
    return None

class Problem(object):

    """The abstract class for a formal problem.  You should subclass
    this and implement the methods actions and result, and possibly
    __init__, goal_test, and path_cost. Then you will create instances
    of your subclass and solve them with the various search functions."""

    def __init__(self, initial, goal=None):
        """The constructor specifies the initial state, and possibly a goal
        state, if there is a unique goal.  Your subclass's constructor can add
        other arguments."""
        self.initial = initial
        self.goal = goal

    def actions(self, state):
        """Return the actions that can be executed in the given
        state. The result would typically be a list, but if there are
        many actions, consider yielding them one at a time in an
        iterator, rather than building them all at once."""
        raise NotImplementedError

    def result(self, state, action):
        """Return the state that results from executing the given
        action in the given state. The action must be one of
        self.actions(state)."""
        raise NotImplementedError

    def goal_test(self, state):
        """Return True if the state is a goal. The default method compares the
        state to self.goal or checks for state in self.goal if it is a
        list, as specified in the constructor. Override this method if
        checking against a single self.goal is not enough."""
        if isinstance(self.goal, list):
            return is_in(state, self.goal)
        else:
            return state == self.goal

    def path_cost(self, c, state1, action, state2):
        """Return the cost of a solution path that arrives at state2 from
        state1 via action, assuming cost c to get up to state1. If the problem
        is such that the path doesn't matter, this function will only look at
        state2.  If the path does matter, it will consider c and maybe state1
        and action. The default method costs 1 for every step in the path."""
        return c + 1

    def value(self, state):
        """For optimization problems, each state has a value.  Hill-climbing
        and related algorithms try to maximize this value."""
        raise NotImplementedError

class Graph:

    """A graph connects nodes (verticies) by edges (links).  Each edge can also
    have a length associated with it.  The constructor call is something like:
        g = Graph({'A': {'B': 1, 'C': 2})
    this makes a graph with 3 nodes, A, B, and C, with an edge of length 1 from
    A to B,  and an edge of length 2 from A to C.  You can also do:
        g = Graph({'A': {'B': 1, 'C': 2}, directed=False)
    This makes an undirected graph, so inverse links are also added. The graph
    stays undirected; if you add more links with g.connect('B', 'C', 3), then
    inverse link is also added.  You can use g.nodes() to get a list of nodes,
    g.get('A') to get a dict of links out of A, and g.get('A', 'B') to get the
    length of the link from A to B.  'Lengths' can actually be any object at
    all, and nodes can be any hashable object."""

    def __init__(self, dict=None, directed=True):
        self.dict = dict or {}
        self.directed = directed
        if not directed:
            self.make_undirected()

    def make_undirected(self):
        """Make a digraph into an undirected graph by adding symmetric edges."""
        for a in list(self.dict.keys()):
            for (b, dist) in self.dict[a].items():
                self.connect1(b, a, dist)

    def connect(self, A, B, distance=1):
        """Add a link from A and B of given distance, and also add the inverse
        link if the graph is undirected."""
        self.connect1(A, B, distance)
        if not self.directed:
            self.connect1(B, A, distance)

    def connect1(self, A, B, distance):
        """Add a link from A to B of given distance, in one direction only."""
        self.dict.setdefault(A, {})[B] = distance

    def get(self, a, b=None):
        """Return a link distance or a dict of {node: distance} entries.
        .get(a,b) returns the distance or None;
        .get(a) returns a dict of {node: distance} entries, possibly {}."""
        links = self.dict.setdefault(a, {})
        if b is None:
            return links
        else:
            return links.get(b)

    def nodes(self):
        """Return a list of nodes in the graph."""
        return list(self.dict.keys())

def UndirectedGraph(dict=None):
    """Build a Graph where every edge (including future ones) goes both ways."""
    return Graph(dict=dict, directed=False)

class GraphProblem(Problem):

    """The problem of searching a graph from one node to another."""

    def __init__(self, initial, goal, graph):
        Problem.__init__(self, initial, goal)
        self.graph = graph

    def actions(self, A):
        """The actions at a graph node are just its neighbors."""
        return list(self.graph.get(A).keys())

    def result(self, state, action):
        """The result of going to a neighbor is just that neighbor."""
        return action

    def path_cost(self, cost_so_far, A, action, B):
        return cost_so_far + (self.graph.get(A, B) or infinity)

    def find_min_edge(self):
        """Find minimum value of edges."""
        m = infinity
        for d in self.graph.dict.values():
            local_min = min(d.values())
            m = min(m, local_min)

        return m

    def h(self, node):
        """h function is straight-line distance from a node's state to goal."""
        locs = getattr(self.graph, 'locations', None)
        if locs:
            if type(node) is str:
                return int(distance(locs[node], locs[self.goal]))

            return int(distance(locs[node.state], locs[self.goal]))
        else:
            return infinity

class NewGraphProblem(GraphProblem):

    "The problem of searching a graph from one node to another."

    def __init__(self, initial, goal, graph):
        Problem.__init__(self, initial, goal)
        self.graph = graph

    def actions(self, A):
        "The actions at a graph node are operators such as Left, Right, Suck, etc."
        return list(self.graph.get(A).keys())

    def result(self, state, action):  # state is  the name of a graph node,which really does not describe the state
                                       # the actual state info. will be in a dictionary called percepts
        "The result of performing an action is the first element in a list (new_st, cost)"
        (new_state, distance) = self.graph.get(state)[action]   # here we equal node with its state
        return new_state  # this has to be modified - the result must be another node in the graph

    def path_cost(self, cost_so_far, A, action, B):
        (new_state, distance) = self.graph.get(A)[action] 
        print('distance=')
        print(distance)
        return int(cost_so_far + (distance or infinity))

    def h(self, node):
        "h function is straight-line distance from a node's state to goal."
        percts  = getattr(self.graph, 'percepts', None)
        #print('percept=')
        #print(percts)
        if percts: 
            print('node.state=', node.state)
            perc = percts[node.state]         # get the configuration which is a vec of length 4
            print('perc[1]=', perc[1])
            if perc[1]== perc[2]==perc[0]== 0:
                return int(0)
            sum1 = perc[1]+perc[2]+perc[0]
            print('sum1=', sum1)
            Loc = perc[3]
            print('Loc = ', Loc)
            if sum1 == 3 and Loc >= 4:
                print('First Branch')
                return int(2*sum1 + 2*sum1-1+4*(sum1-1)+4)
            if sum1 == 3 and Loc < 4:
                return int(2*sum1-1+4*(sum1-1)+4)
            if sum1 == 2 and Loc >= 4:
                return int(2*sum1 + 2*sum1-1+4)
            if sum1 ==2 and  Loc <4:
                return int(2*sum1-1+4)
            if sum1 == 1 and Loc >= 4:
                return 2+1
            if sum1 == 1 and Loc  <4:
                return int(1)
        else:
            return infinity

def astar_search(problem, h=None):
    """A* search is best-first graph search with f(n) = g(n)+h(n).
    You need to specify the h function when you call astar_search, or
    else in your Problem subclass."""
    h = memoize(h or problem.h, 'h')
    iterations, all_node_colors, node=best_first_graph_search(problem, lambda n: n.path_cost + h(n))
    return(iterations, all_node_colors, node)

def main():
	

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