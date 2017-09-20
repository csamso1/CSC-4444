#Clayton Samson
#CSC 4444 HW 1 Code
#CS444455

from agents import ReflexVacuumAgent, ModelBasedVacuumAgent, BrokenLocationVacuumAgent, RandomVacuumAgent, TrivialVacuumEnvironment, TraceAgent, compare_agents
import copy
def main():
	for i in range(4):
		print("-"*60)
		print("Itteration: ", i+1)
		clean_up()
		clean_up2()
	exit(0)

def clean_up():
	#Seting Environment
	env1 = TrivialVacuumEnvironment()
	env2 = copy.deepcopy(env1)
	env3 = copy.deepcopy(env1)
	env4 = copy.deepcopy(env1)
	envStatus = env1.status
    
    #Setting up Agents
	vac1 = ReflexVacuumAgent()
	vac2 = ModelBasedVacuumAgent()
	vac3 = ModelBasedVacuumAgent()
	vac4 = BrokenLocationVacuumAgent()
	env1.add_thing(TraceAgent(vac1))
	env2.add_thing(TraceAgent(vac2))
	env3.add_thing(TraceAgent(vac3))
	env4.add_thing(TraceAgent(vac4))
	#Model based vaccume initial scan
	for location, status in env3.status:
		print("env3 status: ", location, status)
	print("Starting room configurations: ", envStatus)
	print("-"*60)
	#Executing Simple Reflex Vaccume Run
	print("#"*60)
	print("Simple Reflex Agent Run:")
	print("#"*60)
	cleaning1 = env1.run(100)
	simple_performance = vac1.performance
	#Executing Model Based Vaccume Run
	print("#"*60)
	print("Model Based Vaccume Run:")
	print("#"*60)
	cleaning2 = env2.run(100)
	MB_performance = vac2.performance
	#Executing Advanced Model Based Vaccume Run
	print("#"*60)
	print("Advanced Model Based Vaccume Run:")
	print("#"*60)
	cleaning3 = env3.run(100)
	AMB_performance = vac3.performance
	print("#"*60)
	print("Reflex Vacuum Agent With Broken Location Sensor Run:")
	print("#"*60)
	cleaning4 = env4.run(100)
	broken_senesor_performance = vac4.performance
	print("Simpile Refelex Agent performance: ", simple_performance)
	print("Model Based Agent performance: ", MB_performance)
	print("Advanced Refelex Agent performance: ", AMB_performance)
	print("Reflex Vacuum Agent With Broken Location Sensor Performance: ", broken_senesor_performance)

def clean_up2():
	env1 = TrivialVacuumEnvironment()
	env2 = copy.deepcopy(env1)
	reflex_vac = BrokenLocationVacuumAgent()
	random_vac = RandomVacuumAgent()
	env1.add_thing(TraceAgent(reflex_vac))
	env2.add_thing(TraceAgent(random_vac))
	print("Deterministic simple reflex agent run: ")
	cleaning1 = env1.run(100)
	print("Simple Reflex Agent with randomized agent run: ")
	cleaning2 = env2.run(100)
main()