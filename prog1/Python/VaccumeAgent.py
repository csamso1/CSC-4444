#Clayton Samson
#CSC 4444 HW 1 Code
#CS444455

from agents import ReflexVacuumAgent, ModelBasedVacuumAgent, TrivialVacuumEnvironment, TraceAgent, compare_agents
import copy
def main():
	for i in range(4):
		print("-"*50)
		print("Itteration: ", i+1)
		print("Test data")
	clean_up()
	exit(0)

	print("this is a test")

def clean_up():
	#Seting Environment
	env1 = TrivialVacuumEnvironment()
	env2 = copy.deepcopy(env1)
	env3 = copy.deepcopy(env1)
	envStatus = env1.status
    
    #Setting up Agents
	vac1 = ReflexVacuumAgent()
	vac2 = ModelBasedVacuumAgent()
	vac3 = ModelBasedVacuumAgent()
	#Model based vaccume initial scan
	for i in range(3):
		vac3.program((list(env3.status)[i], list(env3.status.values())[i]))
	env1.add_thing(TraceAgent(vac1))
	env2.add_thing(TraceAgent(vac2))
	env3.add_thing(TraceAgent(vac3))
	print("Starting room configurations: ", envStatus)
	print("-"*50)
	#Executing Simple Reflex Vaccume Run
	print("#"*50)
	print("Simple Reflex Agent Run:")
	print("#"*50)
	cleaning1 = env1.run(100)
	simple_performance = vac1.performance
	#Executing Model Based Vaccume Run
	print("#"*50)
	print("Model Based Vaccume Run:")
	print("#"*50)
	cleaning2 = env2.run(100)
	MB_performance = vac2.performance
	#Executing Advanced Model Based Vaccume Run
	print("#"*50)
	print("Advanced Model Based Vaccume Run:")
	print("#"*50)
	cleaning3 = env3.run(100)
	AMB_performance = vac3.performance
	print("Simpile Refelex Agent performance: ", simple_performance)
	print("Model Based Agent performance: ", MB_performance)
	print("Advanced Refelex Agent performance: ", AMB_performance)

main()