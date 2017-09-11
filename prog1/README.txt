Clayton Samson
CSamso1@LSU.edu
CS444438

My code for HW1 has been written in 4 different java programs, the names correspond to the HW question numbers they are answering.  Java Program file names are noted below:

	- Question 1A: Hw1Q1A
	- Question 1B: Hw1Q1B
	- Question 1C: Hw1Q1C
	- Question 2: Hw1Q2

To complie the programs use a standard 'javac FileName.java' to compile each of the 4 programs.

	- javac Hw1Q1A.java
	- javac Hw1Q1B.java
	- javac Hw1Q1C.java
	- javac Hw1Q2.java

Each of the 4 programs is run using the same parameter format, you will need to specify the starting condition of each of the 3 rooms ('0' for clean, '1' for dirty) and the starting location of the agent ('a', 'b', or 'c').  Examples below:

	- to run the first program with only room B dirty and starting location of room a:
	- java Hw1Q1A 0 1 0 a

	- to run the 2nd program with every room dirty and the starting location of room b:
	- java Hw1Q1B 1 1 1 b

Each of the programs will print out the starting condition of the 3 rooms and update the states as it preforms actions.