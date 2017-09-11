/**
 * @author Clayton Samson
 * CSC 4444 - Homework 1 Code
 * CSamso1@LSU.edu
 * cs444438
 */

import java.io.*;
import java.util.Arrays;
import java.util.Scanner;

public class Hw1Q1A
{
	public static int[] rooms_status = new int[3];
	public static String current_room;
	public static void main(String[] args) throws FileNotFoundException, Exception
	{
		try{
			rooms_status[0] = Integer.parseInt(args[0]);
			rooms_status[1] = Integer.parseInt(args[1]);
			rooms_status[2] = Integer.parseInt(args[2]);
			current_room = args[3];
		} catch(Exception e){
	        throw new Exception("ERROR: Invalid command line args. Please provide either 0 or 1 for each room (0 = clean, 1 = dirty)\n and either 'A', 'B', or 'C' to note the starting room.", e);
	    }
	    System.out.printf("Initial room status:\n");
	    print_room_status();
	    print_current_room();
	}
	
	//Method for printing the status of the 3 rooms in a clear manner
	public static void print_room_status()
	{
		String room_a_state;
		String room_b_state;
		String room_c_state;
		if(rooms_status[0] == 0)
		{
			room_a_state = "Clean";
		}
		else
		{
			room_a_state = "Dirty";
		}
		if(rooms_status[1] == 0)
		{
			room_b_state = "Clean";
		}
		else
		{
			room_b_state = "Dirty";
		}
		if(rooms_status[2] == 0)
		{
			room_c_state = "Clean";
		}
		else
		{
			room_c_state = "Dirty";
		}
		System.out.printf("Current Room Status: [Room A: %s, Room B: %s, Room C: %s]\n", room_a_state, room_b_state, room_c_state);
	}

	public static void print_current_room()
	{
		System.out.printf("Current Location of agent: Room %s\n", current_room);
	}
}