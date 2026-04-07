# Python library to send command to EPSON VT6 via TCP/IP
# by Judhi Prasetyo April 2023

import socket
from time import sleep
import numpy as np

#ip_adddress = "192.168.150.2" # real robot
ip_adddress = "127.0.0.1" # simulator robot

## Create a client socket
clientSocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
## Connect to the EPSON robot
clientSocket.connect((ip_adddress,2001))


# format of the coordinate is "x y z u" where u is the wrist rotation angle
# example:
# places = ["100 400 600 0", "0 500 500 0", "-100 600 400 0"]

robot_z = 360 # default z height for picking, can be changed based on the object height


def epsonGo(x = 0, y = 450, robot_z = robot_z, robot_u = 0):
    coordinates = "GO " + f"{x} {y} {robot_z} {robot_u}" + "\r\n"
    print (f"Going to position {x}, {y}, {robot_z}, {robot_u}")
    clientSocket.send(coordinates.encode())
    confirmation = clientSocket.recv(1023) # waiting for confirmation from robot
    print("result:", confirmation)
    sleep(1)

def epsonJump(x = 0, y = 450, robot_z = robot_z, robot_u = 0):
    coordinates = "JUMP " + f"{x} {y} {robot_z} {robot_u}" + "\r\n"
    print (f"Jumping to position {x}, {y}, {robot_z}, {robot_u}")
    clientSocket.send(coordinates.encode())
    confirmation = clientSocket.recv(1023) # waiting for confirmation from robot
    print("result:", confirmation)
    sleep(1)

def epsonPick(x = 0, y = 450, robot_z = robot_z, robot_u = 0):
    coordinates = "PICK " + f"{x} {y} {robot_z} {robot_u}" + "\r\n"
    print (f"Picking at position {x}, {y}, {robot_z}, {robot_u}")
    clientSocket.send(coordinates.encode())
    confirmation = clientSocket.recv(1023) # waiting for confirmation from robot
    print("result:", confirmation)
    sleep(1)

# example, please remove when using this file as library
# epsonJump(0,500,200, 90)
# epsonGo(0,400,500, 90)
epsonPick(0,470,360, 25)

clientSocket.close # close the connection