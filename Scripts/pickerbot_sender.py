import socket
from time import sleep

# ip_adddress = "10.5.x.x" # real robot
ip_adddress = "127.0.0.1" # simulator robot

clientSocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
clientSocket.connect((ip_adddress, 2001))

def _send_command(command, x, y, z, u=0):
    """Centralized helper to format and send TCP/IP commands to EPSON."""
    coordinates = f"{command} {x} {y} {z} {u}\r\n"
    print(f"--> Sending: {command} to World Position X={x}, Y={y}, Z={z}, U={u}")
    
    clientSocket.send(coordinates.encode())
    confirmation = clientSocket.recv(1023)
    
    print("--> EPSON Reply:", confirmation.decode().strip())
    sleep(1)

# Clean, one-line functions that reuse the networking logic
def epsonGo(x=0, y=470, z=360, u=0): _send_command("GO", x, y, z, u)
def epsonJump(x=0, y=470, z=360, u=0): _send_command("JUMP", x, y, z, u)
def epsonMove(x=0, y=470, z=360, u=0): _send_command("MOVE", x, y, z, u)
def epsonPick(x=0, y=470, z=360, u=0): _send_command("PICK", x, y, z, u)

# Example usage
epsonPick(0, 470, 360, 25)

# Fix the bug by adding parentheses to properly close the connection
clientSocket.close()