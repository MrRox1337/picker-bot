import socket
from time import sleep

clientSocket = None

def connect(ip="127.0.0.1", port=2001):
    """Connect to the EPSON robot controller. Call before using any command functions."""
    global clientSocket
    # ip = "192.168.150.2" # real robot
    print(f"Connecting to EPSON at {ip}:{port}...")
    clientSocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    clientSocket.connect((ip, port))
    print("Connected.")

def disconnect():
    """Close the TCP/IP connection to the EPSON robot controller."""
    global clientSocket
    if clientSocket:
        clientSocket.close()
        clientSocket = None
        print("Disconnected.")

def _send_command(command, x, y, z, u=0):
    """Centralized helper to format and send TCP/IP commands to EPSON."""
    coordinates = f"{command} {x} {y} {z} {u}\r\n"
    print(f"\n--> Sending: {command} to World Position X={x}, Y={y}, Z={z}, U={u}")

    clientSocket.send(coordinates.encode())

    # Wait for the robot to finish and reply, then decode it to a string
    confirmation = clientSocket.recv(1023).decode().strip()
    print("--> EPSON Reply:", confirmation)

    sleep(0.5) # Short buffer before the next command
    return confirmation # Return the string so other functions can check it

def epsonGo(x=0, y=470, z=360, u=0): return _send_command("GO", x, y, z, u)
def epsonJump(x=0, y=470, z=360, u=0): return _send_command("JUMP", x, y, z, u)
def epsonMove(x=0, y=470, z=360, u=0): return _send_command("MOVE", x, y, z, u)
def epsonPick(x=0, y=470, z=360, u=0): return _send_command("PICK", x, y, z, u)
def epsonStandby(): return _send_command("STANDBY", 0, 470, 360, 0)

def epsonPickAll(locations):
    """Iterates through an array and only proceeds if the robot replies with 'OK'."""
    print(f"Starting batch pick operation for {len(locations)} locations...")

    for loc in locations:
        # 1. Parse the coordinates
        if isinstance(loc, str) and len(loc.split()) == 4:
            x, y, z, u = map(float, loc.split())
        elif isinstance(loc, (list, tuple)) and len(loc) == 4:
            x, y, z, u = loc
        else:
            print(f"--> Error: Invalid format '{loc}'. Skipping.")
            continue

        # 2. Send the pick command and capture the reply
        reply = epsonPick(x, y, z, u)

        # 3. Halt the batch if the status is not OK
        # (Using .upper() to catch "ok", "Ok", or "OK")
        if "OK" not in reply.upper():
            print(f"--> ABORTING BATCH: Robot returned non-OK status: '{reply}'")
            break # This breaks the loop, stopping any future picks

    print("Batch pick operation sequence ended.")
    print("Going to standby position...")
    epsonStandby() # Move to standby position

if __name__ == "__main__":
    connect()
    # Example usage:
    target_locations = [[0, 470, 360, 25], [50, 470, 360, 0]]
    epsonPickAll(target_locations)
