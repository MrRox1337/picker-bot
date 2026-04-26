import socket
from time import sleep

from pickerbot_lib.config import CONFIG

clientSocket = None

def connect(ip=None, port=None):
    """Connect to the EPSON robot controller. Call before using any command functions."""
    global clientSocket
    if ip is None:
        ip = CONFIG["epson_ip"]
    if port is None:
        port = CONFIG["epson_port"]
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

    confirmation = clientSocket.recv(1023).decode().strip()
    print("--> EPSON Reply:", confirmation)

    sleep(0.5)
    return confirmation

def epsonGo(x=0, y=470, z=None, u=0):
    return _send_command("GO", x, y, CONFIG["robot_z"] if z is None else z, u)

def epsonJump(x=0, y=470, z=None, u=0):
    return _send_command("JUMP", x, y, CONFIG["robot_z"] if z is None else z, u)

def epsonMove(x=0, y=470, z=None, u=0):
    return _send_command("MOVE", x, y, CONFIG["robot_z"] if z is None else z, u)

def epsonPick(x=0, y=470, z=None, u=0):
    return _send_command("PICK", x, y, CONFIG["robot_z"] if z is None else z, u)

def epsonStandby():
    return _send_command("STANDBY", 0, 470, CONFIG["robot_z"], 0)

def epsonPickAll(locations):
    """Iterates through an array and only proceeds if the robot replies with 'OK'."""
    print(f"Starting batch pick operation for {len(locations)} locations...")

    for loc in locations:
        if isinstance(loc, str) and len(loc.split()) == 4:
            x, y, z, u = map(float, loc.split())
        elif isinstance(loc, (list, tuple)) and len(loc) == 4:
            x, y, z, u = loc
        else:
            print(f"--> Error: Invalid format '{loc}'. Skipping.")
            continue

        reply = epsonPick(x, y, z, u)

        if "OK" not in reply.upper():
            print(f"--> ABORTING BATCH: Robot returned non-OK status: '{reply}'")
            break

    print("Batch pick operation sequence ended.")
    print("Going to standby position...")
    epsonStandby()

if __name__ == "__main__":
    connect()
    target_locations = [[0, 470, 360, 25], [50, 470, 360, 0]]
    epsonPickAll(target_locations)
