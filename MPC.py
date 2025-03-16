import cvxpy as cp
import serial
import time
import math
import threading
import tkinter as tk
from tkinter import ttk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import socket
import numpy as np
# -----------------------------
# MPC Parameters
# -----------------------------
N = 10  # Prediction horizon
dt = 1.0  # Time step (seconds)
Q_heading = 1.0
Q_xte = 2.0
R_rz = 0.1
R_u = 0.1
max_rz_thrust = 100.0

serial_port = 'COM5'
baud_rate = 115200
ser = serial.Serial(serial_port, baud_rate, timeout=1)
path = []
start_pos = None  # Will be set to the first valid RMC reading
def degrees_to_radians(deg):
    return deg * math.pi / 180

def calculate_bearing(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(degrees_to_radians, [lat1, lon1, lat2, lon2])
    dlon = lon2 - lon1
    x = math.sin(dlon) * math.cos(lat2)
    y = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(dlon)
    bearing = math.atan2(x, y)
    return (math.degrees(bearing) + 360) % 360

def calculate_distance(lat1, lon1, lat2, lon2):
    R = 6378137
    lat1, lon1, lat2, lon2 = map(degrees_to_radians, [lat1, lon1, lat2, lon2])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = math.sin(dlat/2)**2 + math.cos(lat1)*math.cos(lat2)*math.sin(dlon/2)**2
    return R * (2 * math.atan2(math.sqrt(a), math.sqrt(1-a)))

def normalize_angle(angle):
    while angle > 180:
        angle -= 360
    while angle < -180:
        angle += 360
    return angle

def calculate_checksum(command):
    checksum = 0
    for char in command[1:]:
        checksum ^= ord(char)
    return f"{checksum:02X}"

def send_command(command):
    chk = calculate_checksum(command)
    full_command = f"{command}*{chk}\r\n"
    ser.write(full_command.encode())
    time.sleep(0.1)
    response = ser.read(ser.in_waiting).decode('utf-8')
    if response:
        print("Response:", response.strip())

def parse_rmc(rmc_sentence):
    parts = rmc_sentence.split(',')
    if parts[0] != '$GPRMC' or parts[2] == '':
        return None, None, None, None
    lat = float(parts[3][:2]) + float(parts[3][2:]) / 60
    if parts[4] == 'S':
        lat = -lat
    lon = float(parts[5][:3]) + float(parts[5][3:]) / 60
    if parts[6] == 'W':
        lon = -lon
    speed_knots = float(parts[7]) if parts[7] else 0.0
    speed_mps = speed_knots * 0.514444
    heading = float(parts[8]) if parts[8] else 0.0
    return lat, lon, heading, speed_mps

def parse_waypoints(filename):
    waypoints = []
    with open(filename, 'r') as file:
        for line in file:
            parts = line.strip().split(',')
            if parts[0] == '$MMWPL':
                lat = float(parts[1][:2]) + float(parts[1][2:]) / 60
                if parts[2] == 'S':
                    lat = -lat
                lon = float(parts[3][:3]) + float(parts[3][3:]) / 60
                if parts[4] == 'W':
                    lon = -lon
                waypoints.append((lat, lon))
    return waypoints

def calculate_signed_xte(current_pos, previous_wp, current_wp):
    latA_rad = math.radians(previous_wp[0])
    x_current = (current_pos[1] - previous_wp[1]) * 111319.5 * math.cos(latA_rad)
    y_current = (current_pos[0] - previous_wp[0]) * 111319.5
    x_target = (current_wp[1] - previous_wp[1]) * 111319.5 * math.cos(latA_rad)
    y_target = (current_wp[0] - previous_wp[0]) * 111319.5
    cross = x_target * y_current - y_target * x_current
    path_length = math.hypot(x_target, y_target)
    return cross / path_length if path_length != 0 else 0.0


BASE_FORWARD_THRUST = 84.70    # Corresponds to 5.0 knots
MIN_FORWARD_THRUST = 7.00     # Corresponds to 1.0 knot
TURNING_THRESHOLD = 10     # Degrees error above which full speed reduction is applied
root = tk.Tk()
root.title("Navigation GUI")

fig, ax = plt.subplots(figsize=(15, 15))
canvas = FigureCanvasTkAgg(fig, master=root)
canvas.get_tk_widget().pack()

status_label = ttk.Label(root, text="Initialising...", font=("Arial", 15))
status_label.pack()
speed_label = ttk.Label(root, text="Speed: --.- kts", font=("Arial", 15))
speed_label.pack()
heading_label = ttk.Label(root, text="Heading: ---°", font=("Arial", 15))
heading_label.pack()
xte_label = ttk.Label(root, text="XTE: --.- m", font=("Arial", 15))
xte_label.pack()
distance_label = ttk.Label(root, text="Distance: --.- m", font=("Arial", 15))
distance_label.pack()

# Define ship dynamics model (simplified for heading and position)
import numpy as np

def mpc_controller(current_state, ref_bearing, ref_distance, ref_xte):
    # Initialize variables as numpy arrays
    rz_thrust = np.zeros(N)
    forward_thrust = np.zeros(N)
    
    heading = np.zeros(N+1)
    xte = np.zeros(N+1)
    distance = np.zeros(N+1)
    
    # Initial conditions
    heading[0] = current_state['heading']
    xte[0] = ref_xte
    distance[0] = ref_distance

    # Simple kinematic model with manual calculation
    for t in range(N):
        # Kinematic model for each timestep
        heading[t+1] = heading[t] + rz_thrust[t] * dt
        xte[t+1] = xte[t] + forward_thrust[t] * np.sin(np.radians(heading[t])) * dt
        distance[t+1] = distance[t] - forward_thrust[t] * np.cos(np.radians(heading[t])) * dt

        # Apply constraints
        rz_thrust[t] = np.clip(rz_thrust[t], -max_rz_thrust, max_rz_thrust)
        forward_thrust[t] = np.clip(forward_thrust[t], MIN_FORWARD_THRUST, BASE_FORWARD_THRUST)
    
    # Cost function: penalize heading error, cross-track error, control effort
    cost = 0
    for t in range(N):
        cost += Q_heading * (heading[t] - ref_bearing)**2
        cost += Q_xte * xte[t]**2
        cost += R_rz * rz_thrust[t]**2
        cost += R_u * (forward_thrust[t] - BASE_FORWARD_THRUST)**2

    # Return the first control input
    return rz_thrust[0], forward_thrust[0], cost


# -----------------------------
# Replace this section in your control loop:
# -----------------------------

# ADD THIS BEFORE WHILE TRUE
prev_time = time.time()

# ADD A FUNCTION TO RUN CONTROL LOOP IN A SEPARATE THREAD
def control_loop():
    global start_pos, prev_time
    
    while True:
        send_command('$CCNVO,2,1.0,0,0.0')
        rmc_sentence = ser.readline().decode().strip()
        current_lat, current_lon, current_heading, current_speed = parse_rmc(rmc_sentence)
        if current_lat is None:
            time.sleep(1)
            continue

        if start_pos is None:
            start_pos = (current_lat, current_lon)
        
        waypoints = parse_waypoints('waypoints/waypoints2.txt')
        send_command('$CCAPM,0,64,0,80')
        send_command('$CCTHD,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00')
        time.sleep(1)
        previous_wp = start_pos  # For the first leg

        for index, (waypoint_lat, waypoint_lon) in enumerate(waypoints, start=1):
            current_wp = (waypoint_lat, waypoint_lon)
            status_label.config(text=f"Navigating to Waypoint {index}: ({waypoint_lat}, {waypoint_lon})")
            while True:
                current_time = time.time()
                dt = current_time - prev_time
                prev_time = current_time

                send_command('$CCNVO,2,1.0,0,0.0')
                rmc_sentence = ser.readline().decode().strip()
                current_lat, current_lon, current_heading, current_speed = parse_rmc(rmc_sentence)
                if current_lat is None:
                    time.sleep(1)
                    continue

                bearing = calculate_bearing(current_lat, current_lon, waypoint_lat, waypoint_lon)
                distance = calculate_distance(current_lat, current_lon, waypoint_lat, waypoint_lon)
                xte = calculate_signed_xte((current_lat, current_lon), previous_wp, current_wp)

                # Update GUI labels
                speed_label.config(text=f"Speed: {current_speed:.1f} kts")
                heading_label.config(text=f"Heading: {current_heading:.1f}°")
                xte_label.config(text=f"XTE: {xte:.1f} m")
                distance_label.config(text=f"Distance: {distance:.1f} m")

                # MPC
                current_state = {'heading': current_heading}
                rz_cmd, fwd_cmd = mpc_controller(current_state, bearing, distance, xte)
                print(f"RZ Thrust: {rz_cmd:.2f}, Forward Thrust: {fwd_cmd:.2f}")

                # Send control to vessel
                send_command(f'$CCTHD,{fwd_cmd:.2f},{rz_cmd:.2f},0.00,0.00,0.00,0.00,0.00,0.00')

                if distance < 10:  # Threshold to consider waypoint reached
                    previous_wp = current_wp
                    break

                canvas.draw()
                time.sleep(0.5)

# RUN CONTROL LOOP IN BACKGROUND
threading.Thread(target=control_loop, daemon=True).start()

root.mainloop()

