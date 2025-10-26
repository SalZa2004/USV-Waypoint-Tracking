# Updated and Cleaner Version of the Original Code
import serial
import time
import math
import threading
import tkinter as tk
from tkinter import ttk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import csv
from datetime import datetime
from functools import reduce

# --- Serial Configuration ---
ser = serial.Serial('COM5', 115200, timeout=1)

# --- Constants ---
THRESHOLDS = {
    'heading_error': 5,
    'distance': 10.0,
    'turning': 10
}

THRUST = {
    'max_rz': 100.0,
    'base_fwd': 84.7,
    'min_fwd': 7.0
}

XTE_GAINS = {'Kp': 2.9, 'Ki': 0.032, 'Kd': 0.07}

# --- Global State ---
prev_xte, start_pos = 0.0, None
path = []
log_file = f"log_{datetime.now():%Y%m%d_%H%M%S}.csv"

# --- Logging Setup ---
with open(log_file, 'w', newline='') as f:
    csv.writer(f).writerow([
        "Time", "Lat", "Lon", "Speed (knots)", "Heading (deg)",
        "Distance to WP (m)", "Heading Error (deg)", "XTE (m)", "Thrust"
    ])

# --- GUI Setup ---
root = tk.Tk()
root.title("Navigation GUI")

fig, ax = plt.subplots(figsize=(15, 15))
canvas = FigureCanvasTkAgg(fig, master=root)
canvas.get_tk_widget().pack()

labels = {}
for key in ["status", "speed", "heading", "xte", "wind", "distance"]:
    labels[key] = ttk.Label(root, text=f"{key.capitalize()}: --", font=("Arial", 15))
    labels[key].pack()

# --- Utility Functions ---
def deg2rad(deg): return deg * math.pi / 180

def bearing(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(deg2rad, [lat1, lon1, lat2, lon2])
    dlon = lon2 - lon1
    x = math.sin(dlon) * math.cos(lat2)
    y = math.cos(lat1)*math.sin(lat2) - math.sin(lat1)*math.cos(lat2)*math.cos(dlon)
    return (math.degrees(math.atan2(x, y)) + 360) % 360

def distance(lat1, lon1, lat2, lon2):
    R = 6378137
    lat1, lon1, lat2, lon2 = map(deg2rad, [lat1, lon1, lat2, lon2])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = math.sin(dlat/2)**2 + math.cos(lat1)*math.cos(lat2)*math.sin(dlon/2)**2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

def normalize_angle(a): return (a + 180) % 360 - 180

def checksum(cmd): return f"{reduce(lambda x, y: x ^ y, map(ord, cmd[1:])):02X}"

def send(cmd):
    msg = f"{cmd}*{checksum(cmd)}\r\n"
    ser.write(msg.encode())
    time.sleep(0.1)
    return ser.read(ser.in_waiting).decode('utf-8')

def parse_mwv(s):
    parts = s.split(',')
    if parts[0] != '$IIMWV' or parts[5] != 'A': return None, None
    angle, speed = float(parts[1]), float(parts[3])
    return angle, speed * {'N':1, 'K':0.539957, 'M':1.94384}[parts[4]]

def parse_rmc(s):
    parts = s.split(',')
    if parts[0] != '$GPRMC' or parts[2] == '': return None, None, None, None
    lat = float(parts[3][:2]) + float(parts[3][2:]) / 60
    lat *= -1 if parts[4] == 'S' else 1
    lon = float(parts[5][:3]) + float(parts[5][3:]) / 60
    lon *= -1 if parts[6] == 'W' else 1
    speed = float(parts[7]) * 0.514444 if parts[7] else 0.0
    heading = float(parts[8]) if parts[8] else 0.0
    return lat, lon, heading, speed

def load_waypoints(f):
    with open(f) as file:
        return [(
            float(p[1][:2]) + float(p[1][2:]) / 60 * (-1 if p[2] == 'S' else 1),
            float(p[3][:3]) + float(p[3][3:]) / 60 * (-1 if p[4] == 'W' else 1))
            for l in file if (p := l.strip().split(','))[0] == '$MMWPL']

def xte(curr, prev_wp, curr_wp):
    lat_rad = math.radians(prev_wp[0])
    dx = (curr[1] - prev_wp[1]) * 111319.5 * math.cos(lat_rad)
    dy = (curr[0] - prev_wp[0]) * 111319.5
    tx = (curr_wp[1] - prev_wp[1]) * 111319.5 * math.cos(lat_rad)
    ty = (curr_wp[0] - prev_wp[0]) * 111319.5
    return (tx * dy - ty * dx) / math.hypot(tx, ty) if tx or ty else 0.0

# --- Waypoint Loop Handler ---
def waypoint_loop():
    global prev_xte
    waypoints = load_waypoints("waypoints/waypoints.txt")
    index = 1
    integral = 0.0
    
    while index < len(waypoints):
        line = ser.readline().decode('utf-8', errors='ignore')
        if '$GPRMC' in line:
            lat, lon, heading, speed = parse_rmc(line)
            if not all([lat, lon]): continue

            curr = (lat, lon)
            prev_wp = waypoints[index - 1]
            curr_wp = waypoints[index]

            dist = distance(lat, lon, *curr_wp)
            desired = bearing(*curr, *curr_wp)
            heading_error = normalize_angle(desired - heading)
            cross_track = xte(curr, prev_wp, curr_wp)

            derivative = cross_track - prev_xte
            integral += cross_track
            prev_xte = cross_track

            correction = XTE_GAINS['Kp'] * cross_track + XTE_GAINS['Ki'] * integral + XTE_GAINS['Kd'] * derivative

            fwd = max(THRUST['min_fwd'], THRUST['base_fwd'] - abs(correction))
            rz = max(-THRUST['max_rz'], min(THRUST['max_rz'], correction))

            send(f"$XDRIVE,{fwd:.1f},{rz:.1f}")

            # GUI and log updates
            labels['status'].config(text=f"Status: Heading to WP {index}/{len(waypoints)-1}")
            labels['speed'].config(text=f"Speed: {speed:.2f} m/s")
            labels['heading'].config(text=f"Heading: {heading:.2f}°")
            labels['xte'].config(text=f"XTE: {cross_track:.2f} m")
            labels['distance'].config(text=f"Distance: {dist:.2f} m")

            with open(log_file, 'a', newline='') as f:
                csv.writer(f).writerow([
                    datetime.now().isoformat(), lat, lon, speed / 0.514444, heading,
                    dist, heading_error, cross_track, fwd
                ])

            if dist < THRESHOLDS['distance']:
                index += 1

        elif '$IIMWV' in line:
            angle, speed = parse_mwv(line)
            if angle is not None:
                labels['wind'].config(text=f"Wind: {angle:.1f}° at {speed:.2f} kn")

        time.sleep(0.1)

# Start the loop in a thread
threading.Thread(target=waypoint_loop, daemon=True).start()

root.mainloop()