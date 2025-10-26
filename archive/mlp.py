import time
import serial
import time
from datetime import datetime
import math
import threading
import tkinter as tk
from tkinter import ttk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import csv
import os
import joblib
import numpy as np
from math import atan2, radians, degrees

# -----------------------------
# Serial Configuration
# -----------------------------
serial_port = 'COM5'
baud_rate = 115200
ser = serial.Serial(serial_port, baud_rate, timeout=1)

# Load model and scaler
model = joblib.load("ML_Models/rf_trained.pkl")
scaler = joblib.load("ML_Models/rf_scaler.pkl")
distance_threshold = 10.0 
log_filename = f"ml_data/log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"

with open(log_filename, mode='w', newline='') as logfile:
    log_writer = csv.writer(logfile)

    # Write header for every new file
    log_writer.writerow(["Time", "Lat", "Lon", "Speed (knots)", "Heading (deg)", "Distance to WP (m)", "Heading Error (deg)", "XTE (m)", "RZ Thrust","Forward Thrust", "WP Lat", "WP Lon",'Bearing to WP'])

def compute_features(current_lat, current_lon, current_heading, current_speed, wp_lat, wp_lon, wind_speed, wind_angle, signed_xte):
    bearing_to_wp = calculate_bearing(current_lat, current_lon, wp_lat, wp_lon)
    distance = calculate_distance(current_lat, current_lon, wp_lat, wp_lon)
    heading_error = normalize_angle(bearing_to_wp - current_heading)
    signed_xte = calculate_signed_xte((current_lat, current_lon), (current_lat, current_lon), (wp_lat, wp_lon))
    delta_lat = wp_lat - current_lat
    delta_lon = wp_lon - current_lon
    if wind_angle is None:
        wind_angle = 0.0  # or log and return early
    relative_wind = normalize_angle(wind_angle - current_heading)
    relative_wind = normalize_angle(wind_angle - current_heading)
    return np.array([heading_error, signed_xte, current_speed, distance,  relative_wind, wind_speed])


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
wind_label = ttk.Label(root, text="Wind: ---° @ --.- knots", font=("Arial", 15))
wind_label.pack()

distance_label = ttk.Label(root, text="Distance: --.- m", font=("Arial", 15))
distance_label.pack()

path = []
start_pos = None  # Will be set to the first valid RMC reading

# -----------------------------
# Utility Functions
# -----------------------------
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

def parse_mwv(sentence):
    if sentence.startswith('$IIMWV'):
        try:
            parts = sentence.split(',')
            wind_angle = float(parts[1])
            reference = parts[2]  # R = relative, T = true
            wind_speed = float(parts[3])
            wind_unit = parts[4]  # 'N' = knots
            return wind_angle, wind_speed
        except (ValueError, IndexError):
            return None, None
    return None, None

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



def update_gui():
    global path, start_pos, prev_xte
    prev_xte = 0.0  # Initialize previous cross-track error
    try:
        # Wait for the first valid RMC to set start_pos
        while start_pos is None:
            send_command('$CCNVO,2,1.0,0,0.0')
            rmc_sentence = ser.readline().decode().strip()
            current_lat, current_lon, heading, speed = parse_rmc(rmc_sentence)
            if current_lat is not None:
                start_pos = (current_lat, current_lon)
                print("Start position acquired:", start_pos)

        waypoints = parse_waypoints('waypoints/zigzag.txt')
        send_command('$CCAPM,0,64,0,80')
        send_command('$CCTHD,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00')
        time.sleep(1)
        previous_wp = start_pos  # For the first leg

        prev_time = time.time()


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
                mwv_sentence = ser.readline().decode().strip()
                wind_angle, wind_speed = parse_mwv(mwv_sentence)
                if wind_angle is not None and wind_speed is not None:
                    wind_label.config(text=f"Wind: {wind_angle:.1f}° @ {wind_speed:.1f} kts")


                signed_xte = calculate_signed_xte((current_lat, current_lon), previous_wp, current_wp)
                X = compute_features(current_lat, current_lon, current_heading, current_speed, waypoint_lat, waypoint_lon, wind_speed, wind_angle, signed_xte)
                X = [0.0 if (x is None or np.isnan(x)) else x for x in X]
                X_scaled = scaler.transform([X])
                bearing = calculate_bearing(current_lat, current_lon, waypoint_lat, waypoint_lon)
                distance = calculate_distance(current_lat, current_lon, waypoint_lat, waypoint_lon)
                heading_error = normalize_angle(bearing - current_heading)
                rz_thrust, forward_thrust = model.predict(X_scaled)[0]
                rz_thrust = max(min(rz_thrust, 50), -100)
                forward_thrust = max(min(forward_thrust, 100), 0)
                thrust_command = f'$CCTHD,{forward_thrust:.2f},0.00,0.00,0.00,0.00,{rz_thrust:.2f},0.00,0.00'
                send_command(thrust_command)

                speed_label.config(text=f"Speed: {current_speed*1.94384:.1f} kts")
                heading_label.config(text=f"Heading: {current_heading:.1f}°")
                distance_label.config(text=f"Distance: {distance:.1f} m")
                xte_label.config(text=f"XTE: {abs(signed_xte):.1f} m")

                with open(log_filename, 'a', newline='') as f:  # 'a' for append
                    log_writer = csv.writer(f)
                    log_writer.writerow([
                        time.strftime("%Y-%m-%d %H:%M:%S"),
                        current_lat,
                        current_lon,
                        current_speed*1.94384,
                        current_heading,
                        distance,
                        heading_error,
                        signed_xte,
                        rz_thrust,
                        forward_thrust,
                        waypoint_lat,
                        waypoint_lon,
                        bearing
                    ])  

                              
                # --- Visualization ---
                ax.clear()
                ax.set_title("Navigation Visualisation")
                ax.set_facecolor('xkcd:powder blue')
                ax.grid(True)
                ax.set_xlim(min(wp[1] for wp in waypoints) - 0.001, max(wp[1] for wp in waypoints) + 0.001)
                ax.set_ylim(min(wp[0] for wp in waypoints) - 0.0001, max(wp[0] for wp in waypoints) + 0.0001)

                # Plot start -> first waypoint line
                if previous_wp == start_pos:
                    first_wp = waypoints[0]
                    ax.plot([start_pos[1], first_wp[1]], [start_pos[0], first_wp[0]], 'g--', label='Start -> First WP')

                # Plot planned route and waypoints
                waypoints_lons = [wp[1] for wp in waypoints]
                waypoints_lats = [wp[0] for wp in waypoints]
                ax.plot(waypoints_lons, waypoints_lats, 'k--', linewidth=1, label='Planned Route')
                for i, (latW, lonW) in enumerate(waypoints, start=1):
                    color = 'ro' if (latW, lonW) == (waypoint_lat, waypoint_lon) else 'bo'
                    ax.plot(lonW, latW, color, markersize=5)
                    ax.text(lonW, latW, f"W{i}", fontsize=8, verticalalignment='bottom')

                # Plot current position and heading arrow
                ax.plot(current_lon, current_lat, 'go', markersize=10, label="Current Position")
                arrow_length = 0.0001
                # Convert current heading: if 0° means North, math angle = 90 - current_heading
                plot_heading = 90 - current_heading
                dx = arrow_length * math.cos(math.radians(plot_heading))
                dy = arrow_length * math.sin(math.radians(plot_heading))
                ax.annotate("", xy=(current_lon + dx, current_lat + dy), xytext=(current_lon, current_lat),
                            arrowprops=dict(arrowstyle="->", color='g', lw=2))

                # Plot path taken
                path.append((current_lat, current_lon))
                if len(path) > 1:
                    path_lons = [p[1] for p in path]
                    path_lats = [p[0] for p in path]
                    ax.plot(path_lons, path_lats, 'r-', linewidth=2, label="Path")
                ax.legend()
                canvas.draw()

                if distance < distance_threshold:
                    previous_wp = current_wp
                    time.sleep(2)
                    break
                time.sleep(1)  # 1Hz update rate
        
        send_command('$CCTHD,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00')
        status_label.config(text="All waypoints reached. Mission complete.")
        ser.close()

    except KeyboardInterrupt:
        send_command('$CCTHD,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00')
        status_label.config(text="Navigation Stopped")

if __name__ == '__main__':
# Start GUI Thread
    threading.Thread(target=update_gui, daemon=True).start()
    root.mainloop()
