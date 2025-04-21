import serial
import time
import math
import threading
import tkinter as tk
from tkinter import ttk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import csv
import os
from datetime import datetime
import numpy as np




distance_threshold = 10.0  # meters
# -----------------------------
# Serial Configuration
# -----------------------------
serial_port = 'COM5'
baud_rate = 115200
ser = serial.Serial(serial_port, baud_rate, timeout=1)

# -----------------------------
# Logging Configuration
# -----------------------------
# Create a unique filename based on the current date and time
log_filename = f"wind_data/log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"

with open(log_filename, mode='w', newline='') as logfile:
    log_writer = csv.writer(logfile)

    # Write header for every new file
    log_writer.writerow(["Time", "Lat", "Lon", "Speed (knots)", "Heading (deg)", "Distance to WP (m)", "Heading Error (deg)", "XTE (m)", "RZ Thrust","Forward Thrust", "WP Lat", "WP Lon",'Bearing to WP', 'Wind Speed (knots)', 'Wind Angle (deg)'])



# -----------------------------
# GUI Setup
# -----------------------------
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
    if not sentence.startswith('$IIMWV'):
        return None, None
    try:
        parts = sentence.split(',')
        wind_angle = float(parts[1])
        wind_speed = float(parts[3])
        wind_unit = parts[4]
        if wind_unit != 'N':
            return None, None
        return wind_angle, wind_speed
    except (ValueError, IndexError):
        return None, None


def parse_rmc(sentence):
    if not sentence.startswith('$GPRMC'):
        return None, None, None, None
    try:
        parts = sentence.split(',')
        if parts[2] != 'A':
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
    except (ValueError, IndexError):
        return None, None, None, None


def parse_rot(sentence):
    if not sentence.startswith('$SPROT'):
        return None
    try:
        parts = sentence.split(',')
        rot_raw = parts[1]
        status = parts[2] if len(parts) > 2 else 'V'

        if status != 'A':
            return None

        rot = float(rot_raw) / 60
        return rot
    except (ValueError, IndexError):
        return None


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

def solve_mpc(current_lat, current_lon, current_heading, waypoint_lat, waypoint_lon, current_speed, prev_wp, rot):
    best_cost = float('inf')
    best_forward_thrust = 0.0
    best_rz_thrust = 0.0

    dt = 1.0  # timestep (seconds)
    prediction_horizon = 10  # number of steps (10s prediction)

    # Candidate thrusts to search
    forward_thrust_candidates = np.linspace(20, 100, 5)  # 20, 40, 60, 80, 100
    rz_thrust_candidates = np.linspace(-100, 100, 21)     # wider range for sharper turning

    # Desired target speed
    desired_speed_knots = 5 if current_speed > 2.5 else 1
    desired_speed_mps = desired_speed_knots * 0.5144  # knots to m/s

    def thrust_to_speed(thrust):
        return thrust / 100 * 6.0  # assumes max thrust = ~6m/s (tune if needed)

    for forward_thrust in forward_thrust_candidates:
        for rz_thrust in rz_thrust_candidates:
            # Initialize predicted state
            x = current_lon
            y = current_lat
            heading = current_heading
            speed = thrust_to_speed(forward_thrust)

            for _ in range(prediction_horizon):
                meters_to_deg = 1 / 111320  # approx at equator
                # Predict position update
                x += speed * np.cos(np.deg2rad(heading)) * dt * meters_to_deg
                y += speed * np.sin(np.deg2rad(heading)) * dt * meters_to_deg
                # Predict heading update
                if rot is not None:
                    heading += rot * dt  # slightly bigger gain
                else:
                    heading += rz_thrust * dt / 10

            # Final predicted position and heading
            pred_lat = y
            pred_lon = x
            pred_heading = heading

            # Calculate errors
            pred_distance = calculate_distance(pred_lat, pred_lon, waypoint_lat, waypoint_lon)
            pred_bearing = calculate_bearing(pred_lat, pred_lon, waypoint_lat, waypoint_lon)
            predicted_xte = calculate_signed_xte((pred_lat, pred_lon), prev_wp, (waypoint_lat, waypoint_lon))
            pred_heading_error = abs(normalize_angle(pred_bearing - pred_heading))

            # Cost function
            cost = 0
            cost += 50 * abs(predicted_xte)              # cross track error (main priority)
            cost += 20 * pred_distance                   # reaching the waypoint
            cost += 0.1 * abs(rz_thrust)                  # avoid crazy spinning
            cost += 2 * abs(speed - desired_speed_mps)    # encourage desired speed
            cost += 1000 * max(0, abs(predicted_xte) - 3) # huge penalty if far off path
            cost += 1000 * max(0, abs(pred_heading_error) - 5)  # penalize bad heading

            # If very close to waypoint, care more about heading accuracy
            if pred_distance < 3:
                cost += 500 * abs(pred_heading_error)

            # Select best control input
            if cost < best_cost:
                best_cost = cost
                best_forward_thrust = forward_thrust
                best_rz_thrust = rz_thrust

    return best_forward_thrust, best_rz_thrust



# -----------------------------
# Main Waypoint Tracking Loop
# -----------------------------
def update_gui():
    global path, start_pos, prev_xte
    prev_xte = 0.0  # Initialize previous cross-track error
    try:
        # Wait for the first valid RMC to set start_pos
        sentence = ser.readline().decode().strip()
        while start_pos is None:
            send_command('$CCNVO,2,1.0,0,0.0')
            lat, lon, hdg, spd = parse_rmc(sentence)
            if lat is not None:
                start_pos = (lat, lon)
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
                rmc_sentence = ser.readline().decode().strip()
                rot_sentence = ser.readline().decode().strip()
                mwv_sentence = ser.readline().decode().strip()
                current_time = time.time()
                dt = current_time - prev_time
                prev_time = current_time

                send_command('$CCNVO,2,1.0,0,0.0')
                current_lat, current_lon, current_heading, current_speed = parse_rmc(rmc_sentence)
                if current_lat is None:
                    time.sleep(1)
                    continue
                # Try reading MWV wind data
                wind_angle, wind_speed = parse_mwv(mwv_sentence)
                if wind_angle is not None and wind_speed is not None:
                    wind_label.config(text=f"Wind: {wind_angle:.1f}° @ {wind_speed:.1f} kts")


                bearing = calculate_bearing(current_lat, current_lon, waypoint_lat, waypoint_lon)
                distance = calculate_distance(current_lat, current_lon, waypoint_lat, waypoint_lon)
                heading_error = normalize_angle(bearing - current_heading)
                rot = parse_rot(rot_sentence)
                forward_thrust, rz_thrust = solve_mpc(current_lat, current_lon, current_heading, waypoint_lat, waypoint_lon, current_speed, previous_wp, rot)


                signed_xte = calculate_signed_xte((current_lat, current_lon), previous_wp, current_wp)
                thrust_command = f'$CCTHD,{forward_thrust:.2f},0.00,0.00,0.00,0.00,{rz_thrust:.2f},0.00,0.00'
                send_command(thrust_command)

                xte_label.config(text=f"XTE: {abs(signed_xte):.1f} m")
                speed_label.config(text=f"Speed: {current_speed*1.94384:.1f} kts")
                heading_label.config(text=f"Heading: {current_heading:.1f}°")
                distance_label.config(text=f"Distance: {distance:.1f} m")


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
                        current_wp[0],
                        current_wp[1],
                        bearing,
                        wind_speed, 
                        wind_angle
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
                    send_command('$CCTHD,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00')
                    previous_wp = current_wp
                    time.sleep(2)
                    break
                time.sleep(1)  # 1Hz update rate

        status_label.config(text="All waypoints reached. Mission complete.")
        ser.close()

    except KeyboardInterrupt:
        send_command('$CCTHD,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00')
        status_label.config(text="Navigation Stopped")

if __name__ == '__main__':
# Start GUI Thread
    threading.Thread(target=update_gui, daemon=True).start()
    root.mainloop()