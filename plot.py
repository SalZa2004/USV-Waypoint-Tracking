import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Load CSV
df = pd.read_csv('log_20250402_123640.csv', parse_dates=['Time'])

# Convert Time to seconds since the start
df['Time_seconds'] = (df['Time'] - df['Time'].iloc[0]).dt.total_seconds()

# Convert speed from knots to m/s (optional)
df['Speed_mps'] = df['Speed (knots)'] * 0.514444

# === Existing plots ===

# XTE over time
plt.figure(figsize=(10,5))
sns.lineplot(x='Time_seconds', y='XTE (m)', data=df)
plt.title('Cross-Track Error over Time')
plt.xlabel('Time (s)')
plt.ylabel('XTE (m)')
plt.grid(True)
plt.show()

# Heading Error over time
plt.figure(figsize=(10,5))
sns.lineplot(x='Time_seconds', y='Heading Error (deg)', data=df)
plt.title('Heading Error over Time')
plt.xlabel('Time (s)')
plt.ylabel('Heading Error (deg)')
plt.grid(True)
plt.show()

# Speed over time
plt.figure(figsize=(10,5))
sns.lineplot(x='Time_seconds', y='Speed (knots)', data=df)
plt.title('Speed over Time')
plt.xlabel('Time (s)')
plt.ylabel('Speed (knots)')
plt.grid(True)
plt.show()

# Histogram of XTE
plt.figure(figsize=(8,5))
sns.histplot(df['XTE (m)'], bins=30, kde=True)
plt.title('Histogram of Cross-Track Error')
plt.xlabel('XTE (m)')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()

# Distance to WP over time
plt.figure(figsize=(10,5))
sns.lineplot(x='Time_seconds', y='Distance to WP (m)', data=df)
plt.title('Distance to Waypoint over Time')
plt.xlabel('Time (s)')
plt.ylabel('Distance to WP (m)')
plt.grid(True)
plt.show()

# Thrust over time
plt.figure(figsize=(10,5))
sns.lineplot(x='Time_seconds', y='Thrust', data=df)
plt.title('Thrust Command over Time')
plt.xlabel('Time (s)')
plt.ylabel('Thrust')
plt.grid(True)
plt.show()

# === NEW plots ===

# 1. Path Efficiency (Lat vs Lon plot)
plt.figure(figsize=(8,8))
sns.lineplot(x='Lon', y='Lat', data=df, marker='o', label='Actual Path')
plt.title('Actual USV Path (Lat vs Lon)')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.grid(True)
plt.legend()
plt.show()

# 2. Thrust Smoothness (Thrust rate of change)
df['Thrust_rate'] = df['Thrust'].diff() / df['Time_seconds'].diff()
plt.figure(figsize=(10,5))
sns.lineplot(x='Time_seconds', y='Thrust_rate', data=df)
plt.title('Thrust Rate of Change over Time')
plt.xlabel('Time (s)')
plt.ylabel('dThrust/dt')
plt.grid(True)
plt.show()

# 3. Heading Error and XTE on same plot
plt.figure(figsize=(10,5))
sns.lineplot(x='Time_seconds', y='Heading Error (deg)', data=df, label='Heading Error (deg)')
sns.lineplot(x='Time_seconds', y='XTE (m)', data=df, label='XTE (m)')
plt.title('Heading Error and XTE over Time')
plt.xlabel('Time (s)')
plt.legend()
plt.grid(True)
plt.show()

# 4. Speed vs Distance to WP
plt.figure(figsize=(10,5))
sns.scatterplot(x='Distance to WP (m)', y='Speed_mps', data=df)
plt.title('Speed vs Distance to Waypoint')
plt.xlabel('Distance to WP (m)')
plt.ylabel('Speed (m/s)')
plt.grid(True)
plt.show()

# Optional: print some basic stats
print("Mean XTE:", df['XTE (m)'].mean())
print("Max XTE:", df['XTE (m)'].max())
print("Mean Heading Error:", df['Heading Error (deg)'].mean())
print("Total Track Time:", df['Time_seconds'].iloc[-1], "seconds")
