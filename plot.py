import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load CSV
df = pd.read_csv('log.csv', parse_dates=['Time'])

# Convert Time to seconds since the start
df['Time_seconds'] = (df['Time'] - df['Time'].iloc[0]).dt.total_seconds()

# Convert speed from knots to m/s (optional)
df['Speed_mps'] = df['Speed (knots)'] * 0.514444

# XTE over time (in seconds)
plt.figure(figsize=(10,5))
sns.lineplot(x='Time_seconds', y='XTE (m)', data=df)
plt.title('Cross-Track Error over Time')
plt.xlabel('Time (s)')
plt.ylabel('XTE (m)')
plt.grid(True)
plt.show()

# Heading Error over time (in seconds)
plt.figure(figsize=(10,5))
sns.lineplot(x='Time_seconds', y='Heading Error (deg)', data=df)
plt.title('Heading Error over Time')
plt.xlabel('Time (s)')
plt.ylabel('Heading Error (deg)')
plt.grid(True)
plt.show()

# Speed over time (in seconds)
plt.figure(figsize=(10,5))
sns.lineplot(x='Time_seconds', y='Speed (knots)', data=df)
plt.title('Speed over Time')
plt.xlabel('Time (s)')
plt.ylabel('Speed (knots)')
plt.grid(True)
plt.show()

# Distance to WP over time (in seconds)
plt.figure(figsize=(10,5))
sns.lineplot(x='Time_seconds', y='Distance to WP (m)', data=df)
plt.title('Distance to Waypoint over Time')
plt.xlabel('Time (s)')
plt.ylabel('Distance to WP (m)')
plt.grid(True)
plt.show()

# Thrust over time (in seconds)
plt.figure(figsize=(10,5))
sns.lineplot(x='Time_seconds', y='Thrust', data=df)
plt.title('Thrust Command over Time')
plt.xlabel('Time (s)')
plt.ylabel('Thrust')
plt.grid(True)
plt.show()
