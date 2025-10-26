import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd
import glob
from sklearn.preprocessing import StandardScaler
import math

# 1. Load data
csv_files = glob.glob("ML_Training/*.csv")
df_list = [pd.read_csv(file) for file in csv_files]
combined_df = pd.concat(df_list, ignore_index=True)
combined_df = combined_df.dropna()

# 2. Feature engineering
combined_df['relative_wind'] = combined_df['Wind Angle (deg)'] - combined_df['Heading (deg)']
features = ['Heading Error (deg)', 'Distance to WP (m)', 'relative_wind', 'Wind Speed (knots)']

X = combined_df[features].values

# 3. Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 4. Train-test split
X_train, X_test = train_test_split(X_scaled, test_size=0.2, random_state=42)

# 5. Convert to Torch
X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)


# 6. Neural Net
class WaypointNet(nn.Module):
    def __init__(self):
        super(WaypointNet, self).__init__()
        self.fc1 = nn.Linear(4, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 2)  # Output RZ Thrust and Forward Thrust
    
    def forward(self, x):
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        x = self.fc3(x)
        return x

model = WaypointNet()

# 7. Optimizer
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# 8. Simple physics model
def simple_motion_model(state, thrusts, dt=0.1):
    heading_error, distance_to_wp, relative_wind, wind_speed = state
    rz_thrust, forward_thrust = thrusts

    # Parameters
    max_turn_rate = 30.0  # degrees per second for rz=1 thrust
    max_forward_speed = 5.0  # meters per second for fwd=1 thrust

    # Apply thrust to simulate motion
    heading_change = rz_thrust * max_turn_rate * dt
    distance_change = forward_thrust * max_forward_speed * dt

    # Update heading error
    new_heading_error = heading_error - heading_change
    new_heading_error = (new_heading_error + 180) % 360 - 180  # Normalize to [-180, 180]

    # Update distance
    new_distance_to_wp = max(distance_to_wp - distance_change * math.cos(math.radians(new_heading_error)), 0)

    return new_heading_error, new_distance_to_wp

# 9. Custom Training Loop
# 9. Custom Training Loop
losses = []
for epoch in range(100):  # You can adjust epochs
    model.train()
    epoch_loss = 0
    
    for i in range(X_train.shape[0]):
        state = X_train[i]
        optimizer.zero_grad()
        
        thrusts = model(state)
        
        rz_thrust, fwd_thrust = thrusts[0], thrusts[1]
        
        # Simulate next state
        new_heading_error, new_distance_error = simple_motion_model(state, thrusts)
        
        # Reward:
        #   - Minimize heading error
        #   - Minimize distance error
        #   - Encourage high forward thrust
        loss_heading = (new_heading_error)**2
        loss_distance = (new_distance_error)**2
        loss_speed_bonus = 0.01 * (100 - fwd_thrust)**2  # Encourage fwd_thrust to be close to 1
        
        loss = loss_heading + loss_distance + loss_speed_bonus

        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
    
    avg_loss = epoch_loss / X_train.shape[0]
    losses.append(avg_loss)
    
    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {avg_loss:.4f}")


# 10. Save model + scaler if you want
torch.save(model.state_dict(), "ML_Models/waypoint_net_rewardloss.pth")

import matplotlib.pyplot as plt

plt.plot(losses)
plt.title("Training Loss Over Epochs (Reward Loss)")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid()
plt.show()
