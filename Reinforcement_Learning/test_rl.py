
import torch
from model import DQN
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load your trained model
input_dim = 8
output_dim = 2
policy_net = DQN(input_dim, output_dim).to(device)
policy_net.load_state_dict(torch.load("saved_models/dqn_model_950.pth", map_location=device))
policy_net.eval()

def get_action(state):
    with torch.no_grad():
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        action = policy_net(state)
        return action.cpu().numpy()[0]

# Example main loop
while True:
    # TODO: Replace with your real GPS, heading, speed data reading here
    current_lat = 50.0
    current_lon = -1.0
    current_heading = 0
    current_speed = 0
    wind_speed = 2
    wind_angle = 90
    target_lat = 50.001
    target_lon = -1.001

    state = [current_lat, current_lon, current_heading, current_speed, wind_speed, wind_angle, target_lat, target_lon]

    action = get_action(state)
    rz_thrust, forward_thrust = action

    # Send these thrusts to your vehicle's motor controllers
    print(f"rz_thrust: {rz_thrust:.2f}, forward_thrust: {forward_thrust:.2f}")

    time.sleep(0.1)
