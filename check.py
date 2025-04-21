from joblib import load

model = load('usv_rf_model.pkl')
print(model.coefs_[0].shape)  # (num_inputs, num_neurons)
