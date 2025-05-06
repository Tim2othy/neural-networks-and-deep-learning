# Here I wanted to take the network2 and modify it
# myself to implement covolution layers. Not sure if I'll still bother.
import numpy as np
import torch
from mnist_loader import load_data_wrapper
from network2 import CrossEntropyCost, Network

training_data, validation_data, test_data = load_data_wrapper()


# Function to convert data format
def convert_to_torch(data_list):
    # Extract inputs and targets from the list of tuples
    inputs = [x[0] for x in data_list]
    targets = [x[1] for x in data_list]

    # Stack the data into numpy arrays
    inputs_array = np.stack([x.reshape(-1) for x in inputs])  # Flatten if needed

    # Handle targets differently based on their format
    if isinstance(targets[0], np.ndarray):
        # One-hot encoded targets (training data)
        targets_array = np.stack([y.reshape(-1) for y in targets])
    else:
        # Scalar targets (validation/test data)
        targets_array = np.array(targets)

    # Convert to PyTorch tensors
    inputs_tensor = torch.tensor(inputs_array, dtype=torch.float32)
    targets_tensor = torch.tensor(
        targets_array, dtype=torch.long if targets_array.ndim == 1 else torch.float32
    )

    return inputs_tensor, targets_tensor


# Convert the data to PyTorch format
training_inputs, training_targets = convert_to_torch(training_data)
validation_inputs, validation_targets = convert_to_torch(validation_data)
test_inputs, test_targets = convert_to_torch(test_data)

# Print shapes to verify
print(f"Training inputs shape: {training_inputs.shape}")
print(f"Training targets shape: {training_targets.shape}")
print(f"Validation inputs shape: {validation_inputs.shape}")
print(f"Validation targets shape: {validation_targets.shape}")

# Option 1: If your Network class expects the original format, you can recreate it
# This is the case if you're still using the original network2.py
training_data_torch = list(
    zip(
        training_inputs.reshape(-1, 784, 1).numpy(),
        training_targets.reshape(-1, 10, 1).numpy(),
    )
)
validation_data_torch = list(
    zip(validation_inputs.reshape(-1, 784, 1).numpy(), validation_targets.numpy())
)
test_data_torch = list(
    zip(test_inputs.reshape(-1, 784, 1).numpy(), test_targets.numpy())
)

net = Network([784, 32, 10], cost=CrossEntropyCost)

net.large_weight_initializer()

net.SGD(
    training_data_torch,
    7,
    8,
    0.5,
    evaluation_data=validation_data_torch,
    lmbda=0.1,
    monitor_evaluation_cost=True,
    monitor_evaluation_accuracy=True,
    monitor_training_cost=True,
    monitor_training_accuracy=True,
)
