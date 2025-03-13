import torch

# Generate random floats between 0 and 1 for 3 semantic classes with 50 tokens each
random_tensor = torch.rand(3, 50)

# Scale to the range [1, 49404]
random_tensor = 1 + random_tensor * (49404 - 1)

# Round to the nearest integer but keep as float
random_tensor = torch.round(random_tensor)
