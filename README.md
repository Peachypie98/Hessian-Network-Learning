# Hessian Network Optimization (PyTorch)
## Abstract
Hessian matrix is a square matrix of second-order partial derivative of a scalar-valued function. In the context of neural networks and deep learning, this function typically represents the loss or object function, which the training process aims to minimize. The Hessian Matrix, thus describes the local curvature of the loss function landscape.

When the parameters of a model are organized in a matrix form of *m* x *n*, we vectorize this matrix to convert it into a vector with *mn* elements. For a function *f* that has this vector as input, the Hessian matrix of *f* is the *mn* x *mn* matrix of second partial derivatives. Each element of this Hessian matrix corresponds to the second derivative with respect to two of the *mn* parameters
<p align="center">
  <img width="431" height="295" src="images/hessian_matrix.png">
</p>

## Merits of Using Hessian in Network Learning
* The Hessian matrix provides critical insights into the geometry of the loss surface, thereby informing us about the curvature of the graph. It enables the optimizer to adjust its steps based on this curvature: in regions where the curvature is steep, the optimizer takes smaller steps to prevent overshooting the minimum; conversely, in flatter regions, it can afford to take larger steps.
* Neural network training landscapes often contain numerous saddle points, characterized by zero gradients yet not constituting minima. The Hessian matrix can effectively distinguish these saddle points from true minima, as it shows neither definitively positive nor negative definiteness at these points.
<p align="center">
  <img width="423" height="294" src="images/landscape_curvature.png">
</p>

## Example Codes 
"In these example codes, we configured a dummy model parameter and a target parameter, each with a batch size and feature dimension of 5. The parameters are updated across 40 iterations using a learning rate of 0.1. The criterion applied in this experiment is the Mean Squared Error(MSE) loss."

### Conventional Gradient Descent (1st Order Partial Derivative)
```py
# Create a dummy model parameter and target parameter with batch and feature of 5
params = torch.rand(5,5, requires_grad=True)
target = torch.rand(5,5)
print("Initial Parameters:\n", params)
print("Target Parameters:\n", target, "\n")

# Train configurations
epoch = 40
lr = 0.1
criterion = nn.MSELoss(reduction='mean')

# Compute first order gradient (a' = a - lr*grad)
loss_1 = []
print("Start Training!")
for i in range(epoch):
    loss = criterion(params, target)
    loss_1.append(loss.item())

    params.retain_grad()
    loss.backward(create_graph=True)
    
    params = params - (lr*params.grad)
    
    if i % 10 == 0:
        print(f"Epoch {i} | Loss: {loss.item():.4f}")
```

### Hessian Gradient Descent (2nd Order Partial Derivative)
```py
# Create a dummy model parameters and target parameters
params = torch.rand(5,5, requires_grad=True)
target = torch.rand(5,5)
print("Initial Parameters:\n", params)
print("Target Parameters:\n", target, "\n")

# Train configurations
epoch = 40
lr = 0.1
criterion = nn.MSELoss(reduction='mean')

# Compute second order gradient (a' = a - lr*inv(H)*grad)
loss_2 = []
hessian_matrix = torch.zeros(5*5, 5*5, dtype=torch.float32)
print("Start Training!")
for i in range(epoch):
    loss = criterion(params, target)
    loss_2.append(loss.item())
    params.retain_grad()
    loss.backward(create_graph=True)
    grad = params.grad.flatten()

    # Compute Hessian for each parameter
    for j in range(len(grad)): 
        grad_2nd = torch.autograd.grad(grad[j], params, create_graph=True)[0]
        hessian_matrix[j] = grad_2nd.flatten()
        
    params = params.view(-1) - lr*(torch.inverse(hessian_matrix) @ grad)
    params = params.view(5,5)
    
    if i % 10 == 0:
        print(f"Epoch {i} | Loss: {loss.item():.4f}")
```

### Hessian Gradient Descent PyTorch (2nd Order Partial Derivative)
```py
# Create a dummy model parameters and target parameters
params = torch.rand(5,5, requires_grad=True)
target = torch.rand(5,5)
print("Initial Parameters:\n", params)
print("Target Parameters:\n", target, "\n")

# Train configurations
epoch = 40
lr = 0.1
criterion = nn.MSELoss(reduction='mean')

# Compute second order gradient (a' = a - lr*inv(H)*grad)
def compute_loss(params, target):
    loss = criterion(params, target)
    return loss

loss_3 = []
print("Start Training!")
for i in range(epoch):
    loss = criterion(params, target)
    params.retain_grad()
    loss.backward(create_graph=True)
    loss_3.append(loss.item())
    grad = params.grad.flatten()
    
    hessian_matrix = AF.hessian(compute_loss, (params, target))[0][0].view(25,25)
    params = params.view(-1) - lr*(torch.inverse(hessian_matrix) @ grad)
    params = params.view(5,5)
    
    if i % 10 == 0:
        print(f"Epoch {i} | Loss: {loss.item():.4f}")
```

## Graph Comparison & Relative Percentage Improvement
<p align="center">
  <img width="576" height="432" src="images/graph_result.png">
</p>

| Method | Initial Loss <br> (Epoch 0) | Final Loss <br> (Epoch 40) | Relative Percentage Improvement |
| ----- | --------- | --------- | --------- |
| 1st Order Partial Derivative (Manual) | 0.1446 | 0.0893 | 38.2% |
| 2nd Order Partial Derivative (Manual / PyTorch) | 0.1446 | 0.0003 | 99.8% |
