import numpy as np

#First, we code two simple networks that compete to achieve a non-equilibrium steady state

# Define the parameters of the system
N = 10  # number of nodes in each network
alpha = 0.1  # strength of interaction between the two networks
gamma1 = 0.1  # damping factor for network 1
gamma2 = 0.2  # damping factor for network 2

# Initialize the networks with random values
net1 = np.random.rand(N)
net2 = np.random.rand(N)


# Define the dynamics of the networks
def update_networks(net1, net2):
    # Update network 1
    net1_new = (1 - gamma1) * net1 + gamma1 * (alpha * net2 + (1 - alpha) * np.random.rand(N))

    # Update network 2
    net2_new = (1 - gamma2) * net2 + gamma2 * ((1 - alpha) * net1 + alpha * np.random.rand(N))

    # Return the updated networks
    return net1_new, net2_new


# Run the dynamics of the networks for a fixed number of steps
for i in range(100):
    net1, net2 = update_networks(net1, net2)

# Print the final values of the networks
print("Final values of network 1: ", net1)
print("Final values of network 2: ", net2)

# In this example, we define two networks with N nodes each and initialize them with random values. We also define
# parameters for the strength of interaction between the networks (alpha) and the damping factors for each network
# (gamma1 and gamma2). We then define a function update_networks that updates each network according to a set of rules
# that includes interactions with the other network and random fluctuations. This function returns the updated networks
# as NumPy arrays. Finally, we run the dynamics of the networks for a fixed number of steps using a loop that calls the
# update_networks function. After the loop, we print the final values of the two networks.


# Next we code a machine learning model that uses these 'dissipative' networks as computational building blocks such
# that, At higher levels of the hierarchy, the network could learn to recognize patterns in the input data by composing
# the lower-level modules and clusters into increasingly complex representations. The network could also be designed to
# have multiple time scales of activity, such that slower dynamics at higher levels of the hierarchy could modulate the
# faster dynamics at lower levels.

# Additionally, the network could be designed to dynamically reconfigure itself in response to changes in the input data
# distribution, such as by forming and dissolving clusters or by adjusting the strength of connections between modules.
# This would allow the network to adapt to changes in the environment and maintain its non-equilibrium steady state.

import numpy as np


class Network:
    def __init__(self, n_inputs, n_outputs, n_modules):
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.n_modules = n_modules

        # Initialize modules with random weights
        self.modules = [Module(n_inputs, n_outputs) for _ in range(n_modules)]

        # Initialize connections between modules with random strengths
        self.connections = np.random.randn(n_modules, n_modules)

    def forward(self, x):
        # Compute output of each module
        module_outputs = [module.forward(x) for module in self.modules]

        # Compute output of each cluster as weighted sum of module outputs
        cluster_outputs = np.dot(self.connections, module_outputs)

        # Compute output of network as weighted sum of cluster outputs
        y = np.sum(cluster_outputs, axis=0)
        return y

    def backward(self, dy):
        # Compute gradients of cluster outputs
        d_cluster_outputs = np.outer(dy, np.ones(self.n_modules))

        # Compute gradients of module outputs as weighted sum of cluster output gradients
        d_module_outputs = np.dot(self.connections.T, d_cluster_outputs)

        # Backpropagate gradients through each module
        for i in range(self.n_modules):
            self.modules[i].backward(d_module_outputs[i])

    def update(self, lr):
        # Update weights of each module
        for module in self.modules:
            module.update(lr)

        # Update strengths of connections between modules
        self.connections += lr * np.outer(np.mean(self.modules_outputs, axis=1), np.mean(self.modules_outputs, axis=1))

    def adapt(self, x):
        # Compute current mean of module outputs
        self.modules_outputs = np.array([module.forward(x) for module in self.modules])
        current_mean = np.mean(self.modules_outputs, axis=1)

        # Compute new strengths of connections between modules based on change in mean module outputs
        delta_mean = current_mean - self.prev_mean
        self.connections += self.adapt_lr * np.outer(delta_mean, delta_mean)

        # Save current mean for next iteration
        self.prev_mean = current_mean

# In this implementation, the adapt method takes an input x and adjusts the strengths of connections between modules
# based on the change in mean module outputs since the last call to adapt. Specifically, it computes the difference
# between the current mean module outputs and the previous mean module outputs, and updates the connections using a
# learning rate adapt_lr times the outer product of the difference vector with itself. The new strengths of connections
# cause the module outputs to shift towards a non-equilibrium steady state that is better suited to the current input
# distribution. The adapt method saves the current mean module outputs for use in the next iteration.