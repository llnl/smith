# import numpy as np
# import matplotlib.pyplot as plt

# # Load the single-column data
# data = np.loadtxt("reaction_log.csv", delimiter=",")

# # Create an index for the x-axis
# x = np.arange(len(data))

# # Plot the data
# plt.figure(figsize=(10, 5))
# plt.plot(x, data, label='Reaction Data', color='blue')
# plt.xlabel('Index')
# plt.ylabel('Value')
# plt.title('Reaction Log Data')
# plt.grid(True)
# plt.legend()
# plt.tight_layout()
# plt.show()

import numpy as np
import matplotlib.pyplot as plt

# Load the two-column data
data = np.loadtxt("reaction_log.csv", delimiter=",")

# Split into x and y components
x = data[:, 0]
y = -data[:, 1]

# Plot the data
plt.figure(figsize=(10, 5))
plt.plot(x, y, label='Reaction Data', color='green')
plt.xlabel('Time')
plt.ylabel('Reaction Force')
plt.title('Reaction Log Data')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
