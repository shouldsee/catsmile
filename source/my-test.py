import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(0, 10, 200)
y = np.sin(x)
fig, ax = plt.subplots()
print(123455)
ax.plot(x, y, 'b-', linewidth=2)
