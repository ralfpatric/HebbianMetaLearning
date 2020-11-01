## Create the plots based off of the returned results
import matplotlib.pyplot as plt
import numpy as np
fitnessFile = './heb_coeffs/1603994435/Fitness_values_1603994435_BipedalWalker-v3.npy';

fitnessFileStatic = 'C:/Users/rabl/Coding Projects/mai/ES/weights/1604236633/Fitness_values_1604236633_BipedalWalker-v3.npy';

fitnessValues = np.load(fitnessFileStatic)

print(fitnessValues)

plt.plot(fitnessValues)
plt.ylabel('Fitness Values')
plt.xlabel('Iterations')
plt.show();
