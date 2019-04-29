import numpy as np
import matplotlib
matplotlib.use("qt5agg")
import matplotlib.pyplot as plt

particle_states = np.loadtxt("particle_states.csv", delimiter=",")
particle_means = particle_states.mean(axis=1)
#particle_stds = particle_states.std(axis=1)
true_states = np.loadtxt("true_states.csv")
observations = np.loadtxt("observations.csv")

xs = np.linspace(0, 10, 1001)[:-1]

#plt.subplot(211)
plt.plot(xs, true_states, label="true")
#plt.plot(xs, particle_states, label="particles")
plt.plot(xs, particle_means, label="mean")
plt.plot(xs, observations, label="observations")
#plt.plot(xs, particle_means + particle_stds, label="mean+std. dev.")
#plt.plot(xs, particle_means - particle_stds, label="mean-std. dev.")
plt.legend()
#plt.subplot(212)
#plt.plot(xs, particle_stds, label="std. dev.")
#plt.legend()
plt.show()
