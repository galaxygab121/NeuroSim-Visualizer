import numpy as np
import matplotlib.pyplot as plt

def lif_sim(
    T=0.5, dt=1e-4,
    V_rest=-65.0, V_reset=-70.0, V_thresh=-50.0,
    R=10.0, tau=0.02,
    I_base=1.5, t_stim_start=0.05, t_stim_end=0.45,
    refractory=0.003
):
    n = int(T/dt)
    t = np.linspace(0, T, n, endpoint=False)
    V = np.full(n, V_rest, dtype=float)
    I = np.zeros(n)
    I[(t >= t_stim_start) & (t <= t_stim_end)] = I_base

    spikes = []
    refrac_left = 0.0

    for i in range(1, n):
        if refrac_left > 0.0:
            V[i] = V_reset
            refrac_left -= dt
            continue
        dV = (-(V[i-1] - V_rest) + R * I[i-1]) / tau
        V[i] = V[i-1] + dV * dt
        if V[i] >= V_thresh:
            spikes.append(t[i])
            V[i] = V_reset
            refrac_left = refractory

    return t, V, np.array(spikes), I

def plot_results(t, V, spikes, I, V_thresh):
    fig, ax1 = plt.subplots(figsize=(9, 4.5))
    ax1.plot(t, V, label="Membrane potential (mV)")
    ax1.axhline(V_thresh, linestyle="--", linewidth=1, label="Threshold")
    if spikes.size > 0:
        ax1.scatter(spikes, np.full_like(spikes, V_thresh), marker="x", label="Spikes")
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Voltage (mV)")
    ax1.set_title("Leaky Integrate-and-Fire Neuron")
    ax1.legend(loc="upper right")
    ax2 = ax1.twinx()
    ax2.plot(t, I, alpha=0.4, linewidth=1)
    ax2.set_ylabel("Input current (nA)")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--I", type=float, default=1.5, help="Input current nA")
    p.add_argument("--T", type=float, default=0.5, help="Total time (s)")
    p.add_argument("--dt", type=float, default=1e-4, help="Time step (s)")
    args = p.parse_args()

    t, V, spikes, I = lif_sim(T=args.T, dt=args.dt, I_base=args.I)
    print(f"Spike count: {len(spikes)}")
    plot_results(t, V, spikes, I, V_thresh=-50.0)
