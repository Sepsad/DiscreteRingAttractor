# DiscreteRingAttractor
MATLAB code for Noorman et al. (2024), Maintaining and updating accurate internal representations of continuous variables with a handful of neurons.

All related data can be downloaded from Janelia Figshare (https://doi.org/10.25378/janelia.26169355). The function main.m contains instructions for generating all main and extended data figure panels. This relies on the Circular Statistics Toolbox (https://github.com/circstat/circstat-matlab).

## Python example

The repository also contains a lightweight Python reimplementation of the ring
attractor dynamics. Run `simulate_attractors.py` to simulate drift from many
initial orientations. When dependencies such as `numpy`, `scipy` and
`matplotlib` are available, passing `plot=True` will display a heatmap of the
activity, a scatter plot of initial vs. final orientations and a PCA
visualization of the final states:

```bash
python simulate_attractors.py
```

The script saves the final bump orientations to `attractors.npy` and the
corresponding activity vectors to `attractor_activity.npy`. It also computes an
energy landscape over bump orientation and width, written to
`energy_landscape.npy`.
