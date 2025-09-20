# SelfOrgMap

This folder contains code and data for training and analyzing Self-Organizing Maps (SOMs) on feature vectors derived from radargram data.

## Contents

- **Python Scripts**
    - `minisom.py`: Modified implementation of the MiniSom class for custom SOM training and analysis.
	- `minisom_run.py`: Script to run SOM training and analysis.
	- `model_load.py`: Utilities for loading and visualizing trained SOM models.

- **Feature Vectors**
	- `feature_vectors*.npz`: Numpy arrays containing feature vectors for SOM training.

- **Trained Models**
	- `Runs/RXX/trained_som_XX.pkl`: Pickle files with trained SOM models for different runs.

- **Visualizations**
	- `Runs/RXX/*.png`, `*.jpg`: Plots of SOM results, including U-matrix, hits, planes, and BMU radargrams.

- **Documentation**
	- `README.md`: This file.
	- `minisom-2.3.5.dist-info/`: Metadata and license for the `minisom` package.

## References

- See `minisom-2.3.5.dist-info/licenses/LICENSE` for licensing information.
- For details on radargram data, refer to the parent data folder.
