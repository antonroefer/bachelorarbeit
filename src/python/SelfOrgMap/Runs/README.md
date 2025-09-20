# Runs

This folder contains the results of different Self-Organizing Map (SOM) training runs.

## Structure

- Hyperparameters can be seen in the git commit messages of the repository.
- Typical contents of each run folder:
  - `trained_som_XX.pkl`: Pickle file with the trained SOM model.
  - `*.png`, `*.jpg`: Visualizations of SOM results, such as U-matrix, hits, planes, and BMU radargrams.
- `*.txt`: Error calculation results for the run.
  - `archived/`: May contain older or backup results.

## Usage

- Use the trained SOM models and visualizations for analysis and presentation.
- Refer to the corresponding run folder for specific results and figures.

## Reference to Thesis

The models referenced in the thesis are mapped to specific runs in this project as follows:

- **Model 1** → **Run 09**
  - *Note: Run 05 and Run 07 are also Model 1, trained for 10 and 30 epochs respectively.*
- **Model 2** → **Run 08**
- **Model 3** → **Run 13**
- **Model 4** → **Run 50**



## Notes

- The `.gitignore` file ensures that only relevant results are tracked.
- For more information on how these runs are generated, see the main `SelfOrgMap` folder and its README.
