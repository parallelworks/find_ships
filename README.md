# Find Ships
Use the a convolutional deep neural network model indentify ships in satellite images. This model can be trained with the [TRAIN_SHIP_FINDER](https://demo.parallel.works/u/workflows/edit/623b83c17a96683134eed382) workflow.

Each scene is processed in parallel using slurm as specified in the slurm section of the input form.


### Resources:
Modify the `executors.json` file in the workflow's directory to select the slurm cluster.

### Requirements:
- Singularity container build from the `singularity.file` in the workflow directory with the command `sudo singularity build tensorflow_latest-gpu-jupyter-extra.sif singularity.file`
- The following conda environment in the local and remote environments: `conda create --name parsl_py39 python=3.9; yes | pip install`. The workflow tries to install it if it is not found.
<br>
<div style="text-align:left;"><img src="https://www.dropbox.com/s/wk8uno5m87mle6r/diu_inference.png?raw=1" height="300"></div>
<br>
<div style="text-align:left;"><img src="https://www.dropbox.com/s/s6mgr79cuw0rw0d/diu_workflow_diagram.png?raw=1" height="400"></div>