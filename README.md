# DVC-CNN-TF-pipeline-demo
DVC project for DL usecase using tensorflow

## Project structure -

![](https://github.com/c17hawke/DVC-CNN-TF-pipeline-demo/blob/main/docs/images/DVC-CNN-pipeline@2x%20(1).png?raw=true)

## STEPS -

### STEP 01- Create a repository by using template repository

### STEP 02- Clone the new repository

### STEP 03- Create a conda environment after opening the repository in VSCODE

```bash
conda create --prefix ./env python=3.7 -y
```

```bash
conda activate ./env
```
OR
```bash
source activate ./env
```

### STEP 04- install the requirements
```bash
pip install -r requirements.txt
```

### STEP 05- initialize the dvc project
```bash
dvc init
```

### STEP 06- commit and push the changes to the remote repository