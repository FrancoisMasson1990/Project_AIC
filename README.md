# Project_AIC

Artificial Intelligence algorithm for automatic detection of bioprosthesis valve calcification and prediction of structural bioprosthesis valve degeneration

This project use State-of-the-Art Deep learning method to detect calcification, estimate the Agatston score associated and give prediction on structural valve degeneration.

This score is a score based on the extent of coronary artery calcification detected by an unenhanced low-dose CT scan, which is routinely performed in patients undergoing cardiac CT. Due to an extensive body of research, it allows for early risk stratification as patients with a high Agatston score (>160) have an increased risk for a major adverse cardiac event (MACE) 2. Although it does not allow for the assessment of soft non-calcified plaques, it has shown a good correlation with contrast-enhanced CT coronary angiography 1. 

## Installation
```
git clone https://github.com/FrancoisMasson1990/Project_AIC.git
cd Project_AIC
pip install aic
```

## Docker build

The project can also be used directly in a docker container.
```
./docker_build.sh
./docker_run.sh (Open the Web App via localhost)
```
or if you want to open the container in an interactive mode (shell interface) and overwrite an entrypoint you can run: 
```
./docker_interactive.sh
```

## Project structure

### Aic

The python source code is all in the `aic` subfolder.

### Configs

Configuration file required to run the API.

### Docs

Contain supported articles used to develop the software and algorithm.

### Img

Contain snapshot of the developed code.

### Scripts

Contains several scripts such as
- Perform 2D AI model classification
- Perform 3D AI model classification
- 3D Render Visualization
- Endpoint Web API

![alt gui](img/gui_render.jpg "Gui Render")
