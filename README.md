# Project_AIC

Artificial Intelligence algorithm for automatic detection of bioprosthesis valve calcification and prediction of structural bioprosthesis valve degeneration

This project use State-of-the-Art Deep learning method to detect calcification, estimate the Agatston score associated and give prediction on structural valve degeneration.

This score is a score based on the extent of coronary artery calcification detected by an unenhanced low-dose CT scan, which is routinely performed in patients undergoing cardiac CT. Due to an extensive body of research, it allows for early risk stratification as patients with a high Agatston score (>160) have an increased risk for a major adverse cardiac event (MACE) 2. Although it does not allow for the assessment of soft non-calcified plaques, it has shown a good correlation with contrast-enhanced CT coronary angiography 1. 

## Installation
```
git clone https://github.com/FrancoisMasson1990/Project_AIC.git
cd Project_AIC
pip install -r requirements.txt
pip install -r requirements.txt --no-deps
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


## Method of calculation of Agatston Score
The calculation is based on the weighted density score given to the highest attenuation value (HU) multiplied by the area of the calcification speck.

### Density factor
130-199 HU: 1
200-299 HU: 2
300-399 HU: 3
400+ HU: 4

For example, if a calcified speck has a maximum attenuation value of 400 HU and occupies 8 sq mm area, then its calcium score will be 32.
The score of every calcified speck is summed up to give the total calcium score.  

Grading of coronary artery disease (based on total calcium score)
no evidence of CAD: 0 calcium score
minimal: 1-10
mild: 11-100
moderate: 101-400
severe: >400
Guidelines for coronary calcium scoring by 2010 ACCF task force
These guidelines are latest at time of writing (July 2016):