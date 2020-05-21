# PestDetection_using_CNN_DIP

# Source of dataset
https://www.kaggle.com/c/plant-pathology-2020-fgvc7/data
[Have used only Train data]

### Images were of large size of 2048 × 1365. Reduced the size to 180 x 120 which would be required to run the model on a lower end machine such as Pi in an IoT project.

# Approaches:
### 1. CNN based (Both feature extraction and Dense layer for classification)
### 2. DIP (DWT+Segments+LBP) for feature extraction and SVC for classification

### This is a base code structure written for validating both approaches and can be further improvised by working on other techniques.

### Ideally, one should refer to the Notebook files (.ipynb files) which will walk through the steps. The individual .py files are also made for easier usage.
