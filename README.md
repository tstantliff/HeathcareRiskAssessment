# HeathcareRiskAssessment
This is a predictive model for healthcare risk assessment, medical gap closures, and propensity to marketing efforts to close medical gaps. I evaluate neural network's and ensemble learning methods to determine the highest effectiveness and explainability combination. 
Please see attached report for summary.

## How to run my Code

### Structure
- I have two folders of interest: `dataset_1`, `dataset_2`
- Each dataset folder contains the following:
    - A singular CSV file of the dataset (`dataset_{num}\ data\ {data}.csv`)
    - A `config.yaml` file, which houses the optimal hyperparameters for all three models
    - A `models` folder that contains a nested folder for the model being implemented
   - These preprocessing files are `.py` files necessary to replicate my results.
- The `.gitignore` file is the standard recommended file; I made no changes and take no credit for its structure.
- The `requirements.txt` file contains all required libraries to successfully run my code.
- The `Jupyter notebook` folder houses the code I used to fine-tune the model and assess its effectiveness. This folder is not required to run the code, merely a record of my processes.

### Running the Code
- To run the code, simply call each Python model. I opted for individual runs because each model outputs useful metrics on its effectiveness. If you'd like to see a full rundown of my methodology, fine-tuning, and evaluation, my report, or the Jupyter notebook folder.
  
  Example of running my Neural Network Model for Dataset_2 (replace the pathway with your own Python environment and file path):

```bash
>> & C:/Users/ml_en/anaconda3/envs/CudaDNN/python.exe c:/Assignment-1/dataset_2/models/nn_model.py
```
*you can run the models directly from the model as well because I dynamically set the project path*
