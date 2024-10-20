XAI-thesis
==============================

This is the code for my thesis, which focuses on Explainable AI (XAI) for enhancing customer churn prediction. The repository is organized following a clear structure to ensure reproducibility and ease of understanding. It includes essential files like the LICENSE for legal information, a Makefile for automating tasks such as data preparation and model training, and a README.md providing an overview of the project. The main application is in app.py, supported by additional scripts such as test_environment.py to ensure the environment is correctly set up. The requirements.txt details the dependencies needed to replicate the results, while tox.ini (although this is for future works, therefore it is not related to the thesis) contains settings for continuous testing. The project setup is facilitated by setup.py, which makes it installable. Configuration files are stored in the configs folder, while trained models and artifacts are kept in model_artifact. Jupyter notebooks for data exploration and experimentation are located in the notebooks folder. The src directory contains the source code (although this is for future works, therefore it is not related to the thesis), and app_images holds images used in the application interface. This structure ensures a logical flow for working with XAI methods to analyze and interpret customer churn predictions. 

Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make requirements` (WIP)
    ├── README.md          <- The top-level README for developers using this project.
    ├── .gitignore         <- Git configuration to ignore specific files and directories.
    ├── app.py             <- The main application script.
    ├── test_environment.py <- Script to test the environment setup.
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment.
    ├── tox.ini            <- tox file with settings for running tox. (WIP)
    ├── setup.py           <- Makes the project pip installable (pip install -e .).
    ├── configs            <- Configuration files for the project.
    ├── model_artifact     <- Trained models and related artifacts.
    ├── notebooks          <- Jupyter notebooks for data exploration, model development and XAI.
    ├── src                <- Source code for the project. (Replace with notebooks)(WIP)
    ├── app_images         <- Images produced by the app.

Install packages
------------
To install the required dependencies, run the following command:
```bash
# Recommended if the OS don't have make command
python -m pip install -r requirements.txt
```
If your operating system has make command:
```bash
make requirements
```
Those are the same, but using makefile make it simpler to execute more advanced workflow

Application
------------
In order to run the application, make sure that all the packages have been installed. Then you can run the following command for running the app: 
```bash
python app.py
```
After that, follow the link printed on the terminal to open the application. 
User first select which class they care about. After that the app will sampling from the dataframe with that class and plot the force plot for that instance. The model we are using is Random Forest and the model is trained on the oversampling version of the training data from Telco Telecom Dataset. 

Report
------------
The document can be found in this [link](https://drive.google.com/file/d/1CLefK-n53o_p8V3Tc8RlaYgVrf3XjF-5/view?usp=sharing).

Future works
------------
Future works will focus on upgrade the application. 

Contacts
------------
tel: (+84) 898 539 806

mail: tommyquanglowkey2011@gmail.com
