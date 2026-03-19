# Diabetes Prediction App with Gradio

This project is a simple machine learning demo that predicts whether a patient is likely to have diabetes based on medical input values.

The project includes:

- a Jupyter notebook for training a Decision Tree model
- a saved `.pkl` model file
- a Gradio user interface for entering predictor values
- a lightweight FastAPI wrapper so the app can be deployed on Vercel

## Project Structure

- [app.py](/d:/Model/MK%20-%20AI%20and%20IoT/Basic_ML/app.py): Gradio app and prediction logic
- [api/index.py](/d:/Model/MK%20-%20AI%20and%20IoT/Basic_ML/api/index.py): Vercel Python entrypoint
- [diabetes_decision_tree.ipynb](/d:/Model/MK%20-%20AI%20and%20IoT/Basic_ML/diabetes_decision_tree.ipynb): notebook for model training
- [diabetes_decision_tree.pkl](/d:/Model/MK%20-%20AI%20and%20IoT/Basic_ML/diabetes_decision_tree.pkl): trained model file
- [diabetes.csv](/d:/Model/MK%20-%20AI%20and%20IoT/Basic_ML/diabetes.csv): dataset
- [requirements.txt](/d:/Model/MK%20-%20AI%20and%20IoT/Basic_ML/requirements.txt): Python dependencies
- [vercel.json](/d:/Model/MK%20-%20AI%20and%20IoT/Basic_ML/vercel.json): Vercel configuration

## Features Used for Prediction

The model expects these 8 input features:

- `Pregnancies`
- `Glucose`
- `BloodPressure`
- `SkinThickness`
- `Insulin`
- `BMI`
- `DiabetesPedigreeFunction`
- `Age`

## How the App Works

1. The notebook trains a `DecisionTreeClassifier` using `diabetes.csv`.
2. The trained model is saved as `diabetes_decision_tree.pkl`.
3. The Gradio app loads the saved pickle model.
4. A user enters the predictor values in the UI.
5. The app returns a prediction:
   - `Diabetic`
   - `Not Diabetic`

## Run Locally

Create and activate a virtual environment if you want:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

Install dependencies:

```powershell
python -m pip install -r requirements.txt
```

Run the app:

```powershell
python app.py
```

Then open the local URL shown in the terminal, usually:

```text
http://127.0.0.1:7860
```

## Train or Recreate the Model

Open the notebook below and run all cells:

- [diabetes_decision_tree.ipynb](/d:/Model/MK%20-%20AI%20and%20IoT/Basic_ML/diabetes_decision_tree.ipynb)

This will:

- load the dataset
- split training and testing data
- train the decision tree classifier
- evaluate the model
- save the model as a `.pkl` file

## Deploy to Vercel

This project uses Gradio mounted on FastAPI so it can fit Vercel's Python app style.

### Steps

1. Push this project to a GitHub repository.
2. Go to [Vercel](https://vercel.com/).
3. Create a new project.
4. Import your GitHub repository.
5. Set the project root to this folder if needed.
6. Deploy.

### Important Notes

- Keep `diabetes_decision_tree.pkl` in the repository, because the app loads it at runtime.
- `api/index.py` is used as the Python entrypoint for Vercel.
- `vercel.json` provides the Vercel deployment configuration.

## Why FastAPI Is Used Here

This project is still mainly a Gradio app.

FastAPI is only used as a thin wrapper so the Gradio interface can be mounted into a Vercel-friendly Python web app. You do not need to manually write REST endpoints for this project.

## Tech Stack

- Python
- Pandas
- Scikit-learn
- Gradio
- FastAPI
- Vercel

## Learning Purpose

This project is intentionally simple and designed for learning:

- train a basic machine learning model
- save it with pickle
- build a quick user interface
- deploy it with minimal configuration
