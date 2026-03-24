from pathlib import Path
import pickle

import gradio as gr
import pandas as pd
from fastapi import FastAPI


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "diabetes_decision_tree.pkl"

FEATURE_COLUMNS = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age",
]


class NumpyCompatUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        # Support model artifacts created across NumPy 1.x <-> 2.x internal path changes.
        if module.startswith("numpy._core"):
            module = module.replace("numpy._core", "numpy.core", 1)
        elif module.startswith("numpy.core"):
            module = module.replace("numpy.core", "numpy._core", 1)
        return super().find_class(module, name)


with MODEL_PATH.open("rb") as model_file:
    model = NumpyCompatUnpickler(model_file).load()


def predict_diabetes(
    pregnancies,
    glucose,
    blood_pressure,
    skin_thickness,
    insulin,
    bmi,
    diabetes_pedigree_function,
    age,
):
    values = [
        pregnancies,
        glucose,
        blood_pressure,
        skin_thickness,
        insulin,
        bmi,
        diabetes_pedigree_function,
        age,
    ]

    input_frame = pd.DataFrame([values], columns=FEATURE_COLUMNS)
    prediction = int(model.predict(input_frame)[0])

    if hasattr(model, "predict_proba"):
        probability = float(model.predict_proba(input_frame)[0][1])
        confidence_text = f"\nProbability of diabetes: {probability:.2%}"
    else:
        confidence_text = ""

    result_text = "Diabetic" if prediction == 1 else "Not Diabetic"
    return f"Prediction: {result_text}{confidence_text}"


with gr.Blocks(title="Diabetes Predictor") as demo:
    gr.Markdown("# Diabetes Prediction App")
    gr.Markdown(
        "Fill in the patient values below and click Predict to run the saved decision tree model."
    )
    gr.Markdown(
        "Prediction output: `0` means not diabetic and `1` means diabetic. "
        "The app shows a friendly text label instead of the raw class number."
    )

    with gr.Row():
        pregnancies = gr.Number(
            label="Pregnancies",
            info="Number of times pregnant.",
            value=1,
            precision=0,
        )
        glucose = gr.Number(
            label="Glucose",
            info="Plasma glucose concentration 2 hours after an oral glucose tolerance test.",
            value=120,
            precision=0,
        )
        blood_pressure = gr.Number(
            label="BloodPressure",
            info="Diastolic blood pressure in mm Hg.",
            value=70,
            precision=0,
        )
        skin_thickness = gr.Number(
            label="SkinThickness",
            info="Triceps skin fold thickness in mm.",
            value=20,
            precision=0,
        )

    with gr.Row():
        insulin = gr.Number(
            label="Insulin",
            info="2-hour serum insulin in mu U/ml.",
            value=79,
            precision=0,
        )
        bmi = gr.Number(
            label="BMI",
            info="Body mass index measured as weight in kg divided by height in meters squared.",
            value=28.5,
        )
        diabetes_pedigree_function = gr.Number(
            label="DiabetesPedigreeFunction",
            info="Diabetes pedigree function, a score related to family history of diabetes.",
            value=0.47,
        )
        age = gr.Number(
            label="Age",
            info="Age in years.",
            value=33,
            precision=0,
        )

    predict_button = gr.Button("Predict", variant="primary")
    output = gr.Textbox(label="Prediction Result")

    predict_button.click(
        fn=predict_diabetes,
        inputs=[
            pregnancies,
            glucose,
            blood_pressure,
            skin_thickness,
            insulin,
            bmi,
            diabetes_pedigree_function,
            age,
        ],
        outputs=output,
    )

    # gr.Examples(
    #     examples=[
    #         [6, 148, 72, 35, 0, 33.6, 0.627, 50],
    #         [1, 85, 66, 29, 0, 26.6, 0.351, 31],
    #         [8, 183, 64, 0, 0, 23.3, 0.672, 32],
    #     ],
    #     inputs=[
    #         pregnancies,
    #         glucose,
    #         blood_pressure,
    #         skin_thickness,
    #         insulin,
    #         bmi,
    #         diabetes_pedigree_function,
    #         age,
    #     ],
    # )


app = FastAPI(title="Diabetes Predictor API")
app = gr.mount_gradio_app(app, demo, path="/")


if __name__ == "__main__":
    demo.launch()
