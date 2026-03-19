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


with MODEL_PATH.open("rb") as model_file:
    model = pickle.load(model_file)


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

    with gr.Row():
        pregnancies = gr.Number(label="Pregnancies", value=1, precision=0)
        glucose = gr.Number(label="Glucose", value=120, precision=0)
        blood_pressure = gr.Number(label="BloodPressure", value=70, precision=0)
        skin_thickness = gr.Number(label="SkinThickness", value=20, precision=0)

    with gr.Row():
        insulin = gr.Number(label="Insulin", value=79, precision=0)
        bmi = gr.Number(label="BMI", value=28.5)
        diabetes_pedigree_function = gr.Number(
            label="DiabetesPedigreeFunction",
            value=0.47,
        )
        age = gr.Number(label="Age", value=33, precision=0)

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
