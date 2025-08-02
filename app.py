import gradio as gr
import numpy as np
import joblib
import matplotlib.pyplot as plt

# Load model and scaler
model = joblib.load("diabetes_model.pkl")
scaler = joblib.load("diabetes_scaler.pkl")

# Input feature names
feature_names = ['Glucose', 'BMI', 'Age', 'Insulin', 'SkinThickness']

# Sample synthetic data
sample_data = {
    "Healthy Person": [99, 22.5, 25, 50, 25],
    "Diabetic Person": [160, 35.2, 55, 200, 40]
}

# Prediction function
def predict(glucose, bmi, age, insulin, skin_thickness):
    input_array = np.array([[glucose, bmi, age, insulin, skin_thickness]])
    scaled_input = scaler.transform(input_array)
    
    prediction = model.predict(scaled_input)[0]
    prob = model.predict_proba(scaled_input)[0][1]

    # Plot input features
    fig, ax = plt.subplots()
    ax.barh(feature_names, input_array[0], color="teal" if prediction == 0 else "orangered")
    ax.set_title("Input Feature Values")
    ax.invert_yaxis()

    result = "✅ Not Diabetic" if prediction == 0 else "⚠️ Diabetic"
    prob_text = f"Prediction Confidence: {prob * 100:.2f}%"

    return result + "\n\n" + prob_text, fig

# Create UI
with gr.Blocks(theme=gr.themes.Base()) as demo:
    gr.Markdown("# 🩺 Diabetes Prediction App")
    gr.Markdown("Enter patient health data to check for diabetes.")

    with gr.Row():
        with gr.Column():
            sample_selector = gr.Dropdown(choices=list(sample_data.keys()), label="Choose Sample or Enter Manually", value=None)
            glucose = gr.Slider(0, 200, value=99, label="Glucose")
            bmi = gr.Slider(10, 60, value=22.0, label="BMI")
            age = gr.Slider(10, 90, value=25, label="Age")
            insulin = gr.Slider(0, 300, value=50, label="Insulin")
            skin = gr.Slider(0, 100, value=20, label="Skin Thickness")
            submit = gr.Button("Predict")

        with gr.Column():
            result_text = gr.Textbox(label="Result", lines=2)
            plot_output = gr.Plot()

    def fill_sample(choice):
        return sample_data.get(choice, [99, 22.0, 25, 50, 20])

    sample_selector.change(fn=fill_sample, inputs=sample_selector, 
                           outputs=[glucose, bmi, age, insulin, skin])
    submit.click(fn=predict, 
                 inputs=[glucose, bmi, age, insulin, skin], 
                 outputs=[result_text, plot_output])

demo.launch()
