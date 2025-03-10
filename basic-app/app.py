from shiny import reactive, req
from shiny.express import ui, render, input, app
import h2o
import pandas as pd

# Initialize h2o
h2o.init()

# Load the saved model
model = h2o.load_model("StackedEnsemble_BestOfFamily_4_AutoML_1_20250310_151241")

ui.page_opts(title="Weight Prediction App")

with ui.sidebar():
    ui.input_numeric(
        "age",
        "Age (years)",
        value=30,
        min=0,
        max=120
    )
    ui.input_select(
        "sex",
        "Sex",
        choices=["Male", "Female"]
    )
    ui.input_numeric(
        "height",
        "Height (cm)",
        value=170,
        min=100,
        max=250
    )
    ui.input_numeric(
        "cc",
        "Calf Circumference (cm)",
        value=35,
        min=10,
        max=100
    )
    ui.input_numeric(
        "muac",
        "Mid-Upper Arm Circumference (cm)",
        value=30,
        min=10,
        max=100
    )
    ui.input_select(
        "bmi_cat",
        "BMI Category",
        choices=["Normal", "Overweight", "Underweight", "Obese"]
    )
    ui.input_action_button("predict", "Predict Weight")

with ui.card(full_screen=True):
    ui.card_header("Prediction Result")
    
    # Store prediction result
    prediction_result = reactive.value("")
    
    @reactive.effect
    @reactive.event(input.predict)
    def predict_weight():
        # Create a single row dataframe with the input values
        data = pd.DataFrame({
            "age": [input.age()],
            "sex": [input.sex()],
            "height": [input.height()],
            "cc": [input.cc()],
            "muac": [input.muac()],
            "bmi_cat": [input.bmi_cat()]
        })
        
        # Convert to h2o frame
        h2o_data = h2o.H2OFrame(data)
        
        # Make prediction
        prediction = model.predict(h2o_data)
        predicted_weight = prediction.as_data_frame()['predict'][0]
        
        # Update the reactive value
        prediction_result.set(f"Predicted Weight: {predicted_weight:.1f} kg")
    
    @render.text
    def prediction():
        return prediction_result()
