from shiny import reactive, req
from shiny.express import ui, render, input, app
import h2o
import pandas as pd
import joblib
import numpy as np

# Initialize h2o
h2o.init()

# Load the saved models
weight_model = h2o.load_model("StackedEnsemble_BestOfFamily_4_AutoML_1_20250310_151241")
height_model = joblib.load("huber_regressor_model.joblib")

# Store prediction results as reactive values
weight_prediction = reactive.value("")
height_prediction = reactive.value("")

ui.page_opts(title="Anthro Prediction App", fillable=True)

with ui.navset_tab():
    # Weight Prediction Tab
    with ui.nav_panel("Weight Prediction"):
        with ui.layout_sidebar():
            with ui.sidebar():
                ui.input_numeric(
                    "weight_age",
                    "Age (years)",
                    value=30,
                    min=0,
                    max=120
                )
                ui.input_select(
                    "weight_sex",
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
                ui.input_action_button("predict_weight", "Predict Weight")

            # Main panel content for weight prediction
            with ui.card(
                full_screen=True,
                height="200px",
                title="Predicted Weight"
            ):
                @render.text
                def weight_prediction_display():
                    if weight_prediction() == "":
                        return "Click 'Predict Weight' to see result"
                    return weight_prediction()

    # Height Prediction Tab
    with ui.nav_panel("Height Prediction"):
        with ui.layout_sidebar():
            with ui.sidebar():
                ui.input_numeric(
                    "height_age",
                    "Age (years)",
                    value=30,
                    min=0,
                    max=120
                )
                ui.input_select(
                    "height_sex",
                    "Sex",
                    choices=["Male", "Female"]
                )
                ui.input_numeric(
                    "ulna",
                    "Ulna Length (cm)",
                    value=25,
                    min=10,
                    max=50
                )
                ui.input_action_button("predict_height", "Predict Height")

            # Main panel content for height prediction
            with ui.card(
                full_screen=True,
                height="200px",
                title="Predicted Height"
            ):
                @render.text
                def height_prediction_display():
                    if height_prediction() == "":
                        return "Click 'Predict Height' to see result"
                    return height_prediction()

# Define the weight prediction effect
@reactive.effect
@reactive.event(input.predict_weight)
def predict_weight():
    # Create a single row dataframe with the input values
    data = pd.DataFrame({
        "age": [input.weight_age()],
        "sex": [input.weight_sex()],
        "height": [input.height()],
        "cc": [input.cc()],
        "muac": [input.muac()],
        "bmi_cat": [input.bmi_cat()]
    })
    
    # Convert to h2o frame
    h2o_data = h2o.H2OFrame(data)
 
    # Make prediction
    prediction = weight_model.predict(h2o_data)
    predicted_weight = prediction.as_data_frame(use_multi_thread=True)['predict'][0]
    
    # Update the reactive value
    weight_prediction.set(f"Predicted Weight: {predicted_weight:.1f} kg")

# Define the height prediction effect
@reactive.effect
@reactive.event(input.predict_height)
def predict_height():
    # Get input values
    age = input.height_age()
    gender_Male = 1 if input.height_sex() == "Male" else 0  # Encoding sex as binary
    mean_ulna = input.ulna()
    
    # Create a single row for prediction
    # Adjust the format according to what your model expects
    # inputs = np.array([[age, mean_ulna, gender_Male]])
    features = [[age, mean_ulna, gender_Male]]
    # df_predict = pd.DataFrame(inputs, columns=["age", "mean_ulna", "gender_Male"])
    # Make prediction
    predicted_height = height_model.predict(features)[0]
    # predicted_height = height_model.predict(df_predict)
    
    # Update the reactive value
    height_prediction.set(f"Predicted Height: {predicted_height:.1f} cm")
