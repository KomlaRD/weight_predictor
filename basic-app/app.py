from pathlib import Path

from shiny import reactive
from shiny.express import ui, render, input

import h2o
import pandas as pd
import joblib
import numpy as np


# ============================================================
# PATHS
# ============================================================

APP_DIR = Path(__file__).resolve().parent

weight_model_path = (
    APP_DIR
    / "StackedEnsemble_BestOfFamily_4_AutoML_1_20260927_111656"
)

height_model_path = (
    APP_DIR
    / "huber_regressor_model.joblib"
)


# ============================================================
# INITIALIZE H2O
# ============================================================

h2o.init()


# ============================================================
# LOAD MODELS
# ============================================================

weight_model = h2o.load_model(str(weight_model_path))
height_model = joblib.load(str(height_model_path))


# ============================================================
# REACTIVE VALUES
# ============================================================

weight_prediction = reactive.value("")
height_prediction = reactive.value("")


# ============================================================
# PAGE OPTIONS
# ============================================================

ui.page_opts(
    title="AnthroPredictor",
    fillable=True,
)


# ============================================================
# MATERIAL 3 INSPIRED CSS
# ============================================================

ui.tags.style(
    """
    /* =======================================================
       MATERIAL 3 DESIGN TOKENS
       ======================================================= */

    :root {
        --md-primary: #006A60;
        --md-on-primary: #FFFFFF;
        --md-primary-container: #74F8E5;
        --md-on-primary-container: #00201C;

        --md-secondary: #4A635F;
        --md-secondary-container: #CCE8E2;
        --md-on-secondary-container: #06201C;

        --md-tertiary: #456179;
        --md-tertiary-container: #CCE5FF;

        --md-surface: #F5FBF8;
        --md-surface-container-low: #EFF5F2;
        --md-surface-container: #E9EFEC;
        --md-surface-container-high: #E3E9E6;
        --md-surface-container-highest: #DDE4E1;

        --md-on-surface: #171D1B;
        --md-on-surface-variant: #3F4946;

        --md-outline: #6F7976;
        --md-outline-variant: #BEC9C5;

        --md-error: #BA1A1A;

        --md-radius-small: 12px;
        --md-radius-medium: 16px;
        --md-radius-large: 24px;
        --md-radius-xl: 28px;

        --md-shadow:
            0 1px 2px rgba(0, 0, 0, 0.05),
            0 2px 6px rgba(0, 0, 0, 0.04);
    }


    /* =======================================================
       GLOBAL
       ======================================================= */

    html,
    body {
        background: var(--md-surface);
        color: var(--md-on-surface);
    }

    body {
        font-family:
            Inter,
            system-ui,
            -apple-system,
            BlinkMacSystemFont,
            "Segoe UI",
            sans-serif;
    }

    .container-fluid {
        max-width: 1180px;
        margin: 0 auto;
        padding-left: 24px;
        padding-right: 24px;
    }


    /* =======================================================
       APP HEADER
       ======================================================= */

    .app-header {
        padding: 28px 4px 22px 4px;
    }

    .brand-row {
        display: flex;
        align-items: center;
        gap: 14px;
    }

    .brand-icon {
        width: 52px;
        height: 52px;

        display: flex;
        align-items: center;
        justify-content: center;

        border-radius: 18px;

        background: var(--md-primary-container);
        color: var(--md-on-primary-container);

        font-size: 26px;
    }

    .brand-title {
        margin: 0;
        font-size: 1.65rem;
        font-weight: 700;
        letter-spacing: -0.025em;
    }

    .brand-subtitle {
        margin: 3px 0 0 0;
        color: var(--md-on-surface-variant);
        font-size: 0.95rem;
    }


    /* =======================================================
       NAVIGATION
       ======================================================= */

    .nav-tabs {
        border-bottom: 1px solid var(--md-outline-variant);
        gap: 8px;
        margin-bottom: 24px;
    }

    .nav-tabs .nav-link {
        border: none !important;
        border-radius: 24px 24px 0 0;

        padding: 12px 20px;

        color: var(--md-on-surface-variant);
        font-weight: 600;

        background: transparent;

        transition:
            background-color 0.2s ease,
            color 0.2s ease;
    }

    .nav-tabs .nav-link:hover {
        background: var(--md-surface-container);
        color: var(--md-on-surface);
    }

    .nav-tabs .nav-link.active {
        color: var(--md-primary);
        background: var(--md-secondary-container);
        border-bottom: 3px solid var(--md-primary) !important;
    }


    /* =======================================================
       PREDICTION WORKSPACE
       ======================================================= */

    .prediction-grid {
        display: grid;
        grid-template-columns:
            minmax(0, 1.15fr)
            minmax(320px, 0.85fr);

        gap: 24px;
        align-items: start;

        padding-bottom: 40px;
    }


    /* =======================================================
       MATERIAL CARDS
       ======================================================= */

    .material-card {
        background: #FFFFFF;

        border: 1px solid var(--md-outline-variant);
        border-radius: var(--md-radius-xl);

        padding: 26px;

        box-shadow: var(--md-shadow);
    }

    .card-eyebrow {
        color: var(--md-primary);

        font-size: 0.78rem;
        font-weight: 700;
        letter-spacing: 0.08em;

        text-transform: uppercase;

        margin-bottom: 7px;
    }

    .section-title {
        margin: 0;

        color: var(--md-on-surface);

        font-size: 1.35rem;
        font-weight: 700;
        letter-spacing: -0.02em;
    }

    .section-description {
        margin-top: 7px;
        margin-bottom: 24px;

        color: var(--md-on-surface-variant);
        line-height: 1.55;
    }


    /* =======================================================
       FORM LAYOUT
       ======================================================= */

    .field-grid {
        display: grid;
        grid-template-columns: repeat(2, minmax(0, 1fr));
        gap: 6px 18px;
    }

    .field-full {
        grid-column: 1 / -1;
    }

    .form-group {
        margin-bottom: 18px;
    }

    .form-label,
    label {
        color: var(--md-on-surface);
        font-weight: 600;
        margin-bottom: 7px;
    }


    /* =======================================================
       MATERIAL-LIKE INPUTS
       ======================================================= */

    .form-control,
    .form-select {
        min-height: 52px;

        border: 1px solid var(--md-outline);
        border-radius: var(--md-radius-small);

        background-color: #FFFFFF;

        padding: 12px 14px;

        color: var(--md-on-surface);

        transition:
            border-color 0.15s ease,
            box-shadow 0.15s ease;
    }

    .form-control:hover,
    .form-select:hover {
        border-color: var(--md-on-surface-variant);
    }

    .form-control:focus,
    .form-select:focus {
        border: 2px solid var(--md-primary);
        box-shadow: none;
    }


    /* =======================================================
       ACTION BUTTON
       ======================================================= */

    .btn-primary {
        width: 100%;

        min-height: 52px;

        margin-top: 8px;

        border: none;
        border-radius: 26px;

        background: var(--md-primary);
        color: var(--md-on-primary);

        font-weight: 700;

        padding: 12px 24px;

        box-shadow:
            0 1px 3px rgba(0, 0, 0, 0.12);

        transition:
            transform 0.15s ease,
            box-shadow 0.15s ease,
            background-color 0.15s ease;
    }

    .btn-primary:hover,
    .btn-primary:focus {
        background: #005A51;
        color: #FFFFFF;

        box-shadow:
            0 2px 7px rgba(0, 0, 0, 0.16);
    }

    .btn-primary:active {
        transform: scale(0.985);
    }


    /* =======================================================
       RESULT PANEL
       ======================================================= */

    .result-card {
        position: sticky;
        top: 24px;

        overflow: hidden;

        background:
            linear-gradient(
                145deg,
                var(--md-primary-container),
                #E4FFF8
            );

        border: none;
        border-radius: var(--md-radius-xl);

        padding: 30px;

        min-height: 310px;

        display: flex;
        flex-direction: column;
        justify-content: space-between;
    }

    .result-icon {
        width: 58px;
        height: 58px;

        display: flex;
        align-items: center;
        justify-content: center;

        border-radius: 20px;

        background: rgba(255, 255, 255, 0.65);

        color: var(--md-primary);

        font-size: 28px;

        margin-bottom: 24px;
    }

    .result-label {
        color: var(--md-on-primary-container);

        font-size: 0.82rem;
        font-weight: 700;
        letter-spacing: 0.08em;

        text-transform: uppercase;

        margin-bottom: 10px;
    }

    .prediction-output {
        color: var(--md-on-primary-container);

        font-size: clamp(1.7rem, 4vw, 2.55rem);
        font-weight: 750;
        letter-spacing: -0.035em;
        line-height: 1.15;

        min-height: 60px;
    }

    .result-help {
        margin-top: 24px;

        color: var(--md-on-surface-variant);

        font-size: 0.88rem;
        line-height: 1.5;
    }


    /* =======================================================
       INFORMATION NOTE
       ======================================================= */

    .info-note {
        display: flex;
        gap: 10px;

        margin-top: 20px;
        padding: 14px 16px;

        background: var(--md-surface-container-low);

        border-radius: var(--md-radius-medium);

        color: var(--md-on-surface-variant);

        font-size: 0.86rem;
        line-height: 1.5;
    }

    .info-note-icon {
        color: var(--md-primary);
        font-weight: 700;
    }


    /* =======================================================
       MOBILE RESPONSIVENESS
       ======================================================= */

    @media (max-width: 768px) {

        .container-fluid {
            padding-left: 14px;
            padding-right: 14px;
        }

        .app-header {
            padding-top: 18px;
            padding-bottom: 16px;
        }

        .brand-icon {
            width: 46px;
            height: 46px;
            border-radius: 16px;
        }

        .brand-title {
            font-size: 1.35rem;
        }

        .brand-subtitle {
            font-size: 0.86rem;
        }

        .prediction-grid {
            grid-template-columns: 1fr;
            gap: 16px;
        }

        .field-grid {
            grid-template-columns: 1fr;
        }

        .field-full {
            grid-column: auto;
        }

        .material-card {
            padding: 20px;
            border-radius: 22px;
        }

        .result-card {
            position: static;

            min-height: 230px;

            padding: 24px;

            order: -1;
        }

        .result-icon {
            width: 50px;
            height: 50px;
            margin-bottom: 18px;
        }

        .nav-tabs {
            display: flex;
            flex-wrap: nowrap;

            overflow-x: auto;
            overflow-y: hidden;

            scrollbar-width: none;
        }

        .nav-tabs::-webkit-scrollbar {
            display: none;
        }

        .nav-tabs .nav-item {
            flex: 1 0 auto;
        }

        .nav-tabs .nav-link {
            width: 100%;
            white-space: nowrap;
            text-align: center;

            padding-left: 14px;
            padding-right: 14px;
        }

        .form-control,
        .form-select,
        .btn-primary {
            min-height: 54px;
        }
    }


    @media (max-width: 420px) {

        .brand-subtitle {
            max-width: 260px;
        }

        .material-card {
            padding: 18px;
        }

        .section-title {
            font-size: 1.2rem;
        }

        .prediction-output {
            font-size: 1.8rem;
        }
    }
    """
)


# ============================================================
# APP HEADER
# ============================================================

with ui.div(class_="app-header"):

    with ui.div(class_="brand-row"):

        with ui.div(class_="brand-icon"):
            ui.HTML("&#9878;")

        with ui.div():
            ui.h1(
                "AnthroPredictor",
                class_="brand-title"
            )

            ui.p(
                "Clinical anthropometric estimation",
                class_="brand-subtitle"
            )


# ============================================================
# NAVIGATION
# ============================================================

with ui.navset_tab():

    # ========================================================
    # WEIGHT PREDICTION
    # ========================================================

    with ui.nav_panel("Weight Prediction"):

        with ui.div(class_="prediction-grid"):

            # ------------------------------------------------
            # INPUT CARD
            # ------------------------------------------------

            with ui.div(class_="material-card"):

                ui.div(
                    "Weight estimation",
                    class_="card-eyebrow"
                )

                ui.h2(
                    "Patient measurements",
                    class_="section-title"
                )

                ui.p(
                    (
                        "Enter the patient's demographic and "
                        "anthropometric measurements to estimate "
                        "body weight."
                    ),
                    class_="section-description"
                )

                with ui.div(class_="field-grid"):

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
                        "Calf circumference (cm)",
                        value=35,
                        min=10,
                        max=100
                    )

                    ui.input_numeric(
                        "muac",
                        "Mid-upper arm circumference (cm)",
                        value=30,
                        min=10,
                        max=100
                    )

                    ui.input_select(
                        "bmi_cat",
                        "BMI category",
                        choices=[
                            "Normal",
                            "Overweight",
                            "Underweight",
                            "Obese"
                        ]
                    )

                ui.input_action_button(
                    "predict_weight",
                    "Estimate weight",
                    class_="btn-primary"
                )

                with ui.div(class_="info-note"):

                    ui.span(
                        "ⓘ",
                        class_="info-note-icon"
                    )

                    ui.span(
                        (
                            "Use the most accurate available "
                            "anthropometric measurements. "
                            "The estimate should support, not replace, "
                            "clinical judgement."
                        )
                    )

            # ------------------------------------------------
            # RESULT CARD
            # ------------------------------------------------

            with ui.div(class_="result-card"):

                with ui.div():

                    with ui.div(class_="result-icon"):
                        ui.HTML("&#9878;")

                    ui.div(
                        "Estimated body weight",
                        class_="result-label"
                    )

                    with ui.div(class_="prediction-output"):

                        @render.text
                        def weight_prediction_display():

                            if weight_prediction() == "":
                                return "Ready to estimate"

                            return weight_prediction()

                ui.p(
                    (
                        "Complete the measurements and select "
                        "Estimate weight to generate a prediction."
                    ),
                    class_="result-help"
                )

    # ========================================================
    # HEIGHT PREDICTION
    # ========================================================

    with ui.nav_panel("Height Prediction"):

        with ui.div(class_="prediction-grid"):

            # ------------------------------------------------
            # INPUT CARD
            # ------------------------------------------------

            with ui.div(class_="material-card"):

                ui.div(
                    "Height estimation",
                    class_="card-eyebrow"
                )

                ui.h2(
                    "Patient measurements",
                    class_="section-title"
                )

                ui.p(
                    (
                        "Enter the patient's demographic information "
                        "and ulna length to estimate standing height."
                    ),
                    class_="section-description"
                )

                with ui.div(class_="field-grid"):

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

                    with ui.div(class_="field-full"):

                        ui.input_numeric(
                            "ulna",
                            "Ulna length (cm)",
                            value=25,
                            min=10,
                            max=50
                        )

                ui.input_action_button(
                    "predict_height",
                    "Estimate height",
                    class_="btn-primary"
                )

                with ui.div(class_="info-note"):

                    ui.span(
                        "ⓘ",
                        class_="info-note-icon"
                    )

                    ui.span(
                        (
                            "Measure ulna length carefully using a "
                            "standardised anthropometric technique "
                            "before generating the estimate."
                        )
                    )

            # ------------------------------------------------
            # RESULT CARD
            # ------------------------------------------------

            with ui.div(class_="result-card"):

                with ui.div():

                    with ui.div(class_="result-icon"):
                        ui.HTML("&#8597;")

                    ui.div(
                        "Estimated height",
                        class_="result-label"
                    )

                    with ui.div(class_="prediction-output"):

                        @render.text
                        def height_prediction_display():

                            if height_prediction() == "":
                                return "Ready to estimate"

                            return height_prediction()

                ui.p(
                    (
                        "Complete the measurements and select "
                        "Estimate height to generate a prediction."
                    ),
                    class_="result-help"
                )


# ============================================================
# WEIGHT PREDICTION LOGIC
# ============================================================

@reactive.effect
@reactive.event(input.predict_weight)
def predict_weight():

    data = pd.DataFrame({
        "age": [input.weight_age()],
        "sex": [input.weight_sex()],
        "height": [input.height()],
        "cc": [input.cc()],
        "muac": [input.muac()],
        "bmi_cat": [input.bmi_cat()]
    })

    # Convert pandas dataframe to H2OFrame
    h2o_data = h2o.H2OFrame(data)

    # Generate prediction
    prediction = weight_model.predict(h2o_data)

    predicted_weight = (
        prediction
        .as_data_frame(use_multi_thread=True)["predict"][0]
    )

    # Update displayed result
    weight_prediction.set(
        f"{predicted_weight:.1f} kg"
    )


# ============================================================
# HEIGHT PREDICTION LOGIC
# ============================================================

@reactive.effect
@reactive.event(input.predict_height)
def predict_height():

    age = input.height_age()

    gender_Male = (
        1 if input.height_sex() == "Male" else 0
    )

    mean_ulna = input.ulna()

    features = [
        [
            age,
            mean_ulna,
            gender_Male
        ]
    ]

    predicted_height = height_model.predict(features)[0]

    # Update displayed result
    height_prediction.set(
        f"{predicted_height:.1f} cm"
    )
