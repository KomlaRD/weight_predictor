from pathlib import Path

import h2o
import joblib
import pandas as pd
from shiny import reactive
from shiny.express import input, render, ui


# ============================================================
# PATHS AND MODELS
# ============================================================

APP_DIR = Path(__file__).resolve().parent

weight_model_path = (
    APP_DIR / "StackedEnsemble_BestOfFamily_4_AutoML_1_20260927_111656"
)
height_model_path = APP_DIR / "huber_regressor_model.joblib"

h2o.init()
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

ui.page_opts(title="AnthroPredictor", fillable=True)


# ============================================================
# MATERIAL 3-INSPIRED CSS
# ============================================================

ui.tags.style(
    """
    :root {
        --md-primary: #006A60;
        --md-on-primary: #FFFFFF;
        --md-primary-container: #74F8E5;
        --md-on-primary-container: #00201C;
        --md-secondary-container: #CCE8E2;
        --md-on-secondary-container: #06201C;
        --md-tertiary: #456179;
        --md-tertiary-container: #CCE5FF;
        --md-surface: #F5FBF8;
        --md-surface-container-low: #EFF5F2;
        --md-surface-container: #E9EFEC;
        --md-on-surface: #171D1B;
        --md-on-surface-variant: #3F4946;
        --md-outline: #6F7976;
        --md-outline-variant: #BEC9C5;
        --md-radius-small: 12px;
        --md-radius-medium: 16px;
        --md-radius-large: 24px;
        --md-radius-xl: 28px;
        --md-shadow: 0 1px 2px rgba(0,0,0,.05), 0 2px 6px rgba(0,0,0,.04);
    }

    html, body {
        background: var(--md-surface);
        color: var(--md-on-surface);
    }

    body {
        font-family: Inter, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }

    .container-fluid {
        max-width: 1180px;
        margin: 0 auto;
        padding-left: 24px;
        padding-right: 24px;
    }

    .app-header { padding: 28px 4px 22px; }
    .brand-row { display: flex; align-items: center; gap: 14px; }
    .brand-icon {
        width: 52px; height: 52px; display: flex; align-items: center;
        justify-content: center; border-radius: 18px;
        background: var(--md-primary-container); color: var(--md-on-primary-container);
        font-size: 26px; flex: 0 0 auto;
    }
    .brand-title { margin: 0; font-size: 1.65rem; font-weight: 700; letter-spacing: -.025em; }
    .brand-subtitle { margin: 3px 0 0; color: var(--md-on-surface-variant); font-size: .95rem; }

    .nav-tabs {
        border-bottom: 1px solid var(--md-outline-variant);
        gap: 8px; margin-bottom: 24px;
    }
    .nav-tabs .nav-link {
        border: none !important; border-radius: 24px 24px 0 0;
        padding: 12px 20px; color: var(--md-on-surface-variant);
        font-weight: 600; background: transparent;
    }
    .nav-tabs .nav-link:hover { background: var(--md-surface-container); color: var(--md-on-surface); }
    .nav-tabs .nav-link.active {
        color: var(--md-primary); background: var(--md-secondary-container);
        border-bottom: 3px solid var(--md-primary) !important;
    }

    .prediction-grid {
        display: grid;
        grid-template-columns: minmax(0, 1.15fr) minmax(320px, .85fr);
        gap: 24px; align-items: start; padding-bottom: 40px;
    }

    .material-card, .about-card {
        background: #FFFFFF; border: 1px solid var(--md-outline-variant);
        border-radius: var(--md-radius-xl); box-shadow: var(--md-shadow);
    }
    .material-card { padding: 26px; }
    .about-card { padding: 28px; margin-bottom: 20px; }

    .card-eyebrow, .warning-eyebrow {
        font-size: .78rem; font-weight: 700; letter-spacing: .08em;
        text-transform: uppercase; margin-bottom: 7px;
    }
    .card-eyebrow { color: var(--md-primary); }
    .warning-eyebrow { color: #8C1D18; }

    .section-title, .about-section-title {
        margin: 0; color: var(--md-on-surface); font-weight: 700; letter-spacing: -.02em;
    }
    .section-title { font-size: 1.35rem; }
    .about-section-title { font-size: 1.3rem; margin-bottom: 16px; }
    .section-description {
        margin: 7px 0 24px; color: var(--md-on-surface-variant); line-height: 1.55;
    }

    .field-grid {
        display: grid; grid-template-columns: repeat(2, minmax(0, 1fr));
        gap: 6px 18px;
    }
    .field-full { grid-column: 1 / -1; }
    .form-group { margin-bottom: 18px; }
    .form-label, label { color: var(--md-on-surface); font-weight: 600; margin-bottom: 7px; }
    .form-control, .form-select {
        min-height: 52px; border: 1px solid var(--md-outline);
        border-radius: var(--md-radius-small); background: #FFFFFF;
        padding: 12px 14px; color: var(--md-on-surface);
    }
    .form-control:focus, .form-select:focus {
        border: 2px solid var(--md-primary); box-shadow: none;
    }

    .btn-primary {
        width: 100%; min-height: 52px; margin-top: 8px; border: none;
        border-radius: 26px; background: var(--md-primary); color: var(--md-on-primary);
        font-weight: 700; padding: 12px 24px;
        box-shadow: 0 1px 3px rgba(0,0,0,.12);
    }
    .btn-primary:hover, .btn-primary:focus { background: #005A51; color: #FFFFFF; }
    .btn-primary:active { transform: scale(.985); }

    .result-card {
        position: sticky; top: 24px; overflow: hidden;
        background: linear-gradient(145deg, var(--md-primary-container), #E4FFF8);
        border-radius: var(--md-radius-xl); padding: 30px; min-height: 310px;
        display: flex; flex-direction: column; justify-content: space-between;
    }
    .result-icon {
        width: 58px; height: 58px; display: flex; align-items: center;
        justify-content: center; border-radius: 20px; background: rgba(255,255,255,.65);
        color: var(--md-primary); font-size: 28px; margin-bottom: 24px;
    }
    .result-label {
        color: var(--md-on-primary-container); font-size: .82rem; font-weight: 700;
        letter-spacing: .08em; text-transform: uppercase; margin-bottom: 10px;
    }
    .prediction-output {
        color: var(--md-on-primary-container); font-size: clamp(1.7rem, 4vw, 2.55rem);
        font-weight: 750; letter-spacing: -.035em; line-height: 1.15; min-height: 60px;
    }
    .prediction-output .shiny-text-output { font: inherit; color: inherit; }
    .result-help {
        margin-top: 24px; color: var(--md-on-surface-variant);
        font-size: .88rem; line-height: 1.5;
    }

    .info-note {
        display: flex; gap: 10px; margin-top: 20px; padding: 14px 16px;
        background: var(--md-surface-container-low); border-radius: var(--md-radius-medium);
        color: var(--md-on-surface-variant); font-size: .86rem; line-height: 1.5;
    }
    .info-note-icon { color: var(--md-primary); font-weight: 700; }

    .about-container { max-width: 900px; margin: 0 auto; padding-bottom: 50px; }
    .about-hero { text-align: center; padding: 38px 20px 34px; }
    .about-hero-icon {
        width: 68px; height: 68px; margin: 0 auto 18px; display: flex;
        align-items: center; justify-content: center; border-radius: 24px;
        background: var(--md-primary-container); color: var(--md-on-primary-container);
        font-size: 32px;
    }
    .about-title { margin: 0 0 10px; font-size: 2rem; font-weight: 750; letter-spacing: -.035em; }
    .about-lead {
        max-width: 680px; margin: 0 auto; color: var(--md-on-surface-variant);
        font-size: 1.05rem; line-height: 1.65;
    }
    .about-card p { color: var(--md-on-surface-variant); line-height: 1.7; margin-bottom: 16px; }

    .metric-grid {
        display: grid; grid-template-columns: repeat(5, minmax(100px, 1fr));
        gap: 10px; margin: 24px 0;
    }
    .metric-card {
        text-align: center; padding: 18px 10px;
        background: var(--md-secondary-container); border-radius: 18px;
    }
    .metric-value { color: var(--md-primary); font-size: 1.45rem; font-weight: 750; }
    .metric-label {
        margin-top: 4px; color: var(--md-on-secondary-container);
        font-size: .76rem; font-weight: 700; letter-spacing: .05em;
    }
    .metric-explanation { font-size: .9rem; }

    .study-link {
        display: inline-flex; align-items: center; margin-top: 4px; padding: 11px 18px;
        border-radius: 22px; background: var(--md-secondary-container);
        color: var(--md-primary); font-weight: 700; text-decoration: none;
    }
    .study-link:hover { background: var(--md-surface-container); color: var(--md-primary); }

    .limitation-card { background: #FFF8F6; }
    .limitation-list { margin: 18px 0 0; padding-left: 22px; color: var(--md-on-surface-variant); }
    .limitation-list li { margin-bottom: 12px; line-height: 1.6; }

    .clinical-notice {
        display: flex; gap: 16px; padding: 22px; margin-bottom: 20px;
        background: var(--md-tertiary-container); border-radius: var(--md-radius-large);
    }
    .notice-icon { flex-shrink: 0; color: var(--md-tertiary); font-size: 25px; }
    .clinical-notice h4 { margin: 0 0 7px; font-size: 1rem; font-weight: 700; }
    .clinical-notice p {
        margin: 0; color: var(--md-on-surface-variant); font-size: .9rem; line-height: 1.6;
    }
    .citation-card { background: var(--md-surface-container-low); }
    .citation-text { margin-bottom: 0 !important; font-size: .9rem; }
    .about-footer {
        text-align: center; padding: 20px; color: var(--md-on-surface-variant); font-size: .82rem;
    }

    @media (max-width: 768px) {
        .container-fluid { padding-left: 14px; padding-right: 14px; }
        .app-header { padding-top: 18px; padding-bottom: 16px; }
        .brand-icon { width: 46px; height: 46px; border-radius: 16px; }
        .brand-title { font-size: 1.35rem; }
        .brand-subtitle { font-size: .86rem; }
        .prediction-grid { grid-template-columns: 1fr; gap: 16px; }
        .field-grid { grid-template-columns: 1fr; }
        .field-full { grid-column: auto; }
        .material-card { padding: 20px; border-radius: 22px; }
        .result-card { position: static; min-height: 230px; padding: 24px; order: -1; }
        .result-icon { width: 50px; height: 50px; margin-bottom: 18px; }
        .nav-tabs { display: flex; flex-wrap: nowrap; overflow-x: auto; overflow-y: hidden; scrollbar-width: none; }
        .nav-tabs::-webkit-scrollbar { display: none; }
        .nav-tabs .nav-item { flex: 1 0 auto; }
        .nav-tabs .nav-link { width: 100%; white-space: nowrap; text-align: center; padding-left: 14px; padding-right: 14px; }
        .form-control, .form-select, .btn-primary { min-height: 54px; }
        .about-hero { padding: 26px 12px 24px; }
        .about-title { font-size: 1.65rem; }
        .about-lead { font-size: .96rem; }
        .about-card { padding: 21px; border-radius: 22px; }
        .metric-grid { grid-template-columns: repeat(2, 1fr); }
        .metric-card:last-child { grid-column: 1 / -1; }
        .clinical-notice { padding: 18px; }
    }

    @media (max-width: 420px) {
        .brand-subtitle { max-width: 260px; }
        .material-card { padding: 18px; }
        .section-title { font-size: 1.2rem; }
        .prediction-output { font-size: 1.8rem; }
        .about-hero-icon { width: 58px; height: 58px; border-radius: 20px; font-size: 27px; }
        .about-title { font-size: 1.5rem; }
    }
    """
)


# ============================================================
# REUSABLE UI HELPERS
# ============================================================

def info_note(text):
    return ui.div(
        ui.tags.span("ⓘ", class_="info-note-icon"),
        ui.tags.span(text),
        class_="info-note",
    )


def metric_card(value, label):
    return ui.div(
        ui.div(value, class_="metric-value"),
        ui.div(label, class_="metric-label"),
        class_="metric-card",
    )


# ============================================================
# APP HEADER
# ============================================================

ui.div(
    ui.div(
        ui.div(ui.HTML("&#9878;"), class_="brand-icon"),
        ui.div(
            ui.h1("AnthroPredictor", class_="brand-title"),
            ui.p("Clinical anthropometric estimation", class_="brand-subtitle"),
        ),
        class_="brand-row",
    ),
    class_="app-header",
)


# ============================================================
# NAVIGATION
# ============================================================

with ui.navset_tab():

    # --------------------------------------------------------
    # WEIGHT PREDICTION
    # --------------------------------------------------------
    with ui.nav_panel("Weight Prediction"):
        with ui.layout_columns(
            col_widths={"sm": (12, 12), "lg": (7, 5)},
            gap="24px",
            fill=False,
            class_="prediction-grid",
        ):
            with ui.card(class_="material-card"):
                ui.div("Weight estimation", class_="card-eyebrow")
                ui.h2("Patient measurements", class_="section-title")
                ui.p(
                    "Enter the patient's demographic and anthropometric measurements "
                    "to estimate body weight.",
                    class_="section-description",
                )

                with ui.layout_columns(
                    col_widths={"sm": (12, 12), "md": (6, 6)},
                    gap="18px",
                    fill=False,
                    class_="field-grid",
                ):
                    ui.input_numeric(
                        "weight_age", "Age (years)", value=30, min=0, max=120
                    )
                    ui.input_select(
                        "weight_sex", "Sex", choices=["Male", "Female"]
                    )
                    ui.input_numeric(
                        "height", "Height (cm)", value=170, min=100, max=250
                    )
                    ui.input_numeric(
                        "cc", "Calf circumference (cm)", value=35, min=10, max=100
                    )
                    ui.input_numeric(
                        "muac",
                        "Mid-upper arm circumference (cm)",
                        value=30,
                        min=10,
                        max=100,
                    )
                    ui.input_select(
                        "bmi_cat",
                        "BMI category",
                        choices=["Normal", "Overweight",
                                 "Underweight", "Obese"],
                    )

                ui.input_action_button(
                    "predict_weight", "Estimate weight", class_="btn-primary"
                )
                info_note(
                    "Use the most accurate available anthropometric measurements. "
                    "The estimate should support, not replace, clinical judgement."
                )

            @render.express
            def weight_prediction_display():
                with ui.card(class_="result-card"):
                    ui.div(ui.HTML("&#9878;"), class_="result-icon")
                    ui.div("Estimated body weight", class_="result-label")
                    ui.div(
                        "Ready to estimate"
                        if weight_prediction() == ""
                        else weight_prediction(),
                        class_="prediction-output",
                    )
                    ui.p(
                        "Complete the measurements and select Estimate weight "
                        "to generate a prediction.",
                        class_="result-help",
                    )

    # --------------------------------------------------------
    # HEIGHT PREDICTION
    # --------------------------------------------------------
    with ui.nav_panel("Height Prediction"):
        with ui.layout_columns(
            col_widths={"sm": (12, 12), "lg": (7, 5)},
            gap="24px",
            fill=False,
            class_="prediction-grid",
        ):
            with ui.card(class_="material-card"):
                ui.div("Height estimation", class_="card-eyebrow")
                ui.h2("Patient measurements", class_="section-title")
                ui.p(
                    "Enter the patient's demographic information and ulna length "
                    "to estimate standing height.",
                    class_="section-description",
                )

                with ui.layout_columns(
                    col_widths={"sm": (12, 12, 12), "md": (6, 6, 12)},
                    gap="18px",
                    fill=False,
                    class_="field-grid",
                ):
                    ui.input_numeric(
                        "height_age", "Age (years)", value=30, min=0, max=120
                    )
                    ui.input_select(
                        "height_sex", "Sex", choices=["Male", "Female"]
                    )
                    ui.input_numeric(
                        "ulna", "Ulna length (cm)", value=25, min=10, max=50
                    )

                ui.input_action_button(
                    "predict_height", "Estimate height", class_="btn-primary"
                )
                info_note(
                    "Measure ulna length carefully using a standardised anthropometric "
                    "technique before generating the estimate."
                )

            @render.express
            def height_prediction_display():
                with ui.card(class_="result-card"):
                    ui.div(ui.HTML("&#8597;"), class_="result-icon")
                    ui.div("Estimated height", class_="result-label")
                    ui.div(
                        "Ready to estimate"
                        if height_prediction() == ""
                        else height_prediction(),
                        class_="prediction-output",
                    )
                    ui.p(
                        "Complete the measurements and select Estimate height "
                        "to generate a prediction.",
                        class_="result-help",
                    )

    # --------------------------------------------------------
    # ABOUT
    # --------------------------------------------------------
    with ui.nav_panel("About"):
        ui.div(
            ui.div(
                ui.div(ui.HTML("&#9878;"), class_="about-hero-icon"),
                ui.h2("About AnthroPredictor", class_="about-title"),
                ui.p(
                    "AnthroPredictor is a clinical decision-support tool designed to "
                    "estimate adult body weight and height when direct measurement is "
                    "unavailable or impractical.",
                    class_="about-lead",
                ),
                class_="about-hero",
            ),
            ui.div(
                ui.div("Purpose", class_="card-eyebrow"),
                ui.h3("Why AnthroPredictor?", class_="about-section-title"),
                ui.p(
                    "Accurate body weight and height measurements are important for "
                    "nutrition assessment, estimation of nutritional requirements, "
                    "medication dosing, and other aspects of clinical care. Direct "
                    "measurement may not always be feasible, particularly for bedridden "
                    "or critically ill patients or where suitable equipment is unavailable."
                ),
                ui.p(
                    "AnthroPredictor provides alternative estimates using demographic "
                    "and anthropometric measurements. These estimates are intended to "
                    "support clinical decision-making when measured weight or height "
                    "cannot be obtained."
                ),
                class_="about-card",
            ),
            ui.div(
                ui.div("Weight estimation", class_="card-eyebrow"),
                ui.h3("Weight prediction model", class_="about-section-title"),
                ui.p(
                    "The weight model was developed from a cross-sectional study of "
                    "389 adults at Cape Coast Teaching Hospital, Ghana. Standardised "
                    "measurements included weight, height, mid-upper arm circumference "
                    "(MUAC), and calf circumference (CC)."
                ),
                ui.p(
                    "The dataset was partitioned into 80% training and 20% testing data. "
                    "Models were trained using the H2O.ai AutoML framework. The selected "
                    "stacked ensemble uses age, sex, height, MUAC, calf circumference, "
                    "and BMI category to estimate body weight."
                ),
                ui.div(
                    metric_card("3 kg", "MAE"),
                    metric_card("4 kg", "RMSE"),
                    metric_card("0.90", "R²"),
                    metric_card("90%", "P10"),
                    metric_card("100%", "P20"),
                    class_="metric-grid",
                ),
                ui.p(
                    "In the study test data, the stacked ensemble achieved a mean "
                    "absolute error of approximately 3 kg, RMSE of approximately 4 kg, "
                    "R² of 0.90, and P10 and P20 values of 90% and 100%, respectively.",
                    class_="metric-explanation",
                ),
                ui.tags.a(
                    "Read the published weight study →",
                    href="https://proceedings.mlr.press/v302/anku26a.html",
                    target="_blank",
                    rel="noopener noreferrer",
                    class_="study-link",
                ),
                class_="about-card",
            ),
            ui.div(
                ui.div("Height estimation", class_="card-eyebrow"),
                ui.h3("Height prediction model", class_="about-section-title"),
                ui.p(
                    "The height model uses age, sex, and ulna length to estimate standing "
                    "height. It was developed using data derived from a cross-sectional "
                    "study involving 384 adult outpatients at Cape Coast Teaching Hospital."
                ),
                ui.p(
                    "The source study compared measured standing height with estimates "
                    "derived from published ulna-length equations and found clinically "
                    "important disagreement, supporting the need for population-appropriate "
                    "height-estimation approaches. The AnthroPredictor height component "
                    "uses a subsequently fitted Huber regression model based on age, sex, "
                    "and ulna length."
                ),
                ui.tags.a(
                    "View the height-study record →",
                    href=(
                        "https://scholar.google.com/citations?view_op=view_citation&hl=en&"
                        "user=42x5HGsAAAAJ&citation_for_view=42x5HGsAAAAJ:ZeXyd9-uunAC"
                    ),
                    target="_blank",
                    rel="noopener noreferrer",
                    class_="study-link",
                ),
                class_="about-card",
            ),
            ui.div(
                ui.div("Important", class_="warning-eyebrow"),
                ui.h3("Limitations and appropriate use",
                      class_="about-section-title"),
                ui.p(
                    "AnthroPredictor provides estimates rather than direct anthropometric "
                    "measurements. Predictions should be interpreted within the individual "
                    "patient's clinical context."
                ),
                ui.tags.ul(
                    ui.tags.li(
                        "Direct measurement of weight and height remains preferred whenever "
                        "it is feasible and reliable."
                    ),
                    ui.tags.li(
                        "Both source datasets were obtained at a single tertiary hospital "
                        "in Ghana; model performance may differ in other populations and settings."
                    ),
                    ui.tags.li(
                        "External validation in broader Ghanaian, African, and other "
                        "populations is needed before assuming equivalent performance."
                    ),
                    ui.tags.li(
                        "Prediction error remains possible even when all measurements are "
                        "entered correctly."
                    ),
                    ui.tags.li(
                        "Prediction accuracy depends on accurate and standardised input "
                        "anthropometric measurements."
                    ),
                    ui.tags.li(
                        "The tool should support rather than replace clinical judgement."
                    ),
                    class_="limitation-list",
                ),
                class_="about-card limitation-card",
            ),
            ui.div(
                ui.div(ui.HTML("&#9432;"), class_="notice-icon"),
                ui.div(
                    ui.h4("Clinical decision-support notice"),
                    ui.p(
                        "AnthroPredictor is intended to provide anthropometric estimates "
                        "to support clinical decision-making. Predictions should not be "
                        "interpreted as measured values or used as the sole basis for "
                        "clinical decisions."
                    ),
                ),
                class_="clinical-notice",
            ),
            ui.div(
                ui.div("Research", class_="card-eyebrow"),
                ui.h3("Weight-model publication",
                      class_="about-section-title"),
                ui.p(
                    "Anku E, Sam M, Mohammed S, Asa-Atiemo A, Ekor O. Adult weight "
                    "estimation in a tertiary hospital in Ghana. DLI 2025 Research Track. "
                    "Proceedings of Machine Learning Research. 2026;302:1–11.",
                    class_="citation-text",
                ),
                class_="about-card citation-card",
            ),
            ui.div(
                ui.tags.strong("AnthroPredictor"),
                ui.tags.span(" • Clinical anthropometric estimation"),
                class_="about-footer",
            ),
            class_="about-container",
        )


# ============================================================
# WEIGHT PREDICTION LOGIC
# ============================================================

@reactive.effect
@reactive.event(input.predict_weight)
def predict_weight():
    data = pd.DataFrame(
        {
            "age": [input.weight_age()],
            "sex": [input.weight_sex()],
            "height": [input.height()],
            "cc": [input.cc()],
            "muac": [input.muac()],
            "bmi_cat": [input.bmi_cat()],
        }
    )

    h2o_data = h2o.H2OFrame(data)
    prediction = weight_model.predict(h2o_data)
    predicted_weight = prediction.as_data_frame(
        use_multi_thread=True)["predict"][0]

    weight_prediction.set(f"{predicted_weight:.1f} kg")


# ============================================================
# HEIGHT PREDICTION LOGIC
# ============================================================

@reactive.effect
@reactive.event(input.predict_height)
def predict_height():
    age = input.height_age()
    gender_Male = 1 if input.height_sex() == "Male" else 0
    mean_ulna = input.ulna()

    features = [[age, mean_ulna, gender_Male]]
    predicted_height = height_model.predict(features)[0]

    height_prediction.set(f"{predicted_height:.1f} cm")
