# AnthroPredictor

**Clinical anthropometric estimation when direct measurement is unavailable or impractical**

AnthroPredictor is an open-source clinical decision-support application for estimating **adult body weight and standing height** using demographic and anthropometric measurements.

The application is intended for situations in which direct measurement of weight or height is unavailable or impractical, including patients who cannot safely stand or be transferred for routine anthropometric measurement.

AnthroPredictor currently provides:

- **Weight estimation** using age, sex, height, mid-upper arm circumference (MUAC), calf circumference (CC), and BMI category.
- **Height estimation** using age, sex, and ulna length.
- A responsive **Shiny for Python Express** interface designed for desktop and mobile use.
- An integrated About section describing the scientific basis, limitations, and appropriate interpretation of the predictions.

> **Clinical notice:** AnthroPredictor provides model-derived estimates rather than direct anthropometric measurements. Predictions are intended to support, not replace, clinical judgement.

---

## Table of Contents

- [Background](#background)
- [Features](#features)
- [Scientific Basis](#scientific-basis)
  - [Weight Estimation](#weight-estimation)
  - [Height Estimation](#height-estimation)
- [Current Architecture](#current-architecture)
- [Technology Stack](#technology-stack)
- [Repository Structure](#repository-structure)
- [Installation](#installation)
- [Running the Application](#running-the-application)
- [Using AnthroPredictor](#using-anthropredictor)
- [Model Inputs and Outputs](#model-inputs-and-outputs)
- [Clinical Interpretation](#clinical-interpretation)
- [Limitations](#limitations)
- [Privacy and Data Handling](#privacy-and-data-handling)
- [Testing and Validation](#testing-and-validation)
- [Development Roadmap](#development-roadmap)
- [Citation](#citation)
- [Contributing](#contributing)
- [License](#license)
- [Disclaimer](#disclaimer)
- [Acknowledgements](#acknowledgements)

---

# Background

Accurate body weight and height measurements are important for nutrition assessment and many other aspects of clinical care.

Body weight may be required for:

- estimating energy and protein requirements;
- assessing nutritional status;
- calculating body mass index;
- monitoring changes in nutritional status;
- supporting weight-based medication dosing;
- estimating fluid requirements; and
- performing other weight-dependent clinical calculations.

Standing height is similarly important for BMI calculation, nutritional assessment, prediction equations, and other clinical applications.

However, obtaining direct measurements can be difficult in patients who cannot safely stand or transfer. This problem may be particularly relevant among bedridden or critically ill patients and in healthcare environments where bed scales, hoist scales, or other specialised anthropometric equipment are unavailable.

Clinicians may consequently rely on patient-reported values, visual estimation, or published anthropometric prediction equations.

An important limitation of existing prediction equations is that they may have been developed in populations that differ from the population in which they are subsequently applied.

AnthroPredictor explores a **locally informed, data-driven approach to anthropometric estimation**, initially using data collected from adults at **Cape Coast Teaching Hospital (CCTH), Ghana**.

---

# Features

## Weight prediction

AnthroPredictor estimates adult body weight from:

- age;
- sex;
- height;
- mid-upper arm circumference (MUAC);
- calf circumference (CC); and
- BMI category.

The current prediction engine uses an **H2O AutoML stacked ensemble model**.

## Height prediction

AnthroPredictor estimates adult standing height from:

- age;
- sex; and
- ulna length.

The current prediction engine uses a **Huber regression model**.

## User interface

The current application provides:

- separate Weight Prediction and Height Prediction workflows;
- responsive desktop and mobile layouts;
- clinically labelled anthropometric inputs;
- clearly displayed prediction results;
- a Material 3-inspired visual interface;
- an About section describing the scientific basis of the application;
- model limitations and appropriate-use information; and
- clinical decision-support notices.

---

# Scientific Basis

## Weight Estimation

The weight prediction model was developed from a cross-sectional study involving adult patients at **Cape Coast Teaching Hospital, Ghana**.

### Study population

The study included:

**389 adults**

The median age of participants was approximately **57 years**, and approximately **66% were female**.

Standardised anthropometric measurements included:

- measured body weight;
- standing height;
- mid-upper arm circumference; and
- calf circumference.

Each anthropometric variable was measured twice, with the mean measurement used for analysis.

### Model development

The dataset was partitioned into:

- **80% training data**
- **20% testing data**

Automated machine-learning algorithms were trained using the **H2O.ai AutoML framework**.

The final stacked ensemble model implemented in AnthroPredictor uses:

```text id="b1e9y3"
Age
Sex
Height
Mid-upper arm circumference
Calf circumference
BMI category
```

to estimate adult body weight.

### Model performance

The study reported approximately:

| Performance metric | Result |
|---|---:|
| Mean Absolute Error (MAE) | 3 kg |
| Root Mean Squared Error (RMSE) | 4 kg |
| R² | 0.90 |
| P10 | 90% |
| P20 | 100% |

**P10** represents the proportion of predictions falling within 10% of measured body weight, while **P20** represents predictions falling within 20%.

The AutoML-derived stacked ensemble outperformed the existing weight-estimation approaches evaluated in the study population.

### Publication

**Adult weight estimation in a tertiary hospital in Ghana**

Proceedings of Machine Learning Research, Volume 302.

https://proceedings.mlr.press/v302/anku26a.html

---

## Height Estimation

The height-estimation component originates from anthropometric research conducted among adults at **Cape Coast Teaching Hospital**.

### Source study

The source study included:

**384 adults**

The cross-sectional study compared directly measured standing height with height estimated using ulna length.

The study demonstrated important disagreement between measured height and estimates generated using existing published ulna-based equations in the study population.

These findings highlighted the limitations of applying prediction equations developed in other populations without appropriate local evaluation.

### Current AnthroPredictor model

The current AnthroPredictor implementation uses a **Huber regression model** developed using the anthropometric data originating from this research.

The model uses:

```text id="4qgbce"
Age
Ulna length
Sex
```

to estimate standing height.

### Important distinction

The existing ulna-based equations evaluated in the original study and the Huber regression model currently implemented in AnthroPredictor are **not the same prediction models**.

The 384-participant study provided the anthropometric dataset underlying subsequent model-development work.

Performance statistics reported for previously evaluated ulna equations should therefore **not** be interpreted as performance statistics for the current AnthroPredictor Huber model.

Detailed development and validation results for the final height model will be incorporated into the project documentation when the model-development work is formally reported.

---

# Current Architecture

The current application uses a local inference architecture:

```text id="r5jv2d"
┌───────────────────────────────┐
│             User              │
│       Desktop / Mobile        │
└───────────────┬───────────────┘
                │
                ▼
┌───────────────────────────────┐
│     Shiny for Python          │
│          Express              │
│                               │
│  • User interface             │
│  • Input collection           │
│  • Reactive processing        │
│  • Results presentation       │
└───────────────┬───────────────┘
                │
         ┌──────┴──────┐
         ▼             ▼
┌────────────────┐ ┌────────────────┐
│ Weight Model   │ │ Height Model   │
│                │ │                │
│ H2O AutoML     │ │ Huber          │
│ Stacked        │ │ Regression     │
│ Ensemble       │ │                │
└────────────────┘ └────────────────┘
```

Both models are currently loaded when the Shiny application starts, and predictions are generated locally by the application.

---

## Planned API Architecture

A planned development milestone is to separate the user-interface and model-inference layers using **FastAPI**.

The intended architecture is:

```text id="pqj34f"
Browser / User
      │
      ▼
Shiny Express
      │
      │ HTTPS / JSON
      ▼
FastAPI
      │
      ├── /api/v1/predict/weight
      │
      ├── /api/v1/predict/height
      │
      └── /health
      │
      ├── H2O weight model
      └── Huber height model
```

Under this architecture:

### Shiny Express

will manage:

- the user interface;
- data entry;
- user-facing validation;
- API requests; and
- presentation of prediction results.

### FastAPI

will manage:

- request validation;
- model loading;
- model inference;
- prediction responses;
- model-version metadata;
- API health checks; and
- future integration with other applications.

This separation is intended to improve maintainability, testability, deployment flexibility, model versioning, and future interoperability.

> The FastAPI architecture is planned and is not part of the current application release unless explicitly indicated in a future release.

---

# Technology Stack

| Component | Technology |
|---|---|
| Programming language | Python |
| Web application | Shiny for Python |
| UI approach | Shiny Express |
| Weight modelling | H2O.ai AutoML |
| Height modelling | Huber regression |
| Height model serialisation | joblib |
| Data processing | pandas |
| Planned API | FastAPI |
| Planned API server | Uvicorn |
| Version control | Git |
| Repository hosting | GitHub |
| License | MIT |

---

# Repository Structure

The current repository can be organised approximately as:

```text id="52hxps"
AnthroPredictor/
│
├── app.py
│
├── StackedEnsemble_BestOfFamily_4_AutoML_1_20260927_111656/
│
├── huber_regressor_model.joblib
│
├── requirements.txt
├── README.md
├── LICENSE
├── .gitignore
└── ...
```

As the API architecture is introduced, the project is expected to evolve toward a structure similar to:

```text id="56sn09"
AnthroPredictor/
│
├── api/
│   ├── __init__.py
│   ├── main.py
│   ├── schemas.py
│   ├── model_service.py
│   └── config.py
│
├── shiny_app/
│   └── app.py
│
├── models/
│   ├── weight/
│   │   └── ...
│   │
│   └── height/
│       └── huber_regressor_model.joblib
│
├── tests/
│   ├── test_weight_api.py
│   └── test_height_api.py
│
├── requirements.txt
├── README.md
├── LICENSE
├── .gitignore
└── ...
```

---

# Installation

## Prerequisites

A working Python installation is required.

A virtual environment is strongly recommended.

---

## 1. Clone the repository

```bash id="u2q1x7"
git clone <YOUR-REPOSITORY-URL>
cd AnthroPredictor
```

---

## 2. Create a virtual environment

```bash id="6cm42d"
python -m venv .venv
```

### Windows

```bash id="47qxai"
.venv\Scripts\activate
```

### macOS/Linux

```bash id="tawgo8"
source .venv/bin/activate
```

---

## 3. Install dependencies

```bash id="n0o4yx"
pip install -r requirements.txt
```

Current dependencies include packages such as:

```text id="09ecrh"
shiny
h2o
pandas
joblib
scikit-learn
```

Use the repository's `requirements.txt` as the authoritative dependency list.

---

# Running the Application

From the repository root, run:

```bash id="wh4gjc"
shiny run --reload app.py
```

Shiny will display the local address for the application in the terminal.

Open that address in a web browser.

For development, the `--reload` option automatically reloads the application when source files change.

---

# Using AnthroPredictor

## Weight Prediction

Open the **Weight Prediction** tab.

Enter:

1. Age
2. Sex
3. Height
4. Calf circumference
5. Mid-upper arm circumference
6. BMI category

Select:

**Estimate weight**

The application returns the predicted body weight in **kilograms (kg)**.

---

## Height Prediction

Open the **Height Prediction** tab.

Enter:

1. Age
2. Sex
3. Ulna length

Select:

**Estimate height**

The application returns the predicted standing height in **centimetres (cm)**.

---

## About

The **About** tab provides:

- the purpose of AnthroPredictor;
- background to the weight model;
- background to the height model;
- weight-model performance;
- important limitations;
- appropriate-use guidance; and
- links to relevant research.

---

# Model Inputs and Outputs

## Weight Model

| Variable | Type | Unit/Category |
|---|---|---|
| Age | Numeric | years |
| Sex | Categorical | Male/Female |
| Height | Numeric | cm |
| Calf circumference | Numeric | cm |
| MUAC | Numeric | cm |
| BMI category | Categorical | Underweight/Normal/Overweight/Obese |

### Output

```text id="t5cp46"
Estimated body weight (kg)
```

---

## Height Model

| Variable | Type | Unit/Category |
|---|---|---|
| Age | Numeric | years |
| Sex | Categorical | Male/Female |
| Ulna length | Numeric | cm |

### Output

```text id="gxcdq8"
Estimated standing height (cm)
```

---

# Clinical Interpretation

AnthroPredictor should be considered a **clinical decision-support tool**.

Its outputs represent model-derived estimates and should not be interpreted as directly measured anthropometric values.

When reliable direct measurement is feasible:

> **Direct measurement should generally be preferred over model prediction.**

AnthroPredictor may be useful when direct measurement is unavailable or impractical and an anthropometric estimate is needed to support clinical assessment.

Predictions should be interpreted alongside:

- the patient's clinical condition;
- available anthropometric information;
- recent weight history;
- fluid status;
- body composition;
- measurement quality; and
- professional clinical judgement.

Additional caution may be appropriate when anthropometric measurements are affected by conditions such as substantial oedema, ascites, amputation, severe muscle wasting, anatomical abnormalities, or unusual body proportions.

---

# Limitations

## Population specificity

The current models originate from data collected among adults at a **single tertiary hospital in Ghana**.

The weight model was developed from **389 adults**, while the source dataset for the height model involved **384 adults**.

Performance in substantially different populations cannot be assumed to be equivalent.

Further validation is needed across:

- other Ghanaian populations;
- other healthcare facilities;
- other African populations;
- different age distributions;
- different clinical populations; and
- populations outside Ghana.

---

## Prediction is not measurement

AnthroPredictor generates statistical estimates.

For example, an output of:

```text id="00e2yj"
65.2 kg
```

means that the model has estimated body weight as **65.2 kg**.

It does not mean that 65.2 kg has been directly measured.

Prediction error remains possible even when all inputs have been entered correctly.

---

## Input measurement error

Model predictions depend on the quality of the input data.

Measurement error in:

- standing height;
- MUAC;
- calf circumference; or
- ulna length

may propagate into the resulting prediction.

Standardised anthropometric measurement procedures should therefore be used.

---

## Altered body composition and anthropometry

Model performance may differ in people with substantial alterations in anthropometry or body composition.

Examples include:

- severe oedema;
- ascites;
- limb amputation;
- severe muscle wasting;
- extreme body size; and
- anatomical abnormalities affecting measurement sites.

These populations require specific evaluation before equivalent model performance can be assumed.

---

## Generalisability

Good performance within a model-development or test dataset does not establish performance in all future populations.

Independent external and prospective validation is therefore important before widespread clinical implementation.

---

# Privacy and Data Handling

AnthroPredictor requires only the demographic and anthropometric information necessary to generate predictions.

The prediction models themselves do not require patient names or other direct identifiers.

Clinical deployments should follow the principle of **data minimisation** and avoid collecting identifiable patient information unless it is necessary for an appropriately governed implementation.

Production deployments should consider:

- secure HTTPS communication;
- appropriate access controls;
- secure hosting;
- data minimisation;
- logging practices;
- institutional information-governance policies; and
- applicable data-protection requirements.

Future API logging should avoid storing identifiable patient information unless this is explicitly required and appropriately governed.

---

# Testing and Validation

Software validation and prediction-model validation are related but distinct processes.

## Software testing

Planned automated testing includes:

- model-loading tests;
- input validation;
- valid prediction requests;
- invalid prediction requests;
- prediction-output formatting;
- reproducibility checks;
- API health checks;
- API schema validation; and
- comparison of API predictions with direct model predictions.

---

## Model validation

Further evaluation of the prediction models should consider:

- external validation;
- prospective validation;
- subgroup performance;
- prediction-error distributions;
- clinically relevant error thresholds;
- calibration where appropriate; and
- evaluation in patients in whom routine weight or height measurement is difficult.

---

# Development Roadmap

AnthroPredictor is under active development.

## Application

- [x] Adult weight estimation
- [x] Adult height estimation
- [x] H2O AutoML weight-model integration
- [x] Huber regression height-model integration
- [x] Shiny Express interface
- [x] Responsive desktop/mobile design
- [x] Material 3-inspired interface
- [x] About page
- [x] Model limitations and appropriate-use information
- [x] Clinical decision-support notice
- [x] MIT open-source license

## Model API

- [ ] FastAPI model-serving layer
- [ ] Versioned API structure
- [ ] `/api/v1/predict/weight`
- [ ] `/api/v1/predict/height`
- [ ] `/health`
- [ ] Pydantic request/response schemas
- [ ] Server-side model validation
- [ ] Model-version metadata
- [ ] Structured API error responses
- [ ] Automated API tests
- [ ] Non-identifiable structured logging

## Deployment and software engineering

- [ ] Environment-based configuration
- [ ] Dockerisation
- [ ] Production deployment configuration
- [ ] HTTPS
- [ ] Continuous integration
- [ ] Automated tests on pull requests
- [ ] Release versioning

## Research and validation

- [ ] Complete documentation of height-model performance
- [ ] External validation of the weight model
- [ ] External validation of the height model
- [ ] Subgroup performance assessment
- [ ] Prospective clinical evaluation
- [ ] Evaluation in patients unable to undergo routine anthropometric measurement

---

# Citation

## Weight Model

The weight-estimation model is associated with the following research:

**Adult weight estimation in a tertiary hospital in Ghana**

Proceedings of Machine Learning Research, Volume 302.

https://proceedings.mlr.press/v302/anku26a.html

Please consult the published paper for the complete bibliographic citation.

---

## Height Model

The height model uses anthropometric data originating from research involving **384 adults at Cape Coast Teaching Hospital**, including measured standing height and ulna length.

The current Huber regression model represents subsequent model-development work using these data.

The definitive citation describing the final height prediction model should be added when that work is formally published or otherwise made publicly available.

---

## Citing the Software

If AnthroPredictor is used in academic work, users are encouraged to cite:

1. the relevant model-development research; and
2. the AnthroPredictor software/repository and version used for the analysis.

A `CITATION.cff` file may be provided in a future repository update to support standardised software citation through GitHub.

---

# Contributing

Contributions that improve the scientific validity, software quality, documentation, accessibility, security, or clinical usability of AnthroPredictor are welcome.

A typical contribution workflow is:

```bash id="e8vg4r"
git checkout -b feature/my-feature
```

Make and test the required changes.

Then:

```bash id="b4bmbc"
git add .
git commit -m "Add description of change"
```

Push the branch:

```bash id="dnxqve"
git push -u origin feature/my-feature
```

Then open a pull request.

## Changes affecting prediction models

Changes affecting prediction logic or model artefacts should clearly document:

1. the reason for the change;
2. the model or preprocessing step affected;
3. the scientific or technical evidence supporting the change;
4. whether model inputs or outputs have changed;
5. validation performed;
6. performance before and after the change, where applicable; and
7. implications for backward compatibility.

Model artefacts should not be replaced without corresponding documentation and validation.

---

# License

AnthroPredictor is licensed under the **MIT License**.

You are free to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the software subject to the terms of the MIT License.

See the [`LICENSE`](LICENSE) file in this repository for the complete license terms.

## Research and model considerations

The MIT License governs use and distribution of the software contained in this repository.

Users of AnthroPredictor in academic or research work should appropriately cite the underlying research and the software version used.

Open-source availability does not establish clinical validity in a new population or healthcare setting. Independent validation and appropriate institutional, ethical, regulatory, and clinical governance should be considered before deployment in clinical practice.

---

# Disclaimer

AnthroPredictor is currently a **research and clinical decision-support application**.

It is not intended to replace direct anthropometric measurement, professional clinical assessment, or clinical judgement.

Predictions generated by AnthroPredictor:

- are statistical estimates rather than direct measurements;
- may contain clinically meaningful prediction error;
- may not generalise to populations different from those used for model development;
- depend on the quality of the anthropometric inputs; and
- should not be used as the sole basis for diagnosis, treatment, medication dosing, nutritional prescription, or other clinical decisions.

Users are responsible for determining whether the application is appropriate for their clinical, research, institutional, and regulatory context.

---

# Acknowledgements

AnthroPredictor builds on anthropometric research conducted at **Cape Coast Teaching Hospital, Ghana**.

The contributions of the study participants, research collaborators, clinical staff, and others involved in the underlying studies are gratefully acknowledged.

The project also makes use of open-source technologies including Python, Shiny for Python, H2O.ai, pandas, scikit-learn, and related software libraries.

---

## Project Status

**Active development**

The current release provides local weight and height prediction through a Shiny Express application. API-based model serving, expanded testing, deployment infrastructure, and further model validation are planned.

---

**AnthroPredictor**

*Clinical anthropometric estimation to support decision-making when direct measurement is unavailable or impractical.*


