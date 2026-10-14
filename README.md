# AI Student Performance Predictor

A beginner project that predicts a **student exam score** from study habits.

It uses:

- **Python** to run the code
- **pandas** to read the student table
- **scikit-learn** to train a real machine learning model
- **Streamlit** to show a simple website

This is a **learning demo**. The dataset is fictional. Do not use it to grade real students.

## What each file does

| File or folder | Purpose |
| --- | --- |
| `app.py` | The website. Type in hours, attendance, and other details, then see a predicted score. |
| `train_model.py` | Trains Linear Regression, checks it on a hidden test set, and saves the model. |
| `data/students.csv` | Sample data: 200 made-up students. |
| `models/student_model.pkl` | The saved trained model (created after you run `train_model.py`). |
| `models/metrics.json` | Honest train/test scores (created after training). |
| `requirements.txt` | The Python packages this project needs. |
| `.gitignore` | Files Git should ignore if you add version control later. |

## Dataset (learning data only)

File: `data/students.csv`

Each row is one fictional student. Exam scores were generated from study habits **plus random noise**, so the model cannot be perfect. That is realistic.

| Column | Type | Meaning | Typical range |
| --- | --- | --- | --- |
| `student_id` | text | Fake ID. Not used by the model. | S001 … |
| `hours_studied` | number | Study hours per day | 0.5–12 |
| `attendance_percent` | number | Class attendance | 50–100 |
| `previous_score` | number | Earlier exam score | 30–100 |
| `sleep_hours` | number | Sleep per night | 4–10 |
| `assignments_completed` | number | Assignments finished out of 10 | 0–10 |
| `extra_curricular` | text | Extra activities | Yes / No |
| `parental_support` | text | Home support | Low / Medium / High |
| `exam_score` | number | **Target** the model predicts | 0–100 |

## How the model is trained

1. Load `data/students.csv`.
2. Split rows with `train_test_split`: **80% train**, **20% test**, `random_state=42`.
3. Scale numeric columns and one-hot encode text columns.
4. Fit **Linear Regression**.
5. Report **MAE**, **RMSE**, and **R²** on the **test set** (unseen students).
6. Also print training metrics so you can compare, but the test numbers are the honest result.

## Setup

You need Python 3 and pip.

Open a terminal in this folder (`ai-student-performance-predictor`) and install packages:

```bash
python -m pip install -r requirements.txt
```

If `python` does not work, try:

```bash
py -m pip install -r requirements.txt
```

## Train the model

```bash
python train_model.py
```

You should see test MAE, RMSE, and R² printed in the terminal. Two files should appear:

- `models/student_model.pkl`
- `models/metrics.json`

## Launch the app

```bash
streamlit run app.py
```

Your browser should open a local page (often http://localhost:8501).

If the browser does not open, copy the URL from the terminal.

## If something goes wrong

| Problem | What to try |
| --- | --- |
| `python` is not recognized | Use `py train_model.py` and `py -m streamlit run app.py` |
| Dataset not found | Confirm `data/students.csv` is in this project folder |
| No trained model | Run `python train_model.py` first, then start Streamlit again |
| Package missing | Run `python -m pip install -r requirements.txt` |
| App already using port 8501 | Streamlit will offer another port, or close the old terminal |

## Project status

Train and launch on your computer. This README does not claim the live app is running until you start Streamlit.
