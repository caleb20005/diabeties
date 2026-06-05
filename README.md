# Diabetes Prediction App

This project is a Streamlit app that predicts diabetes risk from basic clinical measurements using a trained machine-learning model.

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy on Streamlit Community Cloud

1. Push this folder to a GitHub repository.
2. Go to Streamlit Community Cloud and create a new app from that repo.
3. Set the main file path to `app.py`.
4. Let Streamlit install the packages listed in `requirements.txt`.

## Notes

- The app downloads a public diabetes dataset at runtime.
- The prediction is a screening estimate, not a medical diagnosis.
