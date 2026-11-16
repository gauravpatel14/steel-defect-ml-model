STEEL DEFECT STREAMLIT APP

Keep these files together:
- app.py
- requirements.txt
- steel_defect_model.pkl 
- model_metrics.json 

Run in VS Code terminal:
python -m pip install -r requirements.txt
python -m streamlit run app.py

Prediction uses a CSV with the 27 feature columns. One row = single prediction; multiple rows = batch.
The app provides a downloadable CSV template.

Deployment:
Push the app files to GitHub. Visit https://share.streamlit.io/, sign in with GitHub,
click Create app
