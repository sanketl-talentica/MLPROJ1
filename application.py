import joblib
import numpy as np
from config.paths_config import MODEL_OUTPUT_PATH
from flask import Flask, render_template, request

app = Flask(__name__)

loaded_model = joblib.load(MODEL_OUTPUT_PATH)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':

        social_media_hours = float(request.form["social_media_hours"])
        daily_screen_time_hours = float(request.form["daily_screen_time_hours"])
        weekend_screen_time = float(request.form["weekend_screen_time"])
        work_study_hours = float(request.form["work_study_hours"])
        sleep_hours = float(request.form["sleep_hours"])
        notifications_per_day = int(request.form["notifications_per_day"])
        gaming_hours = float(request.form["gaming_hours"])
        app_opens_per_day = int(request.form["app_opens_per_day"])
        age = int(request.form["age"])
        academic_work_impact = int(request.form["academic_work_impact"])

        features = np.array([[social_media_hours, daily_screen_time_hours,
                               weekend_screen_time, work_study_hours, sleep_hours,
                               notifications_per_day, gaming_hours, app_opens_per_day,
                               age, academic_work_impact]])

        prediction = loaded_model.predict(features)

        return render_template('index.html', prediction=prediction[0])

    return render_template("index.html", prediction=None)

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8080)
