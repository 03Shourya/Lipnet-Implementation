from flask import Flask, render_template, request, redirect, url_for
import os
from werkzeug.utils import secure_filename
from lipnet_model import run_lipnet_model

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'

if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'video' not in request.files:
            return redirect(request.url)
        
        video = request.files['video']
        if video.filename == '':
            return redirect(request.url)

        filename = secure_filename(video.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        video.save(filepath)

        # Call your LipNet function with the file path
        actual, predicted, accuracy = run_lipnet_model(filepath)

        return render_template('index.html',
                               video_url=filepath,
                               actual=actual,
                               predicted=predicted,
                               accuracy=accuracy*100)

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
