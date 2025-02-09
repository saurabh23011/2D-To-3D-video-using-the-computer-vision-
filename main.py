from flask import Flask, request, render_template, redirect, url_for, send_from_directory
import os
from werkzeug.utils import secure_filename
from video_processing import process_video

app = Flask(__name__)

# Configuration
app.config['UPLOAD_FOLDER'] = 'static/uploads/input_videos/'
app.config['OUTPUT_FOLDER'] = 'static/uploads/output_videos/'
app.config['DEPTH_FOLDER'] = 'static/uploads/depth_maps/'
app.config['POINTCLOUD_FOLDER'] = 'static/uploads/pointclouds/'
app.config['ALLOWED_EXTENSIONS'] = {'mp4', 'avi', 'mov', 'mkv'}
app.config['MAX_CONTENT_LENGTH'] = 2 * 1024 * 1024 * 1024  # 2 GB max file size

# Ensure the upload folders exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)
os.makedirs(app.config['DEPTH_FOLDER'], exist_ok=True)
os.makedirs(app.config['POINTCLOUD_FOLDER'], exist_ok=True)

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/', methods=['POST'])
def upload_video():
    # Check if the POST request has the file part
    if 'video_file' not in request.files:
        return 'No file part in the request', 400

    file = request.files['video_file']

    # If the user does not select a file
    if file.filename == '':
        return 'No selected file', 400

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        input_filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(input_filepath)

        # Process the video
        output_filename = 'output_' + filename
        output_filepath = os.path.join(app.config['OUTPUT_FOLDER'], output_filename)
        depth_output_folder = app.config['DEPTH_FOLDER']
        pointcloud_output_folder = app.config['POINTCLOUD_FOLDER']

        process_video(input_filepath, output_filepath, depth_output_folder, pointcloud_output_folder)

        # After processing, redirect to the 3D viewer
        # Here we assume point clouds are named as 'pointcloud_00000.ply', 'pointcloud_00001.ply', etc.
        first_pointcloud_filename = 'pointcloud_00000.ply'

        return redirect(url_for('viewer', filename=first_pointcloud_filename))

    else:
        return 'Invalid file type', 400

@app.route('/viewer/<filename>')
def viewer(filename):
    return render_template('3d_viewer.html', filename=filename)

@app.route('/download/<filename>')
def download_file(filename):
    return send_from_directory(app.config['OUTPUT_FOLDER'], filename, as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)