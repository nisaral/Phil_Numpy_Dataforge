from flask import Flask, request, jsonify, send_file, render_template
from flask_cors import CORS
import os
import tempfile
import uuid
import threading
import time
from summarize import VideoSummarizer
import json

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Configure upload folder
UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'output'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Store API keys
GEMINI_API_KEY = "AIzaSyBfIWKs0ffmQGdx7suNBUBgM4C_0Ss3LMs"
ELEVENLABS_API_KEY = "sk_caf3c55a9cd7d5211cbde758b98c15241c13836e90328bf3"

# Store processing jobs
processing_jobs = {}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/process-video', methods=['POST'])
def process_video():
    if 'video' not in request.files:
        return jsonify({'error': 'No video file provided'}), 400
    
    video_file = request.files['video']
    if video_file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    # Generate unique ID for this processing job
    job_id = str(uuid.uuid4())
    
    # Save uploaded video to temporary file
    video_path = os.path.join(UPLOAD_FOLDER, f"{job_id}_{video_file.filename}")
    video_file.save(video_path)
    
    # Initialize job status
    processing_jobs[job_id] = {
        'status': 'processing',
        'video_path': video_path,
        'output_dir': os.path.join(OUTPUT_FOLDER, job_id),
        'start_time': time.time()
    }
    
    # Start processing in a separate thread
    thread = threading.Thread(target=process_video_thread, args=(job_id,))
    thread.daemon = True
    thread.start()
    
    return jsonify({
        'job_id': job_id,
        'status': 'processing',
        'message': 'Video processing started'
    })

def process_video_thread(job_id):
    try:
        job = processing_jobs[job_id]
        video_path = job['video_path']
        output_dir = job['output_dir']
        
        # Initialize video summarizer
        summarizer = VideoSummarizer(video_path, GEMINI_API_KEY, model_provider="gemini")
        
        # Process video
        output_files = summarizer.process_video(output_dir=output_dir)
        
        # Update job status
        processing_jobs[job_id]['status'] = 'complete'
        processing_jobs[job_id]['output_files'] = output_files
        
    except Exception as e:
        processing_jobs[job_id]['status'] = 'error'
        processing_jobs[job_id]['error'] = str(e)
        print(f"Error processing video: {e}")

@app.route('/api/status/<job_id>', methods=['GET'])
def get_status(job_id):
    if job_id not in processing_jobs:
        return jsonify({'error': 'Job not found'}), 404
    
    job = processing_jobs[job_id]
    
    if job['status'] == 'error':
        return jsonify({
            'status': 'error',
            'error': job.get('error', 'Unknown error')
        })
    
    return jsonify({
        'status': job['status'],
        'elapsed_time': time.time() - job['start_time']
    })

@app.route('/api/results/<job_id>', methods=['GET'])
def get_results(job_id):
    if job_id not in processing_jobs:
        return jsonify({'error': 'Job not found'}), 404
    
    job = processing_jobs[job_id]
    
    if job['status'] != 'complete':
        return jsonify({'error': 'Results not ready yet'}), 400
    
    try:
        # Read summary text
        summary_path = os.path.join(job['output_dir'], "summary.txt")
        with open(summary_path, 'r') as f:
            summary_text = f.read()
        
        # Return results
        return jsonify({
            'job_id': job_id,
            'summary': summary_text,
            'storyboard_image': f"/api/download/{job_id}/storyboard.png",
            'storyboard_pdf': f"/api/download/{job_id}/storyboard.pdf",
            'summary_video': f"/api/download/{job_id}/summary.mp4",
            'summary_video_orig_audio': f"/api/download/{job_id}/summary_orig_audio.mp4"
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/download/<job_id>/<file_type>', methods=['GET'])
def download_file(job_id, file_type):
    if job_id not in processing_jobs:
        return jsonify({'error': 'Job not found'}), 404
    
    job = processing_jobs[job_id]
    
    if job['status'] != 'complete':
        return jsonify({'error': 'Results not ready yet'}), 400
    
    file_path = os.path.join(job['output_dir'], f"{file_type}")
    
    if not os.path.exists(file_path):
        return jsonify({'error': 'File not found'}), 404
    
    return send_file(file_path)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000) 