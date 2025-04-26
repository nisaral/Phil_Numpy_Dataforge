# Video Summarization API

This application provides a REST API for video summarization using AI-powered analysis and processing. It includes a modern web interface for uploading videos and viewing results.

## Features

- Video upload and processing
- AI-powered content analysis
- Text and speech extraction
- Dynamic video summarization
- Storyboard generation
- High-quality voice synthesis
- Modern web interface
- Asynchronous processing with status updates

## Prerequisites

- Python 3.8 or higher
- FFmpeg installed on your system
- Tesseract OCR installed on your system
- Sufficient disk space for video processing

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd <repository-directory>
```

2. Install Python dependencies:
```bash
pip install -r requirements.txt
```

3. Install system dependencies:

For macOS:
```bash
brew install ffmpeg tesseract
```

For Ubuntu/Debian:
```bash
sudo apt-get update
sudo apt-get install ffmpeg tesseract-ocr
```

## Configuration

1. Create a `.env` file in the root directory with your API keys:
```
GEMINI_API_KEY=your_gemini_api_key
ELEVENLABS_API_KEY=your_elevenlabs_api_key
```

## Usage

1. Start the Flask server:
```bash
python run.py
```

2. Open your web browser and navigate to:
```
http://localhost:5000
```

3. Upload a video file through the web interface
4. Wait for processing to complete (you'll see a progress indicator)
5. View and download the results:
   - Text summary
   - Storyboard PDF
   - Summary video

## API Endpoints

- `POST /api/process-video`: Upload and process a video file
- `GET /api/status/<job_id>`: Check processing status
- `GET /api/results/<job_id>`: Get processing results
- `GET /api/download/<job_id>/<file_type>`: Download processed files

## Project Structure

```
.
├── app.py              # Flask application
├── run.py              # Application runner
├── summarize.py        # Video processing logic
├── templates/          # HTML templates
│   └── index.html     # Web interface
├── static/             # Static assets
├── uploads/            # Temporary upload directory
├── output/             # Processed output directory
├── requirements.txt    # Python dependencies
└── README.md          # Documentation
```

## How It Works

1. The user uploads a video through the web interface
2. The video is processed asynchronously in a background thread
3. The frontend periodically checks the processing status
4. When processing is complete, the results are displayed to the user
5. The user can download the summary, storyboard, and summary video

## License

MIT License

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request 