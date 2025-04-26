import os
import uuid
import cv2
import numpy as np
from flask import Flask, request, render_template, jsonify, send_file
from flask_cors import CORS
from werkzeug.utils import secure_filename
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from bs4 import BeautifulSoup
from youtube_transcript_api import YouTubeTranscriptApi
from sentence_transformers import SentenceTransformer
import faiss
from PIL import Image, ImageDraw, ImageFont
import google.generativeai as genai
import json
from graphviz import Digraph
from ultralytics import YOLO
import yt_dlp
from deep_translator import GoogleTranslator
from concurrent.futures import ThreadPoolExecutor
import subprocess
from typing import List
from crawler import WebCrawler
from extraction_strategy import JsonCssExtractionStrategy

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['RESULTS_FOLDER'] = 'static/results'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['RESULTS_FOLDER'], exist_ok=True)

# Load environment
from dotenv import load_dotenv
load_dotenv()
genai.configure(api_key=os.getenv('GEMINI_API_KEY'))

# Configurations
SUPPORTED_LANGUAGES = {
    'en': {'name': 'English', 'font': 'arial.ttf'},
    'es': {'name': 'Spanish', 'font': 'arial.ttf'},
    'fr': {'name': 'French', 'font': 'arial.ttf'},
    'zh-cn': {'name': 'Chinese', 'font': 'simhei.ttf'},
}
DOMAIN_DESCRIPTORS = {
    'sports': {'focus': ['players', 'action'], 'prompt': 'Analyze sports video for player actions.', 'colors': {'primary': (0, 128, 255), 'text': (0, 0, 0)}},
    'education': {'focus': ['instructor', 'concepts'], 'prompt': 'Analyze educational video for key concepts.', 'colors': {'primary': (76, 175, 80), 'text': (33, 33, 33)}},
    'generic': {'focus': ['scene', 'objects'], 'prompt': 'Analyze video for main elements.', 'colors': {'primary': (0, 150, 136), 'text': (0, 0, 0)}}
}

# Initialize models
embedder = SentenceTransformer('all-MiniLM-L6-v2')
index = faiss.IndexFlatL2(384)
knowledge_base = []
yolo_model = YOLO("yolov11n.pt") if os.path.exists("yolov11n.pt") else None

# Initialize Crawler for YouTube
crawler = WebCrawler()
crawler.warmup()
extraction_strategy = JsonCssExtractionStrategy(
    schema={"text_content": {"css": "p", "type": "string"}}
)

def translate_text(text, lang):
    try:
        return GoogleTranslator(source='auto', target=lang).translate(text) if lang != 'en' else text
    except:
        return text

def get_font(lang, size=20):
    try:
        return ImageFont.truetype(SUPPORTED_LANGUAGES.get(lang, {}).get('font', 'arial.ttf'), size)
    except:
        return ImageFont.truetype('arial.ttf', size)

def crawl_url(url):
    try:
        options = Options()
        options.add_argument("--headless")
        driver = webdriver.Chrome(options=options)
        driver.get(url)
        soup = BeautifulSoup(driver.page_source, 'html.parser')
        text = " ".join(p.get_text(strip=True) for p in soup.find_all('p') if p.get_text(strip=True))
        driver.quit()
        return text
    except Exception as e:
        print(f"Crawl error: {e}")
        return None

def crawl_youtube(video_url: str) -> str:
    try:
        result = crawler.run(url=video_url, extraction_strategy=extraction_strategy, bypass_cache=True)
        if result.success and result.extracted_content:
            return " ".join([item["text_content"] for item in result.extracted_content if "text_content" in item])
        return None
    except Exception as e:
        print(f"Error crawling YouTube {video_url}: {str(e)}")
        return None

def get_youtube_transcript(video_url: str) -> str:
    try:
        # Extract video ID from various YouTube URL formats
        if 'youtube.com/watch?v=' in video_url:
            video_id = video_url.split('watch?v=')[1].split('&')[0]
        elif 'youtu.be/' in video_url:
            video_id = video_url.split('youtu.be/')[1].split('?')[0]
        else:
            return None
            
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
        return " ".join([entry["text"] for entry in transcript])
    except Exception as e:
        print(f"Error fetching transcript for {video_url}: {e}")
        return None

def chunk_text(text: str, max_length: int = 500) -> List[str]:
    words = text.split()
    return [" ".join(words[i:i+max_length]) for i in range(0, len(words), max_length)]

def update_knowledge_base(text: str, source: str, source_type: str):
    if not text:
        return
    chunks = chunk_text(text)
    embeddings = embedder.encode(chunks, convert_to_numpy=True)
    index.add(embeddings)
    for chunk in chunks:
        knowledge_base.append({"text": chunk, "source": source, "type": source_type})
    vector_base.extend(embeddings)
    print(f"Added {len(chunks)} chunks from {source} ({source_type}) to knowledge base.")

def download_youtube_video(url, output_dir="temp_videos"):
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"video_{uuid.uuid4()}.mp4")
    
    try:
        ydl_opts = {
            'format': 'best[ext=mp4]',
            'outtmpl': output_path,
            'quiet': True,
            'no_warnings': True,
            'extract_flat': False,
            'noplaylist': True,
            'ignoreerrors': True
        }
        
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
            
        if os.path.exists(output_path):
            return output_path
        return None
    except Exception as e:
        print(f"YouTube download error: {str(e)}")
        return None

def extract_keyframes(video_path, max_frames=10):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return [], [], 0, 0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    duration = frame_count / fps
    max_frames = min(max_frames, frame_count)
    sample_rate = max(1, frame_count // 1000)
    prev_frame, scores, frame_idx = None, [], 0

    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = []
        while frame_idx < frame_count:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx % sample_rate == 0 and prev_frame is not None and frame.shape == prev_frame.shape:
                futures.append(executor.submit(lambda f, i: (i, np.mean((f.astype(float) - prev_frame.astype(float)) ** 2), f.copy()), frame, frame_idx))
            prev_frame = frame
            frame_idx += 1

        for future in futures:
            scores.append(future.result())
    cap.release()

    scores.sort(key=lambda x: x[1], reverse=True)
    keyframes = [(s[0], s[2]) for s in scores[:max_frames]]
    key_moments = [(s[0], s[2]) for s in scores[:min(5, len(scores))]]
    return sorted(keyframes, key=lambda x: x[0]), sorted(key_moments, key=lambda x: x[0]), fps, duration

def generate_text_summary(frame, frame_idx, fps, domain, lang='en'):
    try:
        # Convert frame to RGB for YOLO
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Detect objects
        objects = []
        if yolo_model:
            results = yolo_model(frame_rgb, verbose=False)
            objects = [yolo_model.names[int(box.cls)] for result in results for box in result.boxes]
        
        # Create prompt for Gemini
        prompt = f"""
        Analyze this video frame at timestamp {frame_idx/fps:.2f}s.
        Objects detected: {', '.join(objects) if objects else 'none'}
        
        Provide a clear, concise description of what's happening in this frame.
        Focus on:
        1. Main subjects/objects
        2. Actions or movements
        3. Important details
        4. Scene context
        
        Keep the description informative but brief (1-2 sentences).
        """
        
        model = genai.GenerativeModel('gemini-1.5-flash')
        response = model.generate_content(prompt)
        summary = response.text.strip()
        
        # Translate if needed
        if lang != 'en':
            summary = translate_text(summary, lang)
            
        return summary
    except Exception as e:
        print(f"Error generating summary: {str(e)}")
        return f"Scene at {frame_idx/fps:.2f}s"

def create_storyboard_image(keyframes, key_moments, summaries, output_dir, domain, video_input, duration, fps, lang='en'):
    try:
        os.makedirs(output_dir, exist_ok=True)
        
        # Calculate grid layout
        num_images = len(keyframes)
        cols = min(3, num_images)
        rows = (num_images + cols - 1) // cols
        
        # Calculate image dimensions
        cell_width, cell_height = 300, 200
        margin = 20
        total_width = cols * (cell_width + margin) + margin
        total_height = rows * (cell_height + 100 + margin) + 100
        
        # Create storyboard image
        storyboard = Image.new('RGB', (total_width, total_height), (255, 255, 255))
        draw = ImageDraw.Draw(storyboard)
        
        # Add title
        title = f"Storyboard - {os.path.basename(video_input)[:50]}"
        draw.text((margin, margin), title, fill=(0, 0, 0), font=ImageFont.truetype("arial.ttf", 24))
        
        # Add frames
        for i, (frame_idx, frame) in enumerate(keyframes):
            # Convert frame to PIL Image
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_pil = Image.fromarray(frame_rgb)
            
            # Resize frame
            frame_pil = frame_pil.resize((cell_width, cell_height))
            
            # Calculate position
            x = margin + (i % cols) * (cell_width + margin)
            y = margin + 50 + (i // cols) * (cell_height + 100 + margin)
            
            # Paste frame
            storyboard.paste(frame_pil, (x, y))
            
            # Add timestamp and summary
            timestamp = f"Time: {frame_idx/fps:.1f}s"
            draw.text((x, y + cell_height + 10), timestamp, fill=(0, 0, 0), font=ImageFont.truetype("arial.ttf", 12))
            
            # Add summary (truncate if too long)
            summary = summaries[i][:100] + "..." if len(summaries[i]) > 100 else summaries[i]
            draw.text((x, y + cell_height + 30), summary, fill=(0, 0, 0), font=ImageFont.truetype("arial.ttf", 12))
            
            # Save individual keyframe
            keyframe_path = os.path.join(output_dir, f"keyframe_{i+1}.jpg")
            frame_pil.save(keyframe_path)
        
        # Save storyboard
        storyboard_path = os.path.join(output_dir, "storyboard.jpg")
        storyboard.save(storyboard_path)
        
        return storyboard_path
    except Exception as e:
        print(f"Error creating storyboard: {str(e)}")
        return None

def process_video(video_input, domain, output_dir, lang='en', enhanced=False):
    os.makedirs(output_dir, exist_ok=True)
    video_path = video_input
    
    # Handle YouTube URL
    if video_input.startswith(('http://', 'https://')):
        video_path = download_youtube_video(video_input)
        if not video_path:
            return None, "Failed to download YouTube video"
    
    if not os.path.exists(video_path):
        return None, "Video file not found"
    
    # Extract keyframes and process video
    keyframes, key_moments, fps, duration = extract_keyframes(video_path)
    if not keyframes:
        return None, "Failed to extract keyframes"
    
    # Generate summaries and save keyframes
    summaries = []
    highlight_frames = []
    for idx, frame in keyframes:
        summary = generate_text_summary(frame, idx, fps, domain, lang)
        output_path = os.path.join(output_dir, f"keyframe_{len(highlight_frames)+1}.jpg")
        cv2.imwrite(output_path, frame)
        summaries.append(summary)
        highlight_frames.append(output_path)
    
    # Create storyboard
    storyboard_path = create_storyboard_image(
        keyframes, key_moments, summaries, 
        output_dir, domain, video_input, 
        duration, fps, lang
    )
    
    # Generate overall summary
    summary = " ".join(summaries)
    
    return highlight_frames, summary

@app.route('/')
def index():
    return render_template('index1.html', domains=DOMAIN_DESCRIPTORS.keys(), languages=SUPPORTED_LANGUAGES.keys(), supported_languages=SUPPORTED_LANGUAGES)

@app.route('/add_web', methods=['POST'])
def add_web():
    task_id = str(uuid.uuid4())
    output_dir = os.path.join(app.config['RESULTS_FOLDER'], task_id)
    os.makedirs(output_dir, exist_ok=True)
    data = request.get_json()
    url, lang = data.get('web_url'), data.get('lang', 'en')
    if not url:
        return jsonify({'error': 'Invalid URL'}), 400
    text = crawl_url(url)
    if text:
        update_knowledge_base(text, url, "web")
        return jsonify({'task_id': task_id, 'content': text})
    return jsonify({'error': 'Failed to crawl'}), 500

@app.route('/add_youtube', methods=['POST'])
def add_youtube():
    task_id = str(uuid.uuid4())
    output_dir = os.path.join(app.config['RESULTS_FOLDER'], task_id)
    os.makedirs(output_dir, exist_ok=True)
    
    data = request.get_json()
    url = data.get('youtube_url')
    lang = data.get('lang', 'en')
    
    if not url:
        return jsonify({'error': 'Invalid URL'}), 400
    
    try:
        # First download the video
        video_path = download_youtube_video(url)
        if not video_path:
            return jsonify({'error': 'Failed to download YouTube video'}), 500
            
        # Extract keyframes and process video
        keyframes, key_moments, fps, duration = extract_keyframes(video_path)
        if not keyframes:
            return jsonify({'error': 'Failed to extract keyframes'}), 500
            
        # Generate summaries for each keyframe
        summaries = []
        for idx, frame in keyframes:
            summary = generate_text_summary(frame, idx, fps, 'generic', lang)
            summaries.append(summary)
            
        # Create storyboard
        storyboard_path = create_storyboard_image(
            keyframes, key_moments, summaries, 
            output_dir, 'generic', url, 
            duration, fps, lang
        )
        
        # Get video description and transcript
        description = crawl_youtube(url) or ""
        transcript = get_youtube_transcript(url) or ""
        
        # Combine all content
        content = f"""
Video Summary:
{chr(10).join(summaries)}

Video Description:
{description}

Transcript:
{transcript}
"""
        
        # Update knowledge base
        update_knowledge_base(content, url, "video")
        
        # Create manifest
        manifest = {
            'keyframes': [{'index': i, 'frame_idx': kf[0], 'timestamp': kf[0]/fps, 'summary': s, 'path': f'/results/{task_id}/keyframe_{i+1}.jpg', 'is_key_moment': kf[0] in [km[0] for km in key_moments]} for i, (kf, s) in enumerate(zip(keyframes, summaries))],
            'key_moments': [{'frame_idx': km[0], 'timestamp': km[0]/fps} for km in key_moments],
            'storyboard': f'/results/{task_id}/storyboard.jpg',
            'description': description,
            'transcript': transcript,
            'duration': duration,
            'fps': fps
        }
        
        return jsonify({
            'task_id': task_id,
            'content': content,
            'manifest': json.dumps(manifest)
        })
        
    except Exception as e:
        print(f"YouTube processing error: {str(e)}")
        return jsonify({'error': f'Failed to process YouTube content: {str(e)}'}), 500

@app.route('/add_text', methods=['POST'])
def add_text():
    task_id = str(uuid.uuid4())
    output_dir = os.path.join(app.config['RESULTS_FOLDER'], task_id)
    os.makedirs(output_dir, exist_ok=True)
    data = request.get_json()
    text, lang = data.get('text'), data.get('lang', 'en')
    if not text:
        return jsonify({'error': 'No text'}), 400
    update_knowledge_base(text, "user_input", "text")
    return jsonify({'task_id': task_id, 'content': text})

@app.route('/ask_question', methods=['POST'])
def ask_question():
    data = request.get_json()
    question, lang = data.get('question'), data.get('lang', 'en')
    if not question:
        return jsonify({'error': 'No question'}), 400
    embeddings = embedder.encode([question], convert_to_numpy=True)
    _, indices = index.search(embeddings, k=5)
    context = "\n".join(knowledge_base[i]['text'] for i in indices[0] if i < len(knowledge_base))
    try:
        model = genai.GenerativeModel('gemini-1.5-flash')
        answer = model.generate_content(f"Context: {context}\nQuestion: {question}").text.strip()
        return jsonify({'answer': translate_text(answer, lang)})
    except:
        return jsonify({'error': 'Failed to answer'}), 500

@app.route('/get_summary', methods=['POST'])
def get_summary():
    data = request.get_json()
    text, lang = data.get('text'), data.get('lang', 'en')
    if not text:
        return jsonify({'error': 'No text'}), 400
    try:
        model = genai.GenerativeModel('gemini-1.5-flash')
        summary = model.generate_content(f"Summarize in 2-3 sentences:\n{text[:2000]}").text.strip()
        return jsonify({'summary': translate_text(summary, lang)})
    except:
        return jsonify({'error': 'Failed to summarize'}), 500

@app.route('/generate_mock_test', methods=['POST'])
def generate_mock_test():
    task_id = str(uuid.uuid4())
    output_dir = os.path.join(app.config['RESULTS_FOLDER'], task_id)
    os.makedirs(output_dir, exist_ok=True)
    data = request.get_json()
    text, domain, lang = data.get('text'), data.get('domain', 'generic'), data.get('lang', 'en')
    if not text:
        return jsonify({'error': 'No text'}), 400
    try:
        model = genai.GenerativeModel('gemini-1.5-flash')
        test = model.generate_content(f"Generate 5 MCQs from:\n{text[:2000]}").text.strip()
        with open(os.path.join(output_dir, 'mock_test.txt'), 'w') as f:
            f.write(test)
        return jsonify({'task_id': task_id, 'mock_test': translate_text(test, lang)})
    except:
        return jsonify({'error': 'Failed to generate test'}), 500

@app.route('/generate_mind_map', methods=['POST'])
def generate_mind_map():
    task_id = str(uuid.uuid4())
    output_dir = os.path.join(app.config['RESULTS_FOLDER'], task_id)
    os.makedirs(output_dir, exist_ok=True)
    data = request.get_json()
    text, domain, lang = data.get('text'), data.get('domain', 'generic'), data.get('lang', 'en')
    if not text:
        return jsonify({'error': 'No text'}), 400
    try:
        model = genai.GenerativeModel('gemini-1.5-flash')
        mind_map = model.generate_content(f"Create a mind map from:\n{text[:2000]}").text.strip()
        output_path = os.path.join(output_dir, 'mind_map.txt')
        with open(output_path, 'w') as f:
            f.write(mind_map)
        return jsonify({'task_id': task_id, 'mind_map': output_path})
    except:
        return jsonify({'error': 'Failed to generate mind map'}), 500

@app.route('/generate_flowchart', methods=['POST'])
def generate_flowchart():
    task_id = str(uuid.uuid4())
    output_dir = os.path.join(app.config['RESULTS_FOLDER'], task_id)
    os.makedirs(output_dir, exist_ok=True)
    data = request.get_json()
    text, domain, lang = data.get('text'), data.get('domain', 'generic'), data.get('lang', 'en')
    if not text:
        return jsonify({'error': 'No text'}), 400
    try:
        model = genai.GenerativeModel('gemini-1.5-flash')
        dot_content = model.generate_content(f"Generate DOT flowchart from:\n{text[:2000]}").text.strip()
        dot = Digraph()
        for line in dot_content.split('\n'):
            if '->' in line:
                a, b = line.split('->')
                dot.edge(a.strip(), b.strip())
            elif line.strip():
                dot.node(line.strip())
        output_path = os.path.join(output_dir, 'flowchart.svg')
        dot.render(output_path, format='svg', cleanup=True)
        return jsonify({'task_id': task_id, 'flowchart': output_path})
    except:
        return jsonify({'error': 'Failed to generate flowchart'}), 500

@app.route('/analyze_video_basic', methods=['POST'])
def analyze_video_basic():
    task_id = str(uuid.uuid4())
    output_dir = os.path.join(app.config['RESULTS_FOLDER'], task_id)
    os.makedirs(output_dir, exist_ok=True)
    
    domain = request.form.get('domain', 'generic')
    lang = request.form.get('lang', 'en')
    video_input = None
    
    if 'video' in request.files:
        file = request.files['video']
        if file.filename:
            filename = secure_filename(file.filename)
            video_input = os.path.join(app.config['UPLOAD_FOLDER'], f"{task_id}_{filename}")
            file.save(video_input)
    elif request.form.get('youtube_url'):
        video_input = request.form.get('youtube_url')
    
    if not video_input:
        return jsonify({'error': 'No video or URL provided'}), 400
    
    keyframes, summary = process_video(video_input, domain, output_dir, lang)
    if not keyframes:
        return jsonify({'error': summary}), 500
    
    # Read the manifest file
    manifest_path = os.path.join(output_dir, 'storyboard_manifest.json')
    with open(manifest_path, 'r') as f:
        manifest = json.load(f)
    
    # Update paths in manifest to be relative to the results directory
    for kf in manifest['keyframes']:
        kf['path'] = f"/results/{task_id}/{os.path.basename(kf['path'])}"
    
    return jsonify({
        'task_id': task_id,
        'keyframes': [f"/results/{task_id}/{os.path.basename(kf)}" for kf in keyframes],
        'storyboard': f"/results/{task_id}/storyboard.jpg",
        'manifest': json.dumps(manifest),
        'summary': summary
    })

@app.route('/process_live_stream', methods=['POST'])
def process_live_stream():
    task_id = str(uuid.uuid4())
    output_dir = os.path.join(app.config['RESULTS_FOLDER'], task_id)
    os.makedirs(output_dir, exist_ok=True)
    data = request.get_json()
    url = data.get('stream_url')
    return jsonify({'task_id': task_id, 'result': 'Live stream processing not supported'})

@app.route('/analyze_video_enhanced', methods=['POST'])
def analyze_video_enhanced():
    return analyze_video_basic()

@app.route('/results/<task_id>/<path:filename>')
def serve_result(task_id, filename):
    result_path = os.path.join(app.config['RESULTS_FOLDER'], task_id, filename)
    if os.path.exists(result_path):
        return send_file(result_path)
    return jsonify({'error': 'File not found'}), 404

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)
