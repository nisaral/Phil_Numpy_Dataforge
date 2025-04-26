import os
import uuid
import logging
import cv2
import numpy as np
from flask import Flask, request, render_template, jsonify, send_file
from werkzeug.utils import secure_filename
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from bs4 import BeautifulSoup
from youtube_transcript_api import YouTubeTranscriptApi
from sentence_transformers import SentenceTransformer
import faiss
from PIL import Image, ImageDraw, ImageFont, ImageEnhance, ImageFilter
import google.generativeai as genai
import json
from typing import List, Dict, Tuple, Optional
from graphviz import Digraph
from ultralytics import YOLO
import yt_dlp
import textwrap
from langdetect import detect, LangDetectException
from deep_translator import GoogleTranslator
from concurrent.futures import ThreadPoolExecutor, as_completed
import subprocess
import warnings
import time

# Suppress warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
logging.getLogger('tensorflow').setLevel(logging.ERROR)
warnings.filterwarnings('ignore', category=DeprecationWarning)

# Flask app setup
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['RESULTS_FOLDER'] = 'static/results'
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB limit

# Ensure directories exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['RESULTS_FOLDER'], exist_ok=True)

# Load environment variables
from dotenv import load_dotenv
load_dotenv()
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY not set in .env file")
genai.configure(api_key=GEMINI_API_KEY)

# Supported languages
SUPPORTED_LANGUAGES = {
    'en': {'name': 'English', 'font': 'arial.ttf'},
    'es': {'name': 'Spanish', 'font': 'arial.ttf'},
    'fr': {'name': 'French', 'font': 'arial.ttf'},
    'de': {'name': 'German', 'font': 'arial.ttf'},
    'zh-cn': {'name': 'Chinese (Simplified)', 'font': 'simhei.ttf'},
    'ja': {'name': 'Japanese', 'font': 'msgothic.ttc'},
    'ko': {'name': 'Korean', 'font': 'malgun.ttf'},
    'ar': {'name': 'Arabic', 'font': 'arial.ttf'},
    'ru': {'name': 'Russian', 'font': 'arial.ttf'},
    'hi': {'name': 'Hindi', 'font': 'mangal.ttf'},
    'pt': {'name': 'Portuguese', 'font': 'arial.ttf'},
    'it': {'name': 'Italian', 'font': 'arial.ttf'}
}

# Domain descriptors
DOMAIN_DESCRIPTORS = {
    'sports': {
        'focus': ['athletes', 'players', 'competition', 'field', 'court', 'action', 'movement', 'score'],
        'prompt': 'Analyze this sports video frame focusing on player positions, action moments, and game dynamics. Identify key athletic moments.',
        'colors': {'primary': (0, 128, 255), 'secondary': (255, 165, 0), 'text': (0, 0, 0)},
        'icons': ['🏃', '⚽', '⏱️']
    },
    'education': {
        'focus': ['instructor', 'classroom', 'slides', 'demonstration', 'explanation', 'experiment'],
        'prompt': 'Analyze this educational video frame focusing on key concepts, teaching moments, and visual demonstrations. Identify the main learning points.',
        'colors': {'primary': (76, 175, 80), 'secondary': (255, 235, 59), 'text': (33, 33, 33)},
        'icons': ['🏫', '📚', '🔬']
    },
    'news': {
        'focus': ['reporter', 'event', 'scene', 'interview', 'headline', 'location'],
        'prompt': 'Analyze this news video frame identifying the main event, location, people involved, and key information being reported.',
        'colors': {'primary': (33, 150, 243), 'secondary': (244, 67, 54), 'text': (0, 0, 0)},
        'icons': ['🌍', '📅', '👤']
    },
    'podcast': {
        'focus': ['speaker', 'host', 'guest', 'discussion', 'interview', 'conversation'],
        'prompt': 'Analyze this podcast video frame focusing on the speakers, conversation dynamics, and subject matter being discussed.',
        'colors': {'primary': (156, 39, 176), 'secondary': (103, 58, 183), 'text': (255, 255, 255)},
        'icons': ['🎙️', '🎧', '👤']
    },
    'surveillance': {
        'focus': ['motion', 'people', 'vehicles', 'activity', 'anomaly', 'security'],
        'prompt': 'Analyze this surveillance video frame identifying important activity, movement patterns, and subjects of interest.',
        'colors': {'primary': (66, 66, 66), 'secondary': (158, 158, 158), 'text': (255, 255, 255)},
        'icons': ['📹', '👁️', '⚠️']
    },
    'generic': {
        'focus': ['scene', 'action', 'people', 'objects', 'environment', 'mood'],
        'prompt': 'Analyze this video frame and describe the main elements, activities, and context of the scene.',
        'colors': {'primary': (0, 150, 136), 'secondary': (255, 193, 7), 'text': (0, 0, 0)},
        'icons': ['🎥', '🖼️', '🌐']
    }
}

# Initialize models
embedder = SentenceTransformer('all-MiniLM-L6-v2')
dimension = 384
index = faiss.IndexFlatL2(dimension)
knowledge_base: List[Dict] = []

try:
    yolo_model = YOLO("yolov11n.pt")
    print("YOLOv11 model loaded successfully.")
except Exception as e:
    print(f"Failed to load YOLOv11 model: {str(e)}. Falling back to vision-based analysis.")
    yolo_model = None

# Helper Functions
def detect_language(text: str) -> str:
    try:
        return detect(text)
    except LangDetectException:
        return 'en'

def translate_text(text: str, target_lang: str) -> str:
    if not text:
        return ""
    try:
        detected_lang = detect_language(text)
        if detected_lang == target_lang:
            return text
        translator = GoogleTranslator(source=detected_lang, target=target_lang)
        return translator.translate(text)
    except Exception as e:
        print(f"Translation error: {str(e)}")
        return text

def get_font_for_language(lang_code: str, size: int = 20) -> ImageFont.FreeTypeFont:
    font_file = SUPPORTED_LANGUAGES.get(lang_code, {}).get('font', 'arial.ttf')
    try:
        return ImageFont.truetype(font_file, size)
    except IOError:
        print(f"Font {font_file} not found. Using default.")
        return ImageFont.truetype('arial.ttf', size)

def check_ffmpeg():
    try:
        subprocess.run(['ffmpeg', '-version'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return True
    except FileNotFoundError:
        print("FFmpeg not found. Falling back to single-format download.")
        return False

# Crawling and Transcript Functions
def crawl_url(url: str, output_file: str = "crawled_data.json") -> str:
    try:
        options = Options()
        options.add_argument("--headless")
        driver = webdriver.Chrome(options=options)
        driver.get(url)
        driver.implicitly_wait(5)
        soup = BeautifulSoup(driver.page_source, 'html.parser')
        paragraphs = soup.find_all('p')
        text_content = " ".join([p.get_text(strip=True) for p in paragraphs if p.get_text(strip=True)])
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump({"url": url, "text": text_content}, f)
        driver.quit()
        return text_content if text_content else None
    except Exception as e:
        print(f"Error crawling {url}: {str(e)}")
        return None

def get_youtube_transcript(video_url: str) -> str:
    try:
        video_id = video_url.split("v=")[-1].split("&")[0]
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
        return " ".join([entry["text"] for entry in transcript])
    except Exception as e:
        print(f"Error fetching transcript for {video_url}: {str(e)}")
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
    print(f"Added {len(chunks)} chunks from {source} ({source_type}) to knowledge base.")

# Video Processing Functions
def download_youtube_video(url: str, output_dir: str = "temp_videos") -> str:
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    timestamp = int(time.time())
    output_path = os.path.join(output_dir, f"video_{timestamp}.mp4")
    ffmpeg_available = check_ffmpeg()
    ydl_opts = {
        'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best' if ffmpeg_available else 'best[ext=mp4]',
        'outtmpl': output_path,
        'merge_output_format': 'mp4' if ffmpeg_available else None,
        'quiet': True
    }
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
        print(f"Successfully downloaded video from {url}")
        return output_path
    except Exception as e:
        print(f"Error downloading YouTube video: {str(e)}")
        return None

def extract_keyframes(video_path, max_frames=None):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError("Could not open video file.")
    
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    duration = frame_count / fps if fps > 0 else 0
    
    if max_frames is None:
        max_frames = min(max(10, int(duration / 10)), 50)
    
    print(f"Video info: {frame_count} frames, {fps:.2f} FPS, {duration:.2f} seconds, targeting {max_frames} keyframes")
    
    sample_rate = max(1, frame_count // 1000) if frame_count > 10000 else 1
    prev_frame = None
    scores = []
    frame_idx = 0
    processed = 0
    
    def process_frame(frame_idx, frame):
        if prev_frame is not None and frame.shape == prev_frame.shape:
            diff = np.mean((frame.astype(float) - prev_frame.astype(float)) ** 2)
            return frame_idx, diff, frame.copy()
        return None
    
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx % sample_rate == 0:
                futures.append(executor.submit(process_frame, frame_idx, frame))
                processed += 1
            prev_frame = frame.copy()
            frame_idx += 1
    
        for future in as_completed(futures):
            result = future.result()
            if result:
                scores.append(result)
    
    cap.release()
    
    scores.sort(key=lambda x: x[1], reverse=True)
    keyframes = [(s[0], s[2]) for s in scores[:max_frames]]
    keyframes.sort(key=lambda x: x[0])
    
    key_moments = [(s[0], s[2]) for s in scores[:min(5, len(scores))]]
    key_moments.sort(key=lambda x: x[0])
    
    return keyframes, key_moments, fps, duration

def generate_text_summary(frame, frame_idx, fps, domain, lang='en'):
    domain_info = DOMAIN_DESCRIPTORS.get(domain, DOMAIN_DESCRIPTORS['generic'])
    objects_in_frame = []
    if yolo_model:
        try:
            results = yolo_model(frame, conf=0.5, verbose=False)
            for result in results:
                for box in result.boxes:
                    class_id = int(box.cls.cpu().numpy().item())
                    label = yolo_model.names[class_id]
                    objects_in_frame.append(label)
        except Exception as e:
            print(f"YOLO detection error: {str(e)}")
    
    domain_prompt = domain_info['prompt']
    focus_areas = ', '.join(domain_info['focus'])
    objects_text = ", ".join(objects_in_frame) if objects_in_frame else "no recognized objects"
    
    prompt = f"""
    {domain_prompt}
    
    Time in video: {frame_idx/fps:.2f} seconds
    Domain: {domain}
    Focus areas: {focus_areas}
    Objects detected: {objects_text}
    
    Provide a concise 1-2 sentence description of this frame.
    """
    
    try:
        model = genai.GenerativeModel('gemini-1.5-flash')
        response = model.generate_content(prompt)
        summary = response.text.strip()
        if lang != 'en':
            summary = translate_text(summary, lang)
        return summary
    except Exception as e:
        print(f"Error generating summary: {str(e)}")
        return f"Scene at {frame_idx/fps:.2f}s in {domain} video."

def generate_mock_test(text: str, domain: str, lang: str = 'en') -> str:
    domain_info = DOMAIN_DESCRIPTORS.get(domain, DOMAIN_DESCRIPTORS['generic'])
    prompt = f"""
    Based on the following {domain} content, generate a mock test with 5 multiple-choice questions, each with 4 options and one correct answer.

    Content: {text[:2000]}

    Focus areas: {', '.join(domain_info['focus'])}
    """
    try:
        model = genai.GenerativeModel('gemini-1.5-flash')
        response = model.generate_content(prompt)
        result = response.text.strip()
        if lang != 'en':
            result = translate_text(result, lang)
        return result
    except Exception as e:
        print(f"Error generating mock test: {str(e)}")
        return "Unable to generate mock test."

def generate_mind_map(text: str, domain: str, output_dir: str, lang: str = 'en') -> str:
    DOMAIN_DESCRIPTORS.get(domain, DOMAIN_DESCRIPTORS['generic'])
    prompt = f"""
    Create a textual representation of a mind map based on the following {domain} content. Structure it as a central topic with 3-5 main branches, each with 2-3 sub-branches. Use bullet points for clarity.

    Content: {text[:2000]}

    """
    try:
        model = genai.GenerativeModel('gemini-1.5-flash')
        response = model.generate_content(prompt)
        mind_map = response.text.strip()
        if lang != 'en':
            mind_map = translate_text(mind_map, lang)
        
        output_path = os.path.join(output_dir, 'mind_map.txt')
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(mind_map)
        return output_path
    except Exception as e:
        print(f"Error generating mind map: {str(e)}")
        return None

def generate_flowchart(text: str, domain: str, output_dir: str, lang: str = 'en') -> str:
    DOMAIN_DESCRIPTORS.get(domain, DOMAIN_DESCRIPTORS['generic'])
    prompt = f"""
    Based on the following {domain} content, generate a flowchart description in DOT format for Graphviz. The flowchart should represent the main process or sequence of events with 5-7 nodes and appropriate connections.

    Content: {text[:2000]}

    """
    try:
        model = genai.GenerativeModel('gemini-1.5-flash')
        response = model.generate_content(prompt)
        dot_content = response.text.strip()
        if lang != 'en':
            dot_content = translate_text(dot_content, lang)
        
        dot = Digraph(comment='Flowchart')
        lines = dot_content.split('\n')
        for line in lines:
            if '->' in line:
                parts = line.split('->')
                if len(parts) == 2:
                    dot.edge(parts[0].strip(), parts[1].strip())
            elif line.strip():
                dot.node(line.strip())
        
        output_path = os.path.join(output_dir, 'flowchart.svg')
        dot.render(output_path, format='svg', cleanup=True)
        return output_path
    except Exception as e:
        print(f"Error generating flowchart: {str(e)}")
        return None

def create_rounded_corners(image, radius=20):
    mask = Image.new('L', image.size, 0)
    draw = ImageDraw.Draw(mask)
    draw.rounded_rectangle((0, 0, image.size[0], image.size[1]), radius=radius, fill=255)
    rounded = Image.new('RGBA', image.size, (0, 0, 0, 0))
    rounded.paste(image, (0, 0), mask)
    return rounded

def create_storyboard_image(keyframes, key_moments, summaries, output_dir, domain, video_input, duration, fps, lang='en'):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    domain_info = DOMAIN_DESCRIPTORS.get(domain, DOMAIN_DESCRIPTORS['generic'])
    primary_color = domain_info['colors']['primary']
    secondary_color = domain_info['colors']['secondary']
    text_color = domain_info['colors']['text']
    
    num_images = len(keyframes)
    cols = min(3, num_images)
    rows = (num_images + cols - 1) // cols
    cell_width, cell_height = 400, 300
    margin = 40
    text_height = 160
    title_height = 120
    progress_height = 25
    
    bg_color = tuple(c//3 for c in primary_color)
    total_width = cols * (cell_width + margin) + margin
    total_height = rows * (cell_height + text_height + progress_height + margin) + margin + title_height
    storyboard = Image.new('RGBA', (total_width, total_height), (255, 255, 255, 0))
    draw = ImageDraw.Draw(storyboard)
    
    bg = Image.new('RGB', (total_width, total_height), (255, 255, 255))
    for y in range(total_height):
        r = int(255 - (255 - bg_color[0]) * y / total_height)
        g = int(255 - (255 - bg_color[1]) * y / total_height)
        b = int(255 - (255 - bg_color[2]) * y / total_height)
        for x in range(total_width):
            bg.putpixel((x, y), (r, g, b))
    bg = bg.filter(ImageFilter.GaussianBlur(radius=10))
    storyboard.paste(bg, (0, 0))
    
    try:
        font = get_font_for_language(lang, 20)
        title_font = get_font_for_language(lang, 32)
    except Exception:
        font = ImageFont.truetype('arial.ttf', 20)
        title_font = ImageFont.truetype('arial.ttf', 32)
    
    video_name = os.path.basename(video_input) if not video_input.startswith("http") else video_input.split("?")[0][-20:]
    if len(video_name) > 50:
        video_name = video_name[:47] + "..."
    
    domain_name = domain.capitalize()
    if lang != 'en':
        domain_name = translate_text(domain_name, lang)
    
    title = f"{SUPPORTED_LANGUAGES.get(lang, {}).get('name', 'Multilingual')} Storyboard: {domain_name} - {video_name}"
    draw.rectangle([0, 0, total_width, title_height + margin], fill=(*primary_color, 180), outline=(255, 255, 255, 200))
    title_bbox = draw.textbbox((0, 0), title, font=title_font)
    title_width = title_bbox[2] - title_bbox[0]
    draw.text(((total_width - title_width) // 2, margin), title, fill='white', font=title_font)
    
    manifest = {
        'keyframes': [],
        'key_moments': [],
        'domain': domain,
        'video_name': video_name,
        'language': lang,
        'duration': duration,
        'cols': cols,
        'rows': rows,
        'cell_width': cell_width,
        'cell_height': cell_height
    }
    
    for i, (frame_idx, frame) in enumerate(keyframes):
        summary = summaries[i]
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame_rgb)
        img = img.resize((cell_width, cell_height), Image.LANCZOS)
        img = create_rounded_corners(img, radius=20)
        
        glow = img.filter(ImageFilter.GaussianBlur(radius=5))
        glow = ImageEnhance.Brightness(glow).enhance(1.5)
        glow = ImageEnhance.Color(glow).enhance(0.8)
        
        shadow = Image.new('RGBA', (cell_width + 12, cell_height + 12), (0, 0, 0, 80))
        x = margin + (i % cols) * (cell_width + margin) - 6
        y = margin + title_height + (i // cols) * (cell_height + text_height + progress_height + margin) - 6
        storyboard.paste(shadow, (x, y), shadow)
        storyboard.paste(glow, (x + 6, y + 6), glow)
        
        border_color = secondary_color
        border_img = Image.new('RGBA', (cell_width + 4, cell_height + 4), (*border_color, 200))
        border_img.paste(img, (2, 2), img)
        storyboard.paste(border_img, (x + 6, y + 6), border_img)
        
        progress_y = y + cell_height + 10
        progress_width = int(cell_width * (frame_idx / (fps * duration)))
        draw.rectangle([x + 6, progress_y, x + 6 + progress_width, progress_y + progress_height], fill=(*secondary_color, 220))
        draw.rectangle([x + 6, progress_y, x + 6 + cell_width, progress_y + progress_height], fill=(200, 200, 200, 100), outline=(*primary_color, 200))
        
        text_y = progress_y + progress_height + 15
        text_x = x + 6
        timestamp = f"Time: {frame_idx/fps:.1f}s"
        if lang != 'en':
            timestamp = translate_text(timestamp, lang)
        
        draw.text((text_x, text_y), timestamp, fill=text_color, font=font)
        
        wrapped_summary = textwrap.fill(summary, width=40)
        draw.text((text_x, text_y + 35), wrapped_summary, fill=text_color, font=font)
        
        icon_text = domain_info['icons'][min(i, len(domain_info['icons'])-1)]
        draw.text((text_x + cell_width - 50, y + 15), icon_text, fill='white', font=font)
        
        manifest['keyframes'].append({
            'index': i,
            'frame_idx': frame_idx,
            'timestamp': frame_idx/fps,
            'summary': summary,
            'icon': icon_text,
            'x': x + 6,
            'y': y + 6,
            'z': -i * 10,
            'path': f"keyframe_{i+1}.jpg",
            'is_key_moment': frame_idx in [km[0] for km in key_moments]
        })
    
    for i, (frame_idx, _) in enumerate(key_moments):
        manifest['key_moments'].append({
            'index': i,
            'frame_idx': frame_idx,
            'timestamp': frame_idx/fps,
            'summary': summaries[keyframes.index((frame_idx, keyframes[[kf[0] for kf in keyframes].index(frame_idx)][1]))],
            'path': f"keyframe_{[kf[0] for kf in keyframes].index(frame_idx)+1}.jpg"
        })
    
    output_path = os.path.join(output_dir, 'storyboard.jpg')
    storyboard.save(output_path, quality=95)
    
    manifest_path = os.path.join(output_dir, 'storyboard_manifest.json')
    with open(manifest_path, 'w', encoding='utf-8') as f:
        json.dump(manifest, f, ensure_ascii=False)
    
    return output_path

def process_video_basic(video_input: str, domain: str, output_dir: str, lang: str = 'en') -> Tuple[Optional[List[str]], str]:
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    video_path = video_input
    if video_input.startswith("http") and ("youtube.com" in video_input or "youtu.be" in video_input):
        video_path = download_youtube_video(video_input)
        if not video_path:
            return None, "Error: Could not download YouTube video."
    
    if not video_path.startswith("http") and not os.path.exists(video_path):
        return None, "Error: Video file does not exist."
    if not video_path.lower().endswith(('.mp4', '.mov')):
        return None, "Error: Only .mp4 and .mov files are supported."
    
    try:
        keyframes, key_moments, fps, duration = extract_keyframes(video_path)
    except Exception as e:
        print(f"Keyframe extraction failed: {str(e)}")
        return None, "Error: Keyframe extraction failed."
    
    summaries = []
    highlight_frames = []
    
    def process_keyframe(frame_idx, frame):
        summary = generate_text_summary(frame, frame_idx, fps, domain, lang)
        output_path = os.path.join(output_dir, f"keyframe_{len(highlight_frames)+1}.jpg")
        cv2.imwrite(output_path, frame)
        return frame_idx, frame, summary, output_path
    
    with ThreadPoolExecutor(max_workers=4) as executor:
        future_to_keyframe = {executor.submit(process_keyframe, idx, frame): (idx, frame) for idx, frame in keyframes}
        for future in as_completed(future_to_keyframe):
            frame_idx, frame, summary, output_path = future.result()
            summaries.append(summary)
            highlight_frames.append(output_path)
    
    if highlight_frames:
        storyboard_path = create_storyboard_image(keyframes, key_moments, summaries, output_dir, domain, video_input, duration, fps, lang)
    
    text_output = generate_domain_specific_output(
        [{'description': s, 'score': 50, 'text': ''} for s in summaries], domain
    )
    
    faiss.write_index(index, os.path.join(output_dir, "faiss_index.bin"))
    with open(os.path.join(output_dir, "knowledge_base.json"), 'w', encoding='utf-8') as f:
        json.dump(knowledge_base, f, ensure_ascii=False)
    
    return highlight_frames, text_output

def process_video_enhanced(video_input: str, domain: str, output_dir: str, lang: str = 'en') -> Tuple[Optional[List[str]], str]:
    return process_video_basic(video_input, domain, output_dir, lang)

def process_live_stream(url: str, domain: str, output_dir: str, lang: str = 'en') -> str:
    print(f"Live stream processing not implemented for {url}")
    return "Live stream processing is not currently supported."

def generate_domain_specific_output(segments, domain):
    domain_info = DOMAIN_DESCRIPTORS.get(domain, DOMAIN_DESCRIPTORS['generic'])
    summaries = [s.get('description', '') for s in segments]
    combined_summary = ' '.join(summaries)
    
    prompt = f"""
    You are analyzing a {domain} video. Based on these key moment descriptions, create a comprehensive summary:
    
    Key moments: {combined_summary}
    
    Focus areas for {domain} videos: {', '.join(domain_info['focus'])}
    
    Provide:
    1. An overall summary (2-3 sentences)
    2. Key highlights (3-4 bullet points)
    3. Main takeaways for this {domain} content
    """
    
    try:
        model = genai.GenerativeModel('gemini-1.5-flash')
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        print(f"Error generating domain-specific output: {str(e)}")
        return f"Summary of {domain} video: " + ' '.join(summaries[:2])

# Flask Routes
@app.route('/')
def index():
    return render_template(
        'index.html',
        domains=DOMAIN_DESCRIPTORS.keys(),
        languages=SUPPORTED_LANGUAGES.keys(),
        supported_languages=SUPPORTED_LANGUAGES
    )

@app.route('/add_web', methods=['POST'])
def add_web():
    task_id = str(uuid.uuid4())
    output_dir = os.path.join(app.config['RESULTS_FOLDER'], task_id)
    os.makedirs(output_dir, exist_ok=True)
    
    data = request.get_json()
    web_url = data.get('web_url')
    lang = data.get('lang', 'en')
    
    if not web_url or not web_url.startswith('http'):
        return jsonify({'error': 'Invalid URL'}), 400
    
    text = crawl_url(web_url, os.path.join(output_dir, 'web_content.json'))
    if text:
        update_knowledge_base(text, web_url, "web")
        return jsonify({'task_id': task_id, 'content': text})
    return jsonify({'error': 'Failed to crawl web content'}), 500

@app.route('/add_youtube', methods=['POST'])
def add_youtube():
    task_id = str(uuid.uuid4())
    output_dir = os.path.join(app.config['RESULTS_FOLDER'], task_id)
    os.makedirs(output_dir, exist_ok=True)
    
    data = request.get_json()
    youtube_url = data.get('youtube_url')
    lang = data.get('lang', 'en')
    
    if not youtube_url or not youtube_url.startswith('http'):
        return jsonify({'error': 'Invalid YouTube URL'}), 400
    
    transcript = get_youtube_transcript(youtube_url)
    if transcript:
        update_knowledge_base(transcript, youtube_url, "video")
        return jsonify({'task_id': task_id, 'content': transcript})
    return jsonify({'error': 'Failed to extract YouTube content'}), 500

@app.route('/add_text', methods=['POST'])
def add_text():
    task_id = str(uuid.uuid4())
    output_dir = os.path.join(app.config['RESULTS_FOLDER'], task_id)
    os.makedirs(output_dir, exist_ok=True)
    
    data = request.get_json()
    text = data.get('text')
    lang = data.get('lang', 'en')
    
    if not text:
        return jsonify({'error': 'No text provided'}), 400
    
    update_knowledge_base(text, "user_input", "text")
    output_path = os.path.join(output_dir, 'user_text.txt')
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(text)
    return jsonify({'task_id': task_id, 'content': text})

@app.route('/ask_question', methods=['POST'])
def ask_question():
    data = request.get_json()
    question = data.get('question')
    lang = data.get('lang', 'en')
    
    if not question:
        return jsonify({'error': 'No question provided'}), 400
    
    embeddings = embedder.encode([question], convert_to_numpy=True)
    _, indices = index.search(embeddings, k=5)
    context = "\n".join([knowledge_base[i]['text'] for i in indices[0] if i < len(knowledge_base)])
    
    prompt = f"""
    Based on the following context, answer the question concisely:

    Context: {context}

    Question: {question}
    """
    try:
        model = genai.GenerativeModel('gemini-1.5-flash')
        response = model.generate_content(prompt)
        answer = response.text.strip()
        if lang != 'en':
            answer = translate_text(answer, lang)
        return jsonify({'answer': answer})
    except Exception as e:
        print(f"Error answering question: {str(e)}")
        return jsonify({'error': 'Failed to generate answer'}), 500

@app.route('/get_summary', methods=['POST'])
def get_summary():
    data = request.get_json()
    text = data.get('text')
    lang = data.get('lang', 'en')
    
    if not text:
        return jsonify({'error': 'No text provided'}), 400
    
    prompt = f"""
    Summarize the following text in 2-3 sentences:

    {text[:2000]}
    """
    try:
        model = genai.GenerativeModel('gemini-1.5-flash')
        response = model.generate_content(prompt)
        summary = response.text.strip()
        if lang != 'en':
            summary = translate_text(summary, lang)
        return jsonify({'summary': summary})
    except Exception as e:
        print(f"Error generating summary: {str(e)}")
        return jsonify({'error': 'Failed to generate summary'}), 500

@app.route('/generate_mock_test', methods=['POST'])
def generate_mock_test_route():
    task_id = str(uuid.uuid4())
    output_dir = os.path.join(app.config['RESULTS_FOLDER'], task_id)
    os.makedirs(output_dir, exist_ok=True)
    
    data = request.get_json()
    text = data.get('text')
    domain = data.get('domain', 'generic')
    lang = data.get('lang', 'en')
    
    if not text:
        return jsonify({'error': 'No text provided'}), 400
    
    mock_test = generate_mock_test(text, domain, lang)
    output_path = os.path.join(output_dir, 'mock_test.txt')
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(mock_test)
    return jsonify({'task_id': task_id, 'mock_test': mock_test})

@app.route('/generate_mind_map', methods=['POST'])
def generate_mind_map_route():
    task_id = str(uuid.uuid4())
    output_dir = os.path.join(app.config['RESULTS_FOLDER'], task_id)
    os.makedirs(output_dir, exist_ok=True)
    
    data = request.get_json()
    text = data.get('text')
    domain = data.get('domain', 'generic')
    lang = data.get('lang', 'en')
    
    if not text:
        return jsonify({'error': 'No text provided'}), 400
    
    output_path = generate_mind_map(text, domain, output_dir, lang)
    if output_path:
        return jsonify({'task_id': task_id, 'mind_map': output_path})
    return jsonify({'error': 'Failed to generate mind map'}), 500

@app.route('/generate_flowchart', methods=['POST'])
def generate_flowchart_route():
    task_id = str(uuid.uuid4())
    output_dir = os.path.join(app.config['RESULTS_FOLDER'], task_id)
    os.makedirs(output_dir, exist_ok=True)
    
    data = request.get_json()
    text = data.get('text')
    domain = data.get('domain', 'generic')
    lang = data.get('lang', 'en')
    
    if not text:
        return jsonify({'error': 'No text provided'}), 400
    
    output_path = generate_flowchart(text, domain, output_dir, lang)
    if output_path:
        return jsonify({'task_id': task_id, 'flowchart': output_path})
    return jsonify({'error': 'Failed to generate flowchart'}), 500

@app.route('/analyze_video_basic', methods=['POST'])
def analyze_video_basic():
    task_id = str(uuid.uuid4())
    output_dir = os.path.join(app.config['RESULTS_FOLDER'], task_id)
    os.makedirs(output_dir, exist_ok=True)
    
    if 'video' in request.files:
        domain = request.form.get('domain', 'generic')
        lang = request.form.get('lang', 'en')
        
        file = request.files['video']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        filename = secure_filename(file.filename)
        video_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{task_id}_{filename}")
        file.save(video_path)
        
        try:
            keyframes, text_output = process_video_basic(video_path, domain, output_dir, lang)
            if keyframes:
                return jsonify({
                    'task_id': task_id,
                    'keyframes': [f"/results/{task_id}/{os.path.basename(kf)}" for kf in keyframes],
                    'storyboard': f"/results/{task_id}/storyboard.jpg",
                    'manifest': f"/results/{task_id}/storyboard_manifest.json",
                    'summary': text_output
                })
            return jsonify({'error': text_output}), 500
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    data = request.get_json()
    youtube_url = data.get('youtube_url')
    domain = data.get('domain', 'generic')
    lang = data.get('lang', 'en')
    
    if not youtube_url or not youtube_url.startswith('http'):
        return jsonify({'error': 'Invalid YouTube URL'}), 400
    
    transcript = get_youtube_transcript(youtube_url)
    if transcript:
        update_knowledge_base(transcript, youtube_url, "video")
    
    try:
        keyframes, text_output = process_video_basic(youtube_url, domain, output_dir, lang)
        if keyframes:
            return jsonify({
                'task_id': task_id,
                'keyframes': [f"/results/{task_id}/{os.path.basename(kf)}" for kf in keyframes],
                'storyboard': f"/results/{task_id}/storyboard.jpg",
                'manifest': f"/results/{task_id}/storyboard_manifest.json",
                'summary': text_output
            })
        return jsonify({'error': text_output}), 500
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/process_live_stream', methods=['POST'])
def process_live_stream_route():
    task_id = str(uuid.uuid4())
    output_dir = os.path.join(app.config['RESULTS_FOLDER'], task_id)
    os.makedirs(output_dir, exist_ok=True)
    
    data = request.get_json()
    stream_url = data.get('stream_url')
    domain = data.get('domain', 'generic')
    lang = data.get('lang', 'en')
    
    if not stream_url:
        return jsonify({'error': 'No stream URL provided'}), 400
    
    result = process_live_stream(stream_url, domain, output_dir, lang)
    return jsonify({'task_id': task_id, 'result': result})

@app.route('/analyze_video_enhanced', methods=['POST'])
def analyze_video_enhanced():
    task_id = str(uuid.uuid4())
    output_dir = os.path.join(app.config['RESULTS_FOLDER'], task_id)
    os.makedirs(output_dir, exist_ok=True)
    
    if 'video' in request.files:
        domain = request.form.get('domain', 'generic')
        lang = request.form.get('lang', 'en')
        
        file = request.files['video']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        filename = secure_filename(file.filename)
        video_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{task_id}_{filename}")
        file.save(video_path)
        
        try:
            keyframes, text_output = process_video_enhanced(video_path, domain, output_dir, lang)
            if keyframes:
                return jsonify({
                    'task_id': task_id,
                    'keyframes': [f"/results/{task_id}/{os.path.basename(kf)}" for kf in keyframes],
                    'storyboard': f"/results/{task_id}/storyboard.jpg",
                    'manifest': f"/results/{task_id}/storyboard_manifest.json",
                    'summary': text_output
                })
            return jsonify({'error': text_output}), 500
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    data = request.get_json()
    youtube_url = data.get('youtube_url')
    domain = data.get('domain', 'generic')
    lang = data.get('lang', 'en')
    
    if not youtube_url or not youtube_url.startswith('http'):
        return jsonify({'error': 'Invalid YouTube URL'}), 400
    
    transcript = get_youtube_transcript(youtube_url)
    if transcript:
        update_knowledge_base(transcript, youtube_url, "video")
    
    try:
        keyframes, text_output = process_video_enhanced(youtube_url, domain, output_dir, lang)
        if keyframes:
            return jsonify({
                'task_id': task_id,
                'keyframes': [f"/results/{task_id}/{os.path.basename(kf)}" for kf in keyframes],
                'storyboard': f"/results/{task_id}/storyboard.jpg",
                'manifest': f"/results/{task_id}/storyboard_manifest.json",
                'summary': text_output
            })
        return jsonify({'error': text_output}), 500
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/results/<task_id>/<path:filename>')
def serve_result(task_id, filename):
    result_path = os.path.join(app.config['RESULTS_FOLDER'], task_id, filename)
    if os.path.exists(result_path):
        return send_file(result_path)
    return jsonify({'error': 'File not found'}), 404

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)