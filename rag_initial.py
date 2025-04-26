from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from bs4 import BeautifulSoup
from youtube_transcript_api import YouTubeTranscriptApi
from crawl4ai import WebCrawler
from crawl4ai.extraction_strategy import JsonCssExtractionStrategy
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import google.generativeai as genai
import re
import json
from typing import List, Dict
from graphviz import Digraph
import os
import cv2
from groq import Groq
import base64
from io import BytesIO
from PIL import Image
import yt_dlp
from scipy import stats
from ultralytics import YOLO

# Configuration
GEMINI_API_KEY = "AIzaSyDq3W6bcmtED-s0vDKmSBZr8uIwy4Gc1Io"
GROQ_API_KEY = "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"
genai.configure(api_key=GEMINI_API_KEY)
groq_client = Groq(api_key=GROQ_API_KEY)

# Initialize models
embedder = SentenceTransformer('all-MiniLM-L6-v2')
dimension = 384
index = faiss.IndexFlatL2(dimension)
knowledge_base: List[Dict] = []
vector_base = []
crawler = WebCrawler()
crawler.warmup()
extraction_strategy = JsonCssExtractionStrategy(
    schema={"text_content": {"css": "p", "type": "string"}}
)

# Initialize YOLOv8 model
try:
    yolo_model = YOLO("yolov8n.pt")  # Nano model for speed
    print("YOLOv8 model loaded successfully.")
except Exception as e:
    print(f"Failed to load YOLOv8 model: {str(e)}. Falling back to vision-based analysis.")
    yolo_model = None

# --- Existing Functions (Unchanged) ---
def crawl_url(url: str, output_file: str = "crawled_data.json") -> str:
    try:
        options = Options()
        options.add_argument("--headless")
        options.add_argument("--disable-gpu")
        options.add_argument("--no-sandbox")
        driver = webdriver.Chrome(options=options)
        driver.get(url)
        driver.implicitly_wait(5)
        soup = BeautifulSoup(driver.page_source, 'html.parser')
        paragraphs = soup.find_all('p')
        text_content = " ".join([p.get_text(strip=True) for p in paragraphs if p.get_text(strip=True)])
        with open(output_file, "w") as f:
            json.dump({"url": url, "text": text_content}, f)
        print(f"Crawled {url} and saved to {output_file}")
        driver.quit()
        return text_content if text_content else None
    except Exception as e:
        print(f"Error crawling {url} with Selenium: {str(e)}")
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
    video_id = video_url.split("v=")[-1].split("&")[0]
    try:
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

def process_input(source: str, source_type: str = "web"):
    if source_type == "web":
        text = crawl_url(source)
        if text:
            update_knowledge_base(text, source, "web")
    elif source_type == "video":
        page_text = crawl_youtube(source)
        transcript = get_youtube_transcript(source)
        text = f"{page_text or ''} {transcript or ''}".strip()
        if text:
            update_knowledge_base(text, source, "video")
    elif source_type == "user":
        update_knowledge_base(source, "User Input", "text")

def retrieve_chunks(query: str, source: str = None) -> List[str]:
    query_embedding = embedder.encode([query], convert_to_numpy=True)
    distances, indices = index.search(query_embedding, k=3)
    if source:
        chunks = [kb["text"] for kb in knowledge_base if kb["source"] == source][:3]
    else:
        chunks = [knowledge_base[i]["text"] for i in indices[0]]
    return chunks

def generate_answer(query: str, source: str = None) -> str:
    context = "\n".join(retrieve_chunks(query, source))
    prompt = f"""
    You are an expert educational assistant. Based on the following context, provide a clear, concise, and accurate answer to the question. Use the context provided and avoid speculation beyond it.

    Context: {context}
    
    Question: {query}
    
    Answer:
    """
    model = genai.GenerativeModel('gemini-1.5-flash')
    response = model.generate_content(prompt)
    return response.text.strip()

def generate_summary(source: str = None) -> str:
    if not knowledge_base:
        return "No content available to summarize."
    if source:
        context = "\n".join([kb["text"] for kb in knowledge_base if kb["source"] == source][:3])
        title = f"Summary for {source}"
    else:
        context = "\n".join([kb["text"] for kb in knowledge_base][:3])
        title = "Summary of All Content"
    prompt = f"""
    You are a skilled summarizer. Provide a concise and informative summary of the following content in 2-3 sentences.

    Content: {context}
    
    Summary:
    """
    model = genai.GenerativeModel('gemini-1.5-flash')
    response = model.generate_content(prompt)
    return f"### {title}\n{response.text.strip()}"

def generate_mock_test(topic: str, source: str = None, num_questions: int = 5) -> str:
    context = "\n".join(retrieve_chunks(topic, source))
    prompt = f"""
    You are an educational agent tasked with creating a mock test. Based on the following context, generate a mock test with {num_questions} multiple-choice questions on the topic '{topic}'. Each question should have 4 options (A, B, C, D) and the correct answer clearly marked.

    Context: {context}
    
    Mock Test:
    """
    model = genai.GenerativeModel('gemini-1.5-flash')
    response = model.generate_content(prompt)
    return f"### Mock Test on {topic}\n{response.text.strip()}"

def generate_mind_map(topic: str, source: str = None) -> str:
    context = "\n".join(retrieve_chunks(topic, source))
    prompt = f"""
    You are a learning expert. Based on the following context, create a Graphviz DOT language script for a mind map on the topic '{topic}'. The mind map should have a central node (the topic), main branches, and sub-branches. Keep it concise and structured.

    Context: {context}
    
    Graphviz DOT Script:
    """
    model = genai.GenerativeModel('gemini-1.5-flash')
    response = model.generate_content(prompt)
    dot_content = response.text.strip()

    print(f"\n### Mind Map DOT Script for {topic}\n{dot_content}")
    try:
        confirm = input("Generate mind map image (PNG) from this script? (y/n): ").strip().lower()
        if confirm == 'y':
            dot = Digraph(comment=f'Mind Map for {topic}', format='png')
            if '```dot' in dot_content:
                dot_content = dot_content.split('```dot')[1].split('```')[0].strip()
            dot.body.append(dot_content)
            output_file = f"mind_map_{topic.replace(' ', '_')}"
            dot.render(output_file, view=True)
            return f"Generated mind map image: {output_file}.png"
    except EOFError:
        print("Input stream closed. Skipping image generation.")
    return "Image generation skipped."

def generate_flowchart(topic: str, source: str = None) -> str:
    context = "\n".join(retrieve_chunks(topic, source))
    prompt = f"""
    You are a process design expert. Based on the following context, create a Graphviz DOT language script for a flowchart on the topic '{topic}'. The flowchart should represent a sequence of steps or decisions in a logical order. Use shapes like 'box' for steps and 'diamond' for decisions.

    Context: {context}
    
    Graphviz DOT Script:
    """
    model = genai.GenerativeModel('gemini-1.5-flash')
    response = model.generate_content(prompt)
    dot_content = response.text.strip()

    print(f"\n### Flowchart DOT Script for {topic}\n{dot_content}")
    try:
        confirm = input("Generate flowchart image (PNG) from this script? (y/n): ").strip().lower()
        if confirm == 'y':
            dot = Digraph(comment=f'Flowchart for {topic}', format='png')
            if '```dot' in dot_content:
                dot_content = dot_content.split('```dot')[1].split('```')[0].strip()
            dot.body.append(dot_content)
            output_file = f"flowchart_{topic.replace(' ', '_')}"
            dot.render(output_file, view=True)
            return f"Generated flowchart image: {output_file}.png"
    except EOFError:
        print("Input stream closed. Skipping image generation.")
    return "Image generation skipped."

# --- Video Analysis Functions ---
def download_youtube_video(url: str, output_dir: str = "temp_videos") -> str:
    """Download a YouTube video as MP4 using yt-dlp."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    ydl_opts = {
        'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best',
        'outtmpl': os.path.join(output_dir, 'video.%(ext)s'),
        'merge_output_format': 'mp4',
        'quiet': True
    }
    
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            return os.path.join(output_dir, f"video.mp4")
    except Exception as e:
        print(f"Error downloading YouTube video: {str(e)}")
        return None

def frame_to_base64(frame):
    """Convert OpenCV frame to base64 string for Groq API."""
    _, buffer = cv2.imencode('.jpg', frame)
    img = Image.open(BytesIO(buffer))
    img_byte_arr = BytesIO()
    img.save(img_byte_arr, format='JPEG')
    return base64.b64encode(img_byte_arr.getvalue()).decode('utf-8')

def analyze_frame_with_groq(frame, domain: str) -> Dict:
    """Analyze a single frame using Groq's LLaMA 3.2 vision model."""
    base64_image = frame_to_base64(frame)
    prompt = f"""
    You are an expert in {domain} video analysis. Analyze this frame from a {domain} video and provide:
    1. A brief description of the scene (actions, objects, people).
    2. Significance score (0-100) based on visual activity (motion, scene change, key objects).
    3. Any visible text in the frame.
    """
    if domain == "generic":
        prompt = """
        Analyze this frame from a video and provide:
        1. A brief description of the scene (actions, objects, people).
        2. Significance score (0-100) based on visual activity (motion, scene change, key objects).
        3. Any visible text in the frame.
        """
    try:
        response = groq_client.chat.completions.create(
            model="llama-3.2-11b-vision-preview",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                    ]
                }
            ],
            max_tokens=200
        )
        result = response.choices[0].message.content
        lines = result.split('\n')
        description = lines[0] if lines else "No description"
        score = int(re.search(r'\d+', lines[1]).group()) if len(lines) > 1 and re.search(r'\d+', lines[1]) else 50
        text = lines[2] if len(lines) > 2 else ""
        return {
            'description': description,
            'score': score,
            'text': text
        }
    except Exception as e:
        print(f"Error analyzing frame with Groq: {str(e)}")
        return {'description': 'Error in analysis', 'score': 50, 'text': ''}

def calculate_motion_score(prev_frame, curr_frame):
    """Calculate motion intensity between two frames."""
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
    diff = cv2.absdiff(prev_gray, curr_gray)
    motion_score = np.mean(diff)
    return motion_score

def detect_scene_change(prev_frame, curr_frame, threshold=0.7):
    """Detect scene changes using histogram correlation."""
    prev_hist = cv2.calcHist([prev_frame], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    curr_hist = cv2.calcHist([curr_frame], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    prev_hist = cv2.normalize(prev_hist, prev_hist).flatten()
    curr_hist = cv2.normalize(curr_hist, curr_hist).flatten()
    correlation = stats.pearsonr(prev_hist, curr_hist)[0]
    return correlation < threshold

def generate_domain_specific_output(keyframes, domain: str) -> str:
    """Generate domain-specific text output using Gemini."""
    if domain == "generic":
        return "Generic domain: Visual storyboard generated without domain-specific summary."
    
    descriptions = [f"Frame {i+1}: {kf['description']} (Score: {kf['score']}, Text: {kf['text']})" for i, kf in enumerate(keyframes)]
    prompt = f"""
    You are an expert in {domain} analysis. Based on the following key moments from a {domain} video, generate a concise summary or commentary. Include relevant details like actions, events, or text visible in the frames.

    Key Moments: {descriptions}

    Output:
    """
    if domain == "sports":
        prompt += "\nGenerate sports commentary and predict the likely winner if applicable."
    elif domain == "podcast":
        prompt += "\nSummarize key discussion points or topics."
    elif domain == "news":
        prompt += "\nExtract key headlines or events."
    elif domain == "education":
        prompt += "\nIdentify key concepts or topics taught."
    elif domain == "surveillance":
        prompt += "\nFlag significant events (e.g., person entry/exit, unusual objects)."

    model = genai.GenerativeModel('gemini-1.5-flash')
    response = model.generate_content(prompt)
    return f"### {domain.capitalize()} Summary\n{response.text.strip()}"

def process_video(video_input: str, domain: str, output_dir: str = "storyboard", max_frames: int = 5) -> tuple:
    """Process a video file or YouTube link and generate storyboard and text output."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Handle YouTube links
    video_path = video_input
    if video_input.startswith("http") and ("youtube.com" in video_input or "youtu.be" in video_input):
        video_path = download_youtube_video(video_input)
        if not video_path:
            return None, "Error: Could not download YouTube video."
    
    # Validate local file
    if not video_path.startswith("http") and not os.path.exists(video_path):
        return None, "Error: Video file does not exist."
    if not video_path.lower().endswith(('.mp4', '.mov')):
        return None, "Error: Only .mp4 and .mov files are supported."
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None, "Error: Could not open video."
    
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    segment_duration = 1.0
    frames_per_segment = int(fps * segment_duration)
    
    segments = []
    prev_frame = None
    frame_idx = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_idx % frames_per_segment == 0:
            segment_score = 0
            segment_frame = frame.copy()
            
            # Analyze frame with Groq's LLaMA vision model
            analysis = analyze_frame_with_groq(frame, domain)
            segment_score += analysis['score'] * 0.5
            
            # Supplement with motion and scene change detection
            if prev_frame is not None:
                motion_score = calculate_motion_score(prev_frame, frame)
                segment_score += motion_score * 0.3
                
                if detect_scene_change(prev_frame, frame):
                    segment_score += 100 * 0.2
            
            segments.append({
                'frame_idx': frame_idx,
                'score': segment_score,
                'frame': segment_frame,
                'description': analysis['description'],
                'text': analysis['text']
            })
        
        prev_frame = frame.copy()
        frame_idx += 1
    
    cap.release()
    
    # Clean up downloaded YouTube video
    if video_path != video_input and os.path.exists(video_path):
        os.remove(video_path)
    
    segments = sorted(segments, key=lambda x: x['score'], reverse=True)
    top_segments = segments[:max_frames]
    
    for i, segment in enumerate(top_segments):
        output_path = os.path.join(output_dir, f"keyframe_{i+1}.jpg")
        cv2.imwrite(output_path, segment['frame'])
        print(f"Saved keyframe {i+1} at frame {segment['frame_idx']} with score {segment['score']:.2f}")
    
    text_output = generate_domain_specific_output(top_segments, domain)
    return [os.path.join(output_dir, f"keyframe_{i+1}.jpg") for i in range(len(top_segments))], text_output

def detect_objects(frame, confidence_threshold=0.5):
    """Detect objects and people in a frame using YOLOv8."""
    if yolo_model is None:
        print("YOLOv8 model not available. Skipping object detection.")
        return frame, []
    
    try:
        # Run YOLOv8 inference
        results = yolo_model(frame, conf=confidence_threshold, imgsz=640, verbose=False)
        
        # Process results
        annotated_frame = frame.copy()
        detected_objects = []
        
        for result in results:
            boxes = result.boxes
            for box in boxes:
                x, y, w, h = box.xywh[0].cpu().numpy()
                x, y = int(x - w/2), int(y - h/2)
                w, h = int(w), int(h)
                confidence = float(box.conf.cpu().numpy())
                class_id = int(box.cls.cpu().numpy())
                label = yolo_model.names[class_id]
                
                # Draw bounding box and label
                color = (0, 255, 0)
                cv2.rectangle(annotated_frame, (x, y), (x + w, y + h), color, 2)
                cv2.putText(annotated_frame, f"{label} {confidence:.2f}", (x, y - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                
                detected_objects.append({
                    "class": label,
                    "confidence": confidence,
                    "box": [x, y, w, h]
                })
        
        return annotated_frame, detected_objects
    except Exception as e:
        print(f"YOLOv8 object detection failed: {str(e)}. Falling back to vision-based analysis.")
        return frame, []

def analyze_motion_patterns(frames, sample_rate=5):
    """Analyze motion patterns across multiple frames."""
    if len(frames) < 2:
        return [], 0
    
    # Initialize optical flow parameters
    feature_params = dict(maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7)
    lk_params = dict(winSize=(15, 15), maxLevel=2,
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
    
    # Convert first frame to grayscale
    old_gray = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)
    
    # Find initial points to track
    p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)
    if p0 is None:
        return [], 0
    
    # Create mask for drawing
    mask = np.zeros_like(frames[0])
    
    # Motion vectors
    motion_vectors = []
    total_motion = 0
    
    # Process frames
    for i in range(1, len(frames), sample_rate):
        if i >= len(frames):
            break
            
        frame = frames[i]
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Calculate optical flow
        p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
        
        # Select good points
        if p1 is not None:
            good_new = p1[st==1]
            good_old = p0[st==1]
            
            # Calculate motion vectors
            frame_vectors = []
            for j, (new, old) in enumerate(zip(good_new, good_old)):
                a, b = new.ravel()
                c, d = old.ravel()
                vector = (a-c, b-d)
                magnitude = np.sqrt((a-c)**2 + (b-d)**2)
                total_motion += magnitude
                frame_vectors.append((vector, magnitude))
                
                # Draw the tracks
                mask = cv2.line(mask, (int(a), int(b)), (int(c), int(d)), (0, 255, 0), 2)
                frame = cv2.circle(frame, (int(a), int(b)), 5, (0, 0, 255), -1)
            
            motion_vectors.append(frame_vectors)
            
            # Update points
            if len(good_new) > 10:
                p0 = good_new.reshape(-1, 1, 2)
                old_gray = frame_gray.copy()
            else:
                # Reset points if too few remain
                p0 = cv2.goodFeaturesToTrack(frame_gray, mask=None, **feature_params)
                old_gray = frame_gray.copy()
    
    # Calculate average motion
    avg_motion = total_motion / len(frames) if frames else 0
    
    return motion_vectors, avg_motion

def process_video_enhanced(video_input: str, domain: str, output_dir: str = "storyboard", max_frames: int = 10) -> tuple:
    """Process a video file or YouTube link with enhanced object detection and motion analysis, optimized for long videos."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Handle YouTube links
    video_path = video_input
    if video_input.startswith("http") and ("youtube.com" in video_input or "youtu.be" in video_input):
        video_path = download_youtube_video(video_input)
        if not video_path:
            return None, "Error: Could not download YouTube video."
    
    # Validate local file
    if not video_path.startswith("http") and not os.path.exists(video_path):
        return None, "Error: Video file does not exist."
    if not video_path.lower().endswith(('.mp4', '.mov')):
        return None, "Error: Only .mp4 and .mov files are supported."
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None, "Error: Could not open video."
    
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    duration = frame_count / fps
    
    print(f"Video info: {frame_count} frames, {fps} FPS, {duration:.2f} seconds")
    
    # Optimize for long videos
    min_sampling_rate = 10  # Sample every 10th frame for very long videos
    max_sampling_rate = 30  # Maximum sampling rate for extremely long videos
    target_samples = 1000   # Target number of samples to process
    sampling_rate = max(min_sampling_rate, min(max_sampling_rate, int(frame_count / target_samples)))
    segment_duration = 10.0  # Analyze segments every 10 seconds for long videos
    frames_per_segment = int(fps * segment_duration)
    
    segments = []
    frames_buffer = []
    prev_frame = None
    frame_idx = 0
    
    print("Analyzing video for key moments...")
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Store frame for motion analysis
        if frame_idx % sampling_rate == 0:
            frames_buffer.append(frame.copy())
        
        if frame_idx % frames_per_segment == 0:
            segment_score = 0
            segment_frame = frame.copy()
            
            # Object detection with YOLOv8
            try:
                annotated_frame, detected_objects = detect_objects(frame)
                person_count = sum(1 for obj in detected_objects if obj["class"] == "person")
                unique_objects = set(obj["class"] for obj in detected_objects)
                object_score = min(100, len(detected_objects) * 10 + person_count * 15)
            except Exception as e:
                print(f"Object detection failed at frame {frame_idx}: {str(e)}")
                annotated_frame = frame
                detected_objects = []
                person_count = 0
                object_score = 0
            
            # Analyze with Groq's LLaMA vision model
            analysis = analyze_frame_with_groq(frame, domain)
            llm_score = analysis['score'] * 0.5
            
            # Motion analysis
            if len(frames_buffer) > 5:
                motion_vectors, avg_motion = analyze_motion_patterns(frames_buffer[-5:])
                motion_score = min(100, avg_motion * 10) * 0.3
            else:
                motion_score = 0
            
            # Scene change detection
            scene_change_score = 0
            if prev_frame is not None:
                if detect_scene_change(prev_frame, frame):
                    scene_change_score = 100 * 0.2
            
            # Combined score
            segment_score = llm_score + object_score * 0.1 + motion_score + scene_change_score
            
            segments.append({
                'frame_idx': frame_idx,
                'score': segment_score,
                'frame': annotated_frame,
                'description': analysis['description'],
                'text': analysis['text'],
                'objects': [obj["class"] for obj in detected_objects],
                'person_count': person_count
            })
            
            # Progress indicator for long videos
            if frame_count > 1000 and frame_idx % (frames_per_segment * 10) == 0:
                print(f"Processing: {frame_idx}/{frame_count} frames ({frame_idx/frame_count*100:.1f}%)")
        
        prev_frame = frame.copy()
        frame_idx += 1
        
        # Limit buffer size to prevent memory issues
        if len(frames_buffer) > 50:
            frames_buffer.pop(0)
    
    cap.release()
    
    # Clean up downloaded YouTube video
    if video_path != video_input and os.path.exists(video_path):
        os.remove(video_path)
    
    if not segments:
        return None, "Error: No significant segments detected in the video."
    
    segments = sorted(segments, key=lambda x: x['score'], reverse=True)
    top_segments = segments[:max_frames]
    
    # Save keyframes and create storyboard
    highlight_frames = []
    for i, segment in enumerate(top_segments):
        output_path = os.path.join(output_dir, f"keyframe_{i+1}.jpg")
        cv2.imwrite(output_path, segment['frame'])
        highlight_frames.append(output_path)
        
        print(f"Saved keyframe {i+1} at frame {segment['frame_idx']} with score {segment['score']:.2f}")
        print(f"  - Objects: {', '.join(segment['objects']) if segment['objects'] else 'None'}")
        print(f"  - People: {segment['person_count']}")
        print(f"  - Description: {segment['description']}")
    
    # Generate content analysis
    text_output = generate_domain_specific_output(top_segments, domain)
    
    # Create side-by-side storyboard image
    if len(highlight_frames) > 1:
        storyboard_path = os.path.join(output_dir, "storyboard.jpg")
        storyboard_images = [cv2.imread(frame) for frame in highlight_frames]
        
        # Resize to common height
        height = 300
        resized_images = []
        for img in storyboard_images:
            if img is None:
                continue
            h, w = img.shape[:2]
            ratio = height / h
            resized = cv2.resize(img, (int(w * ratio), height))
            resized_images.append(resized)
        
        if resized_images:
            # Concatenate horizontally
            storyboard = cv2.hconcat(resized_images)
            cv2.imwrite(storyboard_path, storyboard)
            print(f"Created storyboard image: {storyboard_path}")
    
    # Create highlight reel
    highlight_reel_path = create_highlight_reel(video_input, highlight_frames)
    if highlight_reel_path:
        text_output += f"\n\nCreated highlight reel: {highlight_reel_path}"
    
    return highlight_frames, text_output

def create_highlight_reel(video_input: str, keyframes: list, output_file: str = "storyboard/highlight_reel.mp4"):
    """Create a highlight reel video from the keyframes."""
    if not keyframes:
        return None, "Error: No keyframes provided."
    
    # Ensure output directory exists
    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Extract keyframe numbers from filenames
    frame_indices = []
    for kf in keyframes:
        match = re.search(r'keyframe_(\d+)', os.path.basename(kf))
        if match:
            frame_idx = int(match.group(1)) - 1
            frame_indices.append(frame_idx)
    
    # Handle YouTube links
    video_path = video_input
    if video_input.startswith("http") and ("youtube.com" in video_input or "youtu.be" in video_input):
        video_path = download_youtube_video(video_input)
        if not video_path:
            return None, "Error: Could not download YouTube video."
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None, "Error: Could not open video."
    
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_file, fourcc, fps, (width, height))
    
    # Extract 5-second clips around each keyframe for long videos
    seconds_before = 2.5
    seconds_after = 2.5
    frames_before = int(fps * seconds_before)
    frames_after = int(fps * seconds_after)
    
    for segment_idx, frame_idx in enumerate(frame_indices):
        # Calculate start and end frames
        start_frame = max(0, frame_idx - frames_before)
        end_frame = min(frame_count - 1, frame_idx + frames_after)
        
        # Set video position to start frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        
        # Extract and write frames
        for _ in range(end_frame - start_frame + 1):
            ret, frame = cap.read()
            if not ret:
                break
            
            # Add segment indicator text
            cv2.putText(frame, f"Segment {segment_idx+1}", (30, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            out.write(frame)
    
    cap.release()
    out.release()
    
    # Clean up downloaded YouTube video
    if video_path != video_input and os.path.exists(video_path):
        os.remove(video_path)
    
    return output_file

def process_live_stream(source: str, domain: str, output_dir: str = "live_storyboard", max_frames: int = 5, window_duration: int = 10) -> None:
    """Process a live video stream and generate real-time storyboards."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print("Error: Could not open video stream.")
        return
    
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    window_frames = int(fps * window_duration)
    segments = []
    prev_frame = None
    frame_idx = 0
    window_start = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_idx % int(fps) == 0:
            segment_score = 0
            segment_frame = frame.copy()
            
            # Analyze frame with Groq's LLaMA vision model
            analysis = analyze_frame_with_groq(frame, domain)
            segment_score += analysis['score'] * 0.5
            
            if prev_frame is not None:
                motion_score = calculate_motion_score(prev_frame, frame)
                segment_score += motion_score * 0.3
                
                if detect_scene_change(prev_frame, frame):
                    segment_score += 100 * 0.2
            
            segments.append({
                'frame_idx': frame_idx,
                'score': segment_score,
                'frame': segment_frame,
                'description': analysis['description'],
                'text': analysis['text']
            })
        
        prev_frame = frame.copy()
        frame_idx += 1
        
        if frame_idx - window_start >= window_frames:
            top_segments = sorted(segments, key=lambda x: x['score'], reverse=True)[:max_frames]
            for i, segment in enumerate(top_segments):
                output_path = os.path.join(output_dir, f"keyframe_window_{window_start//window_frames}_{i+1}.jpg")
                cv2.imwrite(output_path, segment['frame'])
                print(f"Saved keyframe {i+1} for window {window_start//window_frames} at frame {segment['frame_idx']}")
            
            text_output = generate_domain_specific_output(top_segments, domain)
            print(f"\n### Live Stream {domain.capitalize()} Summary (Window {window_start//window_frames})\n{text_output}")
            
            segments = []
            window_start = frame_idx
    
    cap.release()

# --- Main Function with EOFError Handling ---
def main():
    print("=== Welcome to KnowledgeForge: AI-Powered Learning & Video Analysis with YOLOv8 ===")
    while True:
        print("\nOptions:")
        print("1. Add web content")
        print("2. Add YouTube video (text/transcript)")
        print("3. Add text")
        print("4. Ask a question")
        print("5. Get text summary")
        print("6. Generate mock test")
        print("7. Generate mind map")
        print("8. Generate flowchart")
        print("9. Analyze video (basic)")
        print("10. Process live stream")
        print("11. Analyze video (enhanced with YOLOv8)")
        print("12. Create highlight reel from analyzed video")
        print("13. Exit")
        try:
            action = input("Choose an option (1-13): ").strip()
        except EOFError:
            print("\nInput stream closed. Exiting...")
            break
        except KeyboardInterrupt:
            print("\nProgram interrupted. Exiting...")
            break

        if action == "1":
            url = input("Enter webpage URL: ")
            process_input(url, "web")
        elif action == "2":
            url = input("Enter YouTube video URL: ")
            process_input(url, "video")
        elif action == "3":
            text = input("Enter text: ")
            process_input(text, "user")
        elif action == "4":
            query = input("Ask your question: ")
            source = input("Optional: Specify a source URL (or press Enter for all content): ").strip() or None
            response = generate_answer(query, source)
            print(f"\n### Question\n{query}\n### Answer\n{response}")
        elif action == "5":
            source = input("Optional: Specify a source URL (or press Enter for all content): ").strip() or None
            summary = generate_summary(source)
            print(f"\n{summary}")
        elif action == "6":
            topic = input("Enter the topic for the mock test: ")
            source = input("Optional: Specify a source URL (or press Enter for all content): ").strip() or None
            test = generate_mock_test(topic, source)
            print(f"\n{test}")
        elif action == "7":
            topic = input("Enter the topic for the mind map: ")
            source = input("Optional: Specify a source URL (or press Enter for all content): ").strip() or None
            result = generate_mind_map(topic, source)
            print(f"\n{result}")
        elif action == "8":
            topic = input("Enter the topic for the flowchart: ")
            source = input("Optional: Specify a source URL (or press Enter for all content): ").strip() or None
            result = generate_flowchart(topic, source)
            print(f"\n{result}")
        elif action == "9":
            video_input = input("Enter video file path or YouTube URL: ")
            domain = input("Enter domain (sports, podcast, news, education, surveillance, generic): ").strip().lower()
            if domain not in ["sports", "podcast", "news", "education", "surveillance", "generic"]:
                print("Invalid domain. Please choose from: sports, podcast, news, education, surveillance, generic")
                continue
            keyframes, text_output = process_video(video_input, domain)
            if keyframes:
                print(f"\n### Video Storyboard for {domain.capitalize()}")
                print("Keyframes saved:", keyframes)
                print(text_output)
            else:
                print(text_output)
        elif action == "10":
            source = input("Enter stream source (e.g., '0' for webcam, RTSP/HTTP URL): ")
            domain = input("Enter domain (sports, podcast, news, education, surveillance, generic): ").strip().lower()
            if domain not in ["sports", "podcast", "news", "education", "surveillance", "generic"]:
                print("Invalid domain. Please choose from: sports, podcast, news, education, surveillance, generic")
                continue
            print("Processing live stream. Press Ctrl+C to stop.")
            try:
                process_live_stream(source, domain)
            except KeyboardInterrupt:
                print("\nLive stream processing stopped.")
        elif action == "11":
            video_input = input("Enter video file path or YouTube URL: ")
            domain = input("Enter domain (sports, podcast, news, education, surveillance, generic): ").strip().lower()
            if domain not in ["sports", "podcast", "news", "education", "surveillance", "generic"]:
                print("Invalid domain. Please choose from: sports, podcast, news, education, surveillance, generic")
                continue
            
            max_frames = input("Enter number of key moments to extract (default: 10): ").strip()
            max_frames = int(max_frames) if max_frames.isdigit() and int(max_frames) > 0 else 10
            
            output_dir = input("Enter output directory (default: 'storyboard'): ").strip() or "storyboard"
            
            print("\nProcessing video with enhanced visual analysis using YOLOv8...")
            keyframes, text_output = process_video_enhanced(video_input, domain, output_dir, max_frames)
            
            if keyframes:
                print(f"\n### Enhanced Video Analysis for {domain.capitalize()}")
                print("Keyframes saved:", keyframes)
                print(text_output)
                
                # Save the latest keyframes for highlight reel creation
                with open("latest_keyframes.json", "w") as f:
                    json.dump({"video": video_input, "keyframes": keyframes}, f)
            else:
                print(text_output)
        elif action == "12":
            try:
                with open("latest_keyframes.json", "r") as f:
                    data = json.load(f)
                    video_input = data["video"]
                    keyframes = data["keyframes"]
                
                if not keyframes:
                    print("No keyframes available. Please analyze a video first.")
                    continue
                
                output_file = input("Enter output filename (default: 'storyboard/highlight_reel.mp4'): ").strip() or "storyboard/highlight_reel.mp4"
                
                print("\nCreating highlight reel...")
                result = create_highlight_reel(video_input, keyframes, output_file)
                if result:
                    print(f"Created highlight reel: {result}")
                else:
                    print("Error creating highlight reel.")
            except FileNotFoundError:
                print("No analyzed video found. Please run the enhanced video analysis first.")
            except Exception as e:
                print(f"Error creating highlight reel: {str(e)}")
        elif action == "13":
            print("Goodbye!")
            break
        else:
            print("Invalid option. Please choose 1-13.")

if __name__ == "__main__":
    main()
    