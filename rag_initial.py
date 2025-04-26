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
import cv2
import os
import base64
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS

# Configure Gemini API
GEMINI_API_KEY = "AIzaSyDq3W6bcmtED-s0vDKmSBZr8uIwy4Gc1Io"
genai.configure(api_key=GEMINI_API_KEY)

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Initialize models
embedder = SentenceTransformer('all-MiniLM-L6-v2')

# FAISS setup
dimension = 384
index = faiss.IndexFlatL2(dimension)
knowledge_base: List[Dict] = []
vector_base = []

# Store sources
sources = []

# Initialize crawler
crawler = WebCrawler()
crawler.warmup()
extraction_strategy = JsonCssExtractionStrategy(
    schema={"text_content": {"css": "p", "type": "string"}}
)

def crawl_url(url: str, output_file: str = "crawled_data.json") -> str:
    try:
        options = Options()
        options.add_argument("--headless=new")
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
    if source not in sources:
        sources.append(source)
    print(f"Added {len(chunks)} chunks from {source} ({source_type}) to knowledge base.")

def process_input(source: str, source_type: str = "web"):
    if source_type == "web":
        text = crawl_url(source)
        if text:
            update_knowledge_base(text, source, "web")
            return True
    elif source_type == "video":
        page_text = crawl_youtube(source)
        transcript = get_youtube_transcript(source)
        text = f"{page_text or ''} {transcript or ''}".strip()
        if text:
            update_knowledge_base(text, source, "video")
            return True
    elif source_type == "user":
        update_knowledge_base(source, "User Input", "text")
        return True
    return False

def retrieve_chunks(query: str, source: str = None) -> List[str]:
    if not knowledge_base:
        return ["No knowledge base available."]
    
    query_embedding = embedder.encode([query], convert_to_numpy=True)
    distances, indices = index.search(query_embedding, k=3)
    
    if source:
        source_chunks = [kb["text"] for kb in knowledge_base if kb["source"] == source]
        if source_chunks:
            return source_chunks[:3]
        return ["No content available for this source."]
    else:
        if len(indices[0]) > 0:
            return [knowledge_base[i]["text"] for i in indices[0] if i < len(knowledge_base)]
        return ["No relevant content found."]

def generate_answer(query: str, source: str = None) -> str:
    context = "\n".join(retrieve_chunks(query, source))
    prompt = f"""
    You are an expert educational assistant. Based on the following context, provide a clear, concise, and accurate answer to the question.

    Context: {context}
    Question: {query}
    Answer:
    """
    model = genai.GenerativeModel('gemini-1.5-flash')
    response = model.generate_content(prompt)
    return response.text.strip()

def generate_summary(source: str = None) -> Dict:
    if not knowledge_base:
        return {"title": "No Content Available", "content": "Add some content to get a summary."}
    
    if source:
        context = "\n".join([kb["text"] for kb in knowledge_base if kb["source"] == source][:3])
        title = f"Summary for {source}"
    else:
        context = "\n".join([kb["text"] for kb in knowledge_base][:3])
        title = "Summary of All Content"
    
    prompt = f"""
    Provide a concise and informative summary of the following content in 3-5 paragraphs.

    Content: {context}
    Summary:
    """
    model = genai.GenerativeModel('gemini-1.5-flash')
    response = model.generate_content(prompt)
    return {"title": title, "content": response.text.strip()}

def generate_mock_test(topic: str, source: str = None, num_questions: int = 5) -> Dict:
    context = "\n".join(retrieve_chunks(topic, source))
    prompt = f"""
    Create a mock test with {num_questions} multiple-choice questions on the topic '{topic}'. Each question should have 4 options (A, B, C, D) and clearly mark the correct answer.

    Context: {context}
    Mock Test:
    """
    model = genai.GenerativeModel('gemini-1.5-flash')
    response = model.generate_content(prompt)
    return {"title": f"Mock Test on {topic}", "content": response.text.strip()}

def generate_mind_map(topic: str, source: str = None) -> Dict:
    context = "\n".join(retrieve_chunks(topic, source))
    prompt = f"""
    Create a mermaid.js mind map diagram for the topic '{topic}'. Use the mindmap syntax.

    Context: {context}
    Mermaid Diagram:
    """
    model = genai.GenerativeModel('gemini-1.5-flash')
    response = model.generate_content(prompt)
    mermaid_content = response.text.strip()
    
    # Clean up the mermaid content
    if '```mermaid' in mermaid_content:
        mermaid_content = mermaid_content.split('```mermaid')[1].split('```')[0].strip()
    
    # Create a unique filename
    filename = f"mindmap_{topic.replace(' ', '_')}.mmd"
    
    # Save the mermaid content to a file
    with open(f"static/{filename}", "w") as f:
        f.write(mermaid_content)
    
    return {"title": f"Mind Map: {topic}", "mermaid": mermaid_content}

def generate_flowchart(topic: str, source: str = None) -> Dict:
    context = "\n".join(retrieve_chunks(topic, source))
    prompt = f"""
    Create a mermaid.js flowchart diagram for the process or concept '{topic}'. Use the flowchart syntax.

    Context: {context}
    Mermaid Diagram:
    """
    model = genai.GenerativeModel('gemini-1.5-flash')
    response = model.generate_content(prompt)
    mermaid_content = response.text.strip()
    
    # Clean up the mermaid content
    if '```mermaid' in mermaid_content:
        mermaid_content = mermaid_content.split('```mermaid')[1].split('```')[0].strip()
    
    # Create a unique filename
    filename = f"flowchart_{topic.replace(' ', '_')}.mmd"
    
    # Save the mermaid content to a file
    with open(f"static/{filename}", "w") as f:
        f.write(mermaid_content)
    
    return {"title": f"Flowchart: {topic}", "mermaid": mermaid_content}

def generate_storyboard(video_source: str) -> Dict:
    """
    Generate a storyboard for the video content in the knowledge base
    """
    if not any(kb["type"] == "video" for kb in knowledge_base):
        return {"success": False, "message": "No video content in the knowledge base."}
    
    # Find the video source if not specified
    if not video_source:
        for kb in knowledge_base:
            if kb["type"] == "video":
                video_source = kb["source"]
                break
    
    # Generate description for key points from the video
    context = "\n".join([kb["text"] for kb in knowledge_base if kb["source"] == video_source][:5])
    prompt = f"""
    Create a visual storyboard for the video content. Identify 5 key points or scenes from the content and describe them.
    For each scene, provide:
    1. A short title
    2. A brief description (1-2 sentences)
    3. A visual description (what would be shown in this frame)

    Context from video: {context}
    Storyboard:
    """
    model = genai.GenerativeModel('gemini-1.5-flash')
    response = model.generate_content(prompt)
    
    # Create a unique filename base
    video_id = video_source.split("v=")[-1].split("&")[0] if "youtube.com" in video_source else "video"
    filename_base = f"storyboard_{video_id}"
    
    return {"title": f"Storyboard for {video_source}", "content": response.text.strip()}

# API Routes
@app.route('/api/get-sources', methods=['GET'])
def api_get_sources():
    return jsonify({"success": True, "sources": sources})

@app.route('/api/add-web-content', methods=['POST'])
def api_add_web_content():
    data = request.get_json()
    url = data.get('url')
    if not url:
        return jsonify({"success": False, "message": "No URL provided"})
    
    success = process_input(url, source_type="web")
    return jsonify({"success": success, "message": "Web content added successfully" if success else "Failed to add web content"})

@app.route('/api/add-video', methods=['POST'])
def api_add_video():
    data = request.get_json()
    url = data.get('url')
    if not url:
        return jsonify({"success": False, "message": "No URL provided"})
    
    success = process_input(url, source_type="video")
    return jsonify({"success": success, "message": "Video content added successfully" if success else "Failed to add video content"})

@app.route('/api/add-text', methods=['POST'])
def api_add_text():
    data = request.get_json()
    text = data.get('text')
    if not text:
        return jsonify({"success": False, "message": "No text provided"})
    
    success = process_input(text, source_type="user")
    return jsonify({"success": success, "message": "Text content added successfully" if success else "Failed to add text content"})

@app.route('/api/ask-question', methods=['POST'])
def api_ask_question():
    data = request.get_json()
    question = data.get('question')
    source = data.get('source')
    if not question:
        return jsonify({"success": False, "message": "No question provided"})
    
    answer = generate_answer(question, source)
    return jsonify({"success": True, "question": question, "answer": answer})

@app.route('/api/get-summary', methods=['POST'])
def api_get_summary():
    data = request.get_json()
    source = data.get('source')
    summary = generate_summary(source)
    return jsonify({"success": True, "summary": summary})

@app.route('/api/generate-mock-test', methods=['POST'])
def api_generate_mock_test():
    data = request.get_json()
    topic = data.get('topic')
    source = data.get('source')
    if not topic:
        return jsonify({"success": False, "message": "No topic provided"})
    
    mock_test = generate_mock_test(topic, source)
    return jsonify({"success": True, "mock_test": mock_test})

@app.route('/api/generate-mind-map', methods=['POST'])
def api_generate_mind_map():
    data = request.get_json()
    topic = data.get('topic')
    source = data.get('source')
    if not topic:
        return jsonify({"success": False, "message": "No topic provided"})
    
    mind_map = generate_mind_map(topic, source)
    return jsonify({"success": True, "title": mind_map["title"], "mermaid": mind_map["mermaid"]})

@app.route('/api/generate-flowchart', methods=['POST'])
def api_generate_flowchart():
    data = request.get_json()
    topic = data.get('topic')
    source = data.get('source')
    if not topic:
        return jsonify({"success": False, "message": "No topic provided"})
    
    flowchart = generate_flowchart(topic, source)
    return jsonify({"success": True, "title": flowchart["title"], "mermaid": flowchart["mermaid"]})

@app.route('/api/generate-storyboard', methods=['POST'])
def api_generate_storyboard():
    data = request.get_json()
    video_source = data.get('source')
    
    storyboard = generate_storyboard(video_source)
    return jsonify({"success": True, "title": storyboard["title"], "content": storyboard["content"]})

@app.route('/static/<path:filename>')
def serve_static(filename):
    return send_file(f"static/{filename}")

if __name__ == "__main__":
    # Create static directory if it doesn't exist
    os.makedirs("static", exist_ok=True)
    app.run(host="0.0.0.0", port=5000)
