import elevenlabs  # For high-quality voice synthesis
from pydub.effects import speedup, normalize
from scipy import signal
import re
import numpy as np
import os
import matplotlib.pyplot as plt
from PIL import Image
import speech_recognition as sr
from moviepy.editor import VideoFileClip, AudioFileClip, TextClip, CompositeVideoClip, concatenate_videoclips, ImageClip
import pytesseract
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import tempfile
import textwrap
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import base64
import requests
import json
from io import BytesIO
import gtts
from pydub import AudioSegment
import time
import cv2
import librosa
import soundfile as sf

# Set ElevenLabs API key
elevenlabs.api_key = "sk_caf3c55a9cd7d5211cbde758b98c15241c13836e90328bf3"

class VideoSummarizer:
    def __init__(self, video_path, api_key, model_provider="gemini"):
        """Initialize the video summarizer with the path to the video file.
        
        Args:
            video_path: Path to the video file
            api_key: API key for Gemini or Groq
            model_provider: Either "gemini" or "groq"
        """
        self.video_path = video_path
        self.frames = []
        self.keyframes = []
        self.text_content = ""
        self.visual_summary = ""
        self.combined_summary = ""
        self.storyboard_frames = []
        
        # Store API parameters
        self.api_key = api_key
        self.model_provider = model_provider.lower()
        
        # Set up API endpoints and model names based on provider
        if self.model_provider == "gemini":
            self.api_url = "https://generativelanguage.googleapis.com/v1/models/gemini-1.5-flash:generateContent"
            self.text_api_url = "https://generativelanguage.googleapis.com/v1/models/gemini-1.5-flash:generateContent"
            self.model_name = "gemini-1.5-flash"  # Using Gemini 2.0 Flash
            self.text_model_name = "gemini-1.5-flash"
        elif self.model_provider == "groq":
            self.api_url = "https://api.groq.com/openai/v1/chat/completions" 
            self.text_api_url = "https://api.groq.com/openai/v1/chat/completions"
            self.model_name = "llama3-70b-8192"
            self.text_model_name = "llama3-70b-8192"
        else:
            raise ValueError("Model provider must be either 'gemini' or 'groq'")
        
    def extract_video_segments(self, segment_duration=3):
        """Extract dynamic video segments around each keyframe.
        
        Args:
            segment_duration: Duration in seconds for each segment
            
        Returns:
            List of (start_time, end_time) tuples for each segment
        """
        if not self.storyboard_frames:
            self.select_keyframes()
        
        # Get timestamps of keyframes
        keyframe_timestamps = self._get_keyframe_timestamps()
        
        # Calculate segment boundaries around each keyframe
        segments = []
        video = VideoFileClip(self.video_path)
        video_duration = video.duration
        
        for timestamp in keyframe_timestamps:
            # Calculate start and end times for segment
            start_time = max(0, timestamp - segment_duration / 2)
            end_time = min(video_duration, timestamp + segment_duration / 2)
            
            # Ensure minimum duration
            if end_time - start_time < 1.5:
                end_time = min(video_duration, start_time + 1.5)
            
            segments.append((start_time, end_time))
        
        return segments

    def extract_frames(self, sample_rate=1):
        """Extract frames from the video at the given sample rate (in seconds)."""
        cap = cv2.VideoCapture(self.video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_interval = int(fps * sample_rate)
        
        current_frame = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            if current_frame % frame_interval == 0:
                self.frames.append(frame)
                
            current_frame += 1
            
        cap.release()
        print(f"Extracted {len(self.frames)} frames from video")
        return self.frames
    
    def select_keyframes(self, min_clusters=5, max_clusters=None):
        """Dynamic scene detection based on video content complexity.
        
        Args:
            min_clusters: Minimum number of scenes to detect
            max_clusters: Maximum number of scenes to detect (None for unlimited)
            
        Returns:
            List of key frames
        """
        if not self.frames:
            self.extract_frames()
            
        # Calculate frame differences to detect scene transitions
        frame_diffs = []
        for i in range(1, len(self.frames)):
            diff = np.mean(np.abs(self.frames[i].astype(float) - self.frames[i-1].astype(float)))
            frame_diffs.append(diff)
        
        # Normalize differences
        frame_diffs = np.array(frame_diffs)
        if np.max(frame_diffs) > np.min(frame_diffs):
            frame_diffs = (frame_diffs - np.min(frame_diffs)) / (np.max(frame_diffs) - np.min(frame_diffs))
        
        # Find scene transitions using a dynamic threshold
        scene_changes = []
        window_size = 5
        for i in range(window_size, len(frame_diffs) - window_size):
            window = frame_diffs[i-window_size:i+window_size]
            if frame_diffs[i] > np.mean(window) + 2 * np.std(window):
                scene_changes.append(i+1)  # +1 because frame_diffs is offset by 1
        
        # Ensure we have at least min_clusters scene changes
        if len(scene_changes) < min_clusters:
            # Use KMeans fallback with dynamic cluster count
            # Convert frames to feature vectors
            features = []
            for frame in self.frames:
                small_frame = cv2.resize(frame, (32, 32))
                feature = small_frame.flatten().astype(np.float32)
                features.append(feature)
                
            features = np.array(features)
            
            # Determine optimal cluster count using silhouette score
            best_score = -1
            best_k = min_clusters
            
            # If max_clusters is None, use a dynamic approach to find optimal clusters
            if max_clusters is None:
                # Start with a reasonable upper limit based on video length
                # More frames = more potential scenes
                upper_limit = min(100, max(min_clusters, len(self.frames) // 10))
                
                # Try different cluster counts to find the optimal one
                for k in range(min_clusters, upper_limit + 1):
                    if len(self.frames) <= k:
                        break
                    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                    labels = kmeans.fit_predict(features)
                    
                    # Calculate silhouette score if we have enough samples
                    if len(set(labels)) > 1:
                        try:
                            score = silhouette_score(features, labels)
                            if score > best_score:
                                best_score = score
                                best_k = k
                        except Exception:
                            # If silhouette score fails, use inertia (lower is better)
                            inertia = kmeans.inertia_
                            if best_score == -1 or inertia < best_score:
                                best_score = inertia
                                best_k = k
            else:
                # Use the provided max_clusters
                for k in range(min_clusters, min(max_clusters + 1, len(self.frames) // 2)):
                    if len(self.frames) <= k:
                        break
                    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                    labels = kmeans.fit_predict(features)
                    score = silhouette_score(features, labels) if len(set(labels)) > 1 else -1
                    if score > best_score:
                        best_score = score
                        best_k = k
            
            # Apply KMeans with optimal cluster count
            kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
            labels = kmeans.fit_predict(features)
            
            # Select frames closest to the cluster centers
            self.keyframes = []
            self.storyboard_frames = []
            frame_indices = []
            
            for i in range(best_k):
                cluster_indices = np.where(labels == i)[0]
                if len(cluster_indices) > 0:
                    cluster_features = features[cluster_indices]
                    center = kmeans.cluster_centers_[i]
                    distances = np.linalg.norm(cluster_features - center, axis=1)
                    closest_index_in_cluster = np.argmin(distances)
                    original_index = cluster_indices[closest_index_in_cluster]
                    
                    self.keyframes.append(self.frames[original_index])
                    frame_indices.append((original_index, self.frames[original_index]))
        else:
            # Use actual scene changes
            scene_changes = sorted(scene_changes)
            
            # Limit to max_clusters if needed
            if max_clusters is not None and len(scene_changes) > max_clusters:
                # Take evenly spaced samples
                indices = np.round(np.linspace(0, len(scene_changes) - 1, max_clusters)).astype(int)
                scene_changes = [scene_changes[i] for i in indices]
            
            # Get keyframes from scene changes
            self.keyframes = [self.frames[i] for i in scene_changes if i < len(self.frames)]
            frame_indices = [(i, self.frames[i]) for i in scene_changes if i < len(self.frames)]
        
        # Sort storyboard frames by their order in the video
        frame_indices.sort(key=lambda x: x[0])
        self.storyboard_frames = [frame for _, frame in frame_indices]
        
        print(f"Detected {len(self.keyframes)} scenes dynamically based on video content")
        return self.keyframes
    
    def extract_text_from_frames(self):
        """Extract any visible text from key frames using OCR."""
        if not self.keyframes:
            self.select_keyframes()
            
        all_text = []
        for frame in self.keyframes:
            # Convert to grayscale for better OCR
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # Apply some preprocessing to improve OCR
            gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
            
            # Extract text using pytesseract
            text = pytesseract.image_to_string(gray)
            if text.strip():
                all_text.append(text.strip())
                
        ocr_text = " ".join(all_text)
        print(f"Extracted {len(ocr_text)} characters of text from frames")
        
        # Add to the text content
        if ocr_text:
            self.text_content += "Visual Text Content:\n" + ocr_text + "\n\n"
        
        return ocr_text
    
    def extract_speech(self):
        """Extract speech from the video and convert it to text."""
        # Extract audio from video
        temp_dir = tempfile.mkdtemp()
        temp_audio = os.path.join(temp_dir, "temp_audio.wav")
        
        try:
            video = VideoFileClip(self.video_path)
            video.audio.write_audiofile(temp_audio, verbose=False, logger=None)
            
            # Use speech recognition to convert audio to text
            recognizer = sr.Recognizer()
            
            # Add retry mechanism for connection issues
            max_retries = 3
            retry_count = 0
            text = ""
            
            # Check if PocketSphinx is available
            sphinx_available = False
            try:
                import pocketsphinx
                sphinx_available = True
            except ImportError:
                print("PocketSphinx not available, will use Google Speech Recognition only")
            
            while retry_count < max_retries:
                try:
                    with sr.AudioFile(temp_audio) as source:
                        # Adjust for ambient noise to improve recognition
                        recognizer.adjust_for_ambient_noise(source, duration=0.5)
                        audio_data = recognizer.record(source)
                        
                        # Try multiple recognition services
                        try:
                            # First try Google's service
                            text = recognizer.recognize_google(audio_data)
                            break
                        except sr.RequestError:
                            # If Google fails and Sphinx is available, try Sphinx (offline)
                            if sphinx_available:
                                try:
                                    text = recognizer.recognize_sphinx(audio_data)
                                    break
                                except Exception as sphinx_error:
                                    print(f"Sphinx recognition failed: {sphinx_error}")
                                    retry_count += 1
                                    time.sleep(1)  # Wait before retrying
                            else:
                                retry_count += 1
                                time.sleep(1)  # Wait before retrying
                
                except (sr.UnknownValueError, sr.RequestError, ConnectionError, BrokenPipeError) as e:
                    print(f"Speech recognition attempt {retry_count + 1} failed: {e}")
                    retry_count += 1
                    time.sleep(1)  # Wait before retrying
                
                except Exception as e:
                    print(f"Unexpected error in speech recognition: {e}")
                    retry_count += 1
                    time.sleep(1)  # Wait before retrying
            
            if text:
                self.text_content += "Speech Content:\n" + text + "\n\n"
                print(f"Extracted {len(text)} characters of speech")
                return text
            else:
                print("Could not extract speech after multiple attempts")
                return ""
                    
        except Exception as e:
            print(f"Error in speech extraction: {e}")
        finally:
            # Clean up
            try:
                if os.path.exists(temp_audio):
                    os.remove(temp_audio)
                if os.path.exists(temp_dir):
                    os.rmdir(temp_dir)
            except Exception as cleanup_error:
                print(f"Error during cleanup: {cleanup_error}")
                
        return ""
    
    def analyze_visual_content(self):
        """Analyze visual content from key frames with improved context.
        Ensures more accurate understanding of the video's actual content.
        """
        if not self.keyframes:
            self.select_keyframes()
        
        # Extract video metadata
        video = VideoFileClip(self.video_path)
        video_duration = video.duration
        
        # Get timestamps for keyframes
        keyframe_timestamps = self._get_keyframe_timestamps()
        
        # Generate captions for each keyframe with timestamp context
        captions = []
        for i, (frame, timestamp) in enumerate(zip(self.keyframes, keyframe_timestamps)):
            # Convert from BGR to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Convert to PIL Image
            pil_img = Image.fromarray(rgb_frame)
            
            # Calculate timestamp percentage through video
            time_percent = (timestamp / video_duration) * 100
            
            # Create a more contextual prompt
            prompt = f"""
            This is frame {i+1} of {len(self.keyframes)} from a video, captured at {timestamp:.1f} seconds 
            ({time_percent:.1f}% through the video).
            
            Describe what's happening in this specific scene from the video in detail.
            Focus on:
            1. The main action or information being presented
            2. Any people, objects, or visual elements that are central to understanding
            3. The purpose or message this scene is trying to convey
            4. How this scene might connect to the overall narrative of the video
            
            Be specific and avoid generic descriptions. Explain what makes this scene important.
            """
            
            # Get caption for the image
            caption = self._get_image_caption(pil_img, prompt)
            
            # Store with timestamp
            captions.append(f"Scene {i+1} [{timestamp:.1f}s]: {caption}")
        
        # Analyze the sequence of scenes for overall narrative
        scenes_text = "\n".join(captions)
        narrative_prompt = f"""
        Below are descriptions of {len(self.keyframes)} key scenes from a video in chronological order:
        
        {scenes_text}
        
        Based only on these scene descriptions, provide a brief analysis of:
        1. What type of video this appears to be (educational, entertainment, tutorial, etc.)
        2. The main topic or subject matter of the video
        3. The apparent narrative structure or flow of information
        
        Keep your analysis concise but specific to the actual content described.
        """
        
        narrative_analysis = self._get_text_completion(narrative_prompt)
        
        self.visual_summary = f"Visual Content Analysis:\n\nScene Breakdown:\n{scenes_text}\n\nOverall Narrative Analysis:\n{narrative_analysis}"
        print("Enhanced visual content analysis complete with narrative understanding")
        return self.visual_summary
        
    def _get_image_caption(self, image, prompt):
        """Get a caption for an image using either Gemini or Groq.
        
        Args:
            image: PIL Image
            prompt: Prompt to guide the image captioning
            
        Returns:
            String caption for the image
        """
        # Convert image to base64
        buffered = BytesIO()
        image.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
        
        try:
            if self.model_provider == "gemini":
                # Create request payload for Gemini
                payload = {
                    "contents": [{
                        "parts": [
                            {"text": prompt},
                            {
                                "inline_data": {
                                    "mime_type": "image/jpeg",
                                    "data": img_str
                                }
                            }
                        ]
                    }]
                }
                
                # Make API call to Gemini
                headers = {
                    "Content-Type": "application/json",
                    "x-goog-api-key": self.api_key
                }
                
                response = requests.post(self.api_url, headers=headers, json=payload)
                response_json = response.json()
                
                # Extract caption from response
                if "candidates" in response_json and len(response_json["candidates"]) > 0:
                    return response_json["candidates"][0]["content"]["parts"][0]["text"]
                else:
                    print("Warning: Could not get caption from Gemini API")
                    return "No description available"
                    
            elif self.model_provider == "groq":
                # For Groq, we'll use their GPT-4 Vision compatible API
                # Convert to base64 data URL format
                img_data_url = f"data:image/jpeg;base64,{img_str}"
                
                # Create request payload
                payload = {
                    "model": self.model_name,
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": prompt},
                                {"type": "image_url", "image_url": {"url": img_data_url}}
                            ]
                        }
                    ],
                    "max_tokens": 300
                }
                
                # Make API call to Groq
                headers = {
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {self.api_key}"
                }
                
                response = requests.post(self.api_url, headers=headers, json=payload)
                response_json = response.json()
                
                # Extract caption from response
                if "choices" in response_json and len(response_json["choices"]) > 0:
                    return response_json["choices"][0]["message"]["content"]
                else:
                    print("Warning: Could not get caption from Groq API")
                    return "No description available"
                    
        except Exception as e:
            print(f"Error getting image caption: {e}")
            return "Error getting image description"
    
    def generate_summary(self):
        """Generate a more accurate summary based on actual video content."""
        # Make sure we've extracted all necessary components
        if not self.keyframes:
            self.select_keyframes()
        
        if not self.visual_summary:
            self.analyze_visual_content()
            
        # Extract text from frames
        if "Visual Text Content" not in self.text_content:
            self.extract_text_from_frames()
            
        # Extract speech
        if "Speech Content" not in self.text_content:
            self.extract_speech()
        
        # Get video metadata for context
        video = VideoFileClip(self.video_path)
        video_duration = video.duration
        keyframe_timestamps = self._get_keyframe_timestamps()
        
        # Create enhanced context for LLM
        video_context = {
            "duration": f"{video_duration:.1f} seconds",
            "scene_count": len(self.keyframes),
            "has_speech": "Speech Content" in self.text_content and len(self.text_content) > 20,
            "has_text": "Visual Text Content" in self.text_content and len(self.text_content) > 20,
            "scene_timestamps": [f"{t:.1f}s" for t in keyframe_timestamps]
        }
        
        # Combine all information
        full_content = self.visual_summary + "\n\n" + self.text_content
        
        # Use the LLM to generate a precisely targeted script
        prompt = f"""
        You are creating an accurate, content-focused narration script for a video summary. 
        
        Video metadata:
        - Duration: {video_context['duration']}
        - Number of scenes: {video_context['scene_count']}
        - Scene timestamps: {', '.join(video_context['scene_timestamps'])}
        - Contains speech: {video_context['has_speech']}
        - Contains visible text: {video_context['has_text']}
        
        Below is information extracted from the video, including descriptions of key scenes, text visible in the video, and speech content.
        
        {full_content}
        
        Your task is to write a narration script that:
        
        1. ACCURATELY describes what is actually in this specific video (not generic content)
        2. Follows the video's structure and flow precisely
        3. Focuses on the most important information or story being presented
        4. Is written in clear, conversational language suitable for narration
        5. Consists of approximately 200-250 words (~90 seconds of narration)
        6. NEVER uses generic phrases like "in this video we can see" or "this video shows"
        7. Sounds natural and engaging when read aloud
        8. Is divided into timed segments that match the key scenes (timestamp each paragraph)
        9. Covers the ACTUAL CONTENT of the video, not just descriptions of what's visible
        
        Write a precise narration script that would help viewers understand exactly what this specific video is about.
        """
        
        self.combined_summary = self._get_text_completion(prompt)
        print("Generated content-accurate narration script based on video analysis")
        return self.combined_summary
        
    def _get_text_completion(self, prompt):
        """Get text completion using either Gemini or Groq with improved parameters.
        
        Args:
            prompt: Prompt to send to the API
            
        Returns:
            String response from the API
        """
        try:
            if self.model_provider == "gemini":
                # Create request payload for Gemini text model with improved parameters
                payload = {
                    "contents": [{
                        "parts": [
                            {"text": prompt}
                        ]
                    }],
                    "generationConfig": {
                        "temperature": 0.7,
                        "topP": 0.95,
                        "topK": 40,
                        "maxOutputTokens": 1024
                    }
                }
                
                # Make API call to Gemini
                headers = {
                    "Content-Type": "application/json",
                    "x-goog-api-key": self.api_key
                }
                
                response = requests.post(self.text_api_url, headers=headers, json=payload)
                response_json = response.json()
                
                # Extract text from response
                if "candidates" in response_json and len(response_json["candidates"]) > 0:
                    return response_json["candidates"][0]["content"]["parts"][0]["text"]
                else:
                    print("Warning: Could not get text completion from Gemini API")
                    return "Error generating summary"
                    
            elif self.model_provider == "groq":
                # Create request payload for Groq with improved parameters
                payload = {
                    "model": self.text_model_name,
                    "messages": [
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ],
                    "max_tokens": 1024,
                    "temperature": 0.7,
                    "top_p": 0.95,
                    "frequency_penalty": 0.5,
                    "presence_penalty": 0.5
                }
                
                # Make API call to Groq
                headers = {
                    "Content-Type": "application/json", 
                    "Authorization": f"Bearer {self.api_key}"
                }
                
                response = requests.post(self.text_api_url, headers=headers, json=payload)
                response_json = response.json()
                
                # Extract text from response
                if "choices" in response_json and len(response_json["choices"]) > 0:
                    return response_json["choices"][0]["message"]["content"]
                else:
                    print("Warning: Could not get text completion from Groq API")
                    return "Error generating summary"
                    
        except Exception as e:
            print(f"Error getting text completion: {e}")
            return f"Error generating summary: {str(e)}"
    
    def create_storyboard_image(self, output_path):
        """Create a storyboard image from key frames."""
        if not self.storyboard_frames:
            self.select_keyframes()
            
        num_frames = len(self.storyboard_frames)
        if num_frames == 0:
            print("No frames available for storyboard")
            return None
            
        # Determine grid size
        grid_size = int(np.ceil(np.sqrt(num_frames)))
        
        # Create a figure with subplots
        fig, axes = plt.subplots(grid_size, grid_size, figsize=(15, 15))
        
        # Flatten axes array for easier indexing
        axes = axes.flatten() if hasattr(axes, 'flatten') else [axes]
        
        # Add frames to the grid
        for i, frame in enumerate(self.storyboard_frames):
            if i < len(axes):
                # Convert BGR to RGB for matplotlib
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                axes[i].imshow(rgb_frame)
                axes[i].set_title(f"Scene {i+1}")
                axes[i].axis('off')
        
        # Hide unused subplots
        for i in range(num_frames, len(axes)):
            axes[i].axis('off')
            
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()
        print(f"Storyboard saved to {output_path}")
        return output_path
    
    def create_storyboard_pdf(self, output_path):
        """Create a PDF storyboard with frames and captions."""
        if not self.storyboard_frames:
            self.select_keyframes()
            
        if not self.combined_summary:
            self.generate_summary()
            
        # Create a new PDF
        c = canvas.Canvas(output_path, pagesize=letter)
        width, height = letter
        
        # Add title
        c.setFont("Helvetica-Bold", 16)
        c.drawString(50, height - 50, "Video Summary Storyboard")
        
        # Add summary
        c.setFont("Helvetica", 10)
        y_position = height - 80
        
        # Wrap text to fit page width
        wrapped_text = textwrap.fill(self.combined_summary, width=100)
        for line in wrapped_text.split('\n'):
            c.drawString(50, y_position, line)
            y_position -= 15
            
        # Add some space
        y_position -= 20
        
        # Calculate frame size and positions
        frame_width = (width - 100) / 2
        frame_height = frame_width * 9 / 16  # Assuming 16:9 aspect ratio
        
        # Add frames to PDF
        for i, frame in enumerate(self.storyboard_frames):
            # Check if we need a new page
            if y_position < 100:
                c.showPage()
                y_position = height - 50
                
            # Calculate x position (2 columns)
            x_position = 50 if i % 2 == 0 else 50 + frame_width + 20
            
            # Start a new row after every 2 frames
            if i % 2 == 0 and i > 0:
                y_position -= frame_height + 40
                
            # Save frame as temporary image
            temp_image = f"temp_frame_{i}.jpg"
            cv2.imwrite(temp_image, frame)
            
            # Add frame to PDF
            c.drawImage(temp_image, x_position, y_position - frame_height, width=frame_width, height=frame_height)
            
            # Add caption below frame
            c.setFont("Helvetica", 8)
            c.drawString(x_position, y_position - frame_height - 15, f"Scene {i+1}")
            
            # Delete temporary image
            os.remove(temp_image)
            
        # Save PDF
        c.save()
        print(f"PDF storyboard saved to {output_path}")
        return output_path
    
    def process_video(self, output_dir="./output"):
        """Process the video and generate all outputs with progress reporting."""
        # Create output directory if it doesn't exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        # Generate base filename from input video
        base_name = os.path.splitext(os.path.basename(self.video_path))[0]
        
        # Extract all information with progress reporting
        print("Step 1/7: Extracting and selecting keyframes...")
        start_time = time.time()
        self.select_keyframes(min_clusters=10, max_clusters=None)  # Use dynamic clustering
        print(f"Completed in {time.time() - start_time:.2f} seconds")
        
        print("Step 2/7: Extracting text from frames...")
        start_time = time.time()
        self.extract_text_from_frames()
        print(f"Completed in {time.time() - start_time:.2f} seconds")
        
        print("Step 3/7: Extracting speech from audio...")
        start_time = time.time()
        self.extract_speech()
        print(f"Completed in {time.time() - start_time:.2f} seconds")
        
        print("Step 4/7: Analyzing visual content...")
        start_time = time.time()
        self.analyze_visual_content()
        print(f"Completed in {time.time() - start_time:.2f} seconds")
        
        print("Step 5/7: Generating summary...")
        start_time = time.time()
        self.generate_summary()
        print(f"Completed in {time.time() - start_time:.2f} seconds")
        
        # Generate outputs
        print("Step 6/7: Generating storyboards and text summary...")
        start_time = time.time()
        summary_path = os.path.join(output_dir, f"{base_name}_summary.txt")
        with open(summary_path, 'w') as f:
            f.write(self.combined_summary)
            
        storyboard_img_path = os.path.join(output_dir, f"{base_name}_storyboard.png")
        self.create_storyboard_image(storyboard_img_path)
        
        storyboard_pdf_path = os.path.join(output_dir, f"{base_name}_storyboard.pdf")
        self.create_storyboard_pdf(storyboard_pdf_path)
        print(f"Completed in {time.time() - start_time:.2f} seconds")
        
        # Create summary videos
        print("Step 7/7: Creating summary videos...")
        start_time = time.time()
        
        print("  Creating summary video with narration...")
        summary_video_path = os.path.join(output_dir, f"{base_name}_summary.mp4")
        self.create_summary_video(summary_video_path, duration_per_segment=4, include_original_audio=False)

        print("  Creating extended summary video with original audio...")
        summary_video_orig_audio_path = os.path.join(output_dir, f"{base_name}_summary_orig_audio.mp4")
        self.create_summary_video(summary_video_orig_audio_path, duration_per_segment=5, include_original_audio=True)
        
        print(f"Video processing completed in {time.time() - start_time:.2f} seconds")
        
        return {
            "summary": summary_path,
            "storyboard_image": storyboard_img_path,
            "storyboard_pdf": storyboard_pdf_path,
            "summary_video": summary_video_path,
            "summary_video_orig_audio": summary_video_orig_audio_path
        }
    
    def create_summary_video(self, output_path, duration_per_segment=None, include_original_audio=False):
        """Create a summary video with improved dynamic segment selection.
        
        Args:
            output_path: Path to save the output video
            duration_per_segment: Override for segment duration (None for auto)
            include_original_audio: Whether to include original audio
        """
        if not self.storyboard_frames:
            self.select_keyframes(min_clusters=10, max_clusters=None)  # Use dynamic clustering
            
        if not self.combined_summary:
            self.generate_summary()
        
        # Create temporary directory
        temp_dir = tempfile.mkdtemp()
        temp_files = []
        
        try:
            # Generate narration
            narration_path = os.path.join(temp_dir, "narration.mp3")
            self._generate_narration(self.combined_summary, narration_path)
            temp_files.append(narration_path)
            
            # Extract video segments based on content changes
            original_video = VideoFileClip(self.video_path)
            video_duration = original_video.duration
            
            # Get timestamps of keyframes
            keyframe_timestamps = self._get_keyframe_timestamps()
            
            # Create clips for each significant scene
            video_clips = []
            
            # Create an intro clip
            title_text = "Video Summary"
            title_clip = TextClip(title_text, fontsize=70, color='white', bg_color='black', 
                            size=(original_video.w, original_video.h))
            title_clip = title_clip.set_position('center').set_duration(3)  # Longer intro
            video_clips.append(title_clip)
            
            # Calculate adaptive segment durations
            if duration_per_segment is None:
                # Adjust segment durations based on video complexity
                if len(keyframe_timestamps) <= 5:
                    # For videos with few scenes, use longer segments
                    duration_per_segment = 6.0
                elif len(keyframe_timestamps) <= 10:
                    duration_per_segment = 4.5
                elif len(keyframe_timestamps) <= 20:
                    duration_per_segment = 3.5
                elif len(keyframe_timestamps) <= 50:
                    duration_per_segment = 2.5
                else:
                    # For videos with many scenes, use shorter segments
                    duration_per_segment = 2.0
            
            # Create clips with dynamic durations based on scene importance
            for i, timestamp in enumerate(keyframe_timestamps):
                # Calculate segment boundaries
                start_time = max(0, timestamp - (duration_per_segment * 0.4))
                
                # For the end time, use the next keyframe or adaptive duration
                if i < len(keyframe_timestamps) - 1:
                    next_timestamp = keyframe_timestamps[i + 1]
                    # If scenes are close together, don't overlap them
                    if next_timestamp - timestamp < duration_per_segment:
                        end_time = next_timestamp - 0.2  # Leave a small gap
                    else:
                        end_time = min(video_duration, timestamp + (duration_per_segment * 0.6))
                else:
                    # For the last segment, use the remaining duration or adaptive duration
                    end_time = min(video_duration, timestamp + (duration_per_segment * 0.6))
                
                # Ensure segment is long enough to be meaningful
                if end_time - start_time < 1.5:
                    end_time = min(video_duration, start_time + 1.5)
                
                # Only include the segment if it's valid
                if end_time > start_time and end_time <= video_duration:
                    # Extract segment from original video
                    segment_clip = original_video.subclip(start_time, end_time)
                    
                    # Skip very static clips (little to no movement)
                    if self._is_static_clip(segment_clip):
                        print(f"Skipping static scene at {timestamp:.1f}s")
                        continue
                    
                    # Add subtle scene caption
                    scene_number = f"Scene {i+1}"
                    txt_clip = TextClip(scene_number, fontsize=30, color='white', bg_color='black',
                                size=(segment_clip.w, 40)).set_opacity(0.7)
                    txt_clip = txt_clip.set_position(('right', 'bottom')).set_duration(segment_clip.duration)
                    
                    # Combine video and text
                    composite_clip = CompositeVideoClip([segment_clip, txt_clip])
                    
                    # Mute original audio if not including it
                    if not include_original_audio:
                        composite_clip = composite_clip.without_audio()
                        
                    # Add transitions between clips - simple fade in/out
                    composite_clip = composite_clip.fadein(0.5).fadeout(0.5)
                        
                    video_clips.append(composite_clip)
            
            # If we have too few clips, use more frames from the video
            if len(video_clips) < 5:
                print("Adding additional scenes for better coverage")
                # Extract more frames using motion detection
                additional_timestamps = self._detect_motion_scenes(min(15, 30 - len(video_clips)))
                
                for timestamp in additional_timestamps:
                    start_time = max(0, timestamp - 1.5)
                    end_time = min(video_duration, timestamp + 2.0)
                    
                    # Check if this segment overlaps with existing clips
                    overlaps = False
                    for clip in video_clips[1:]:  # Skip title clip
                        clip_start = clip.start if hasattr(clip, 'start') else 0
                        clip_end = clip_start + clip.duration if hasattr(clip, 'duration') else 0
                        
                        if (start_time <= clip_end and end_time >= clip_start):
                            overlaps = True
                            break
                    
                    if not overlaps and end_time > start_time:
                        segment_clip = original_video.subclip(start_time, end_time)
                        
                        # Only add if not static
                        if not self._is_static_clip(segment_clip):
                            video_clips.append(segment_clip.fadein(0.5).fadeout(0.5))
            
            # Concatenate all clips
            final_clip = concatenate_videoclips(video_clips, method="compose")
            
            # Synchronize narration
            if not include_original_audio:
                synchronized_narration = self.synchronize_narration_with_visuals(
                    narration_path, len(video_clips), os.path.join(temp_dir, "synchronized_narration.mp3")
                )
                temp_files.append(synchronized_narration)
                
                # Load synchronized narration
                narration_audio = AudioFileClip(synchronized_narration)
                
                # Handle duration mismatch
                if narration_audio.duration < final_clip.duration:
                    # Stretch narration slightly to better match
                    stretch_factor = min(1.2, final_clip.duration / narration_audio.duration)
                    
                    y, sr = librosa.load(synchronized_narration, sr=None)
                    y_stretched = librosa.effects.time_stretch(y, rate=(1/stretch_factor))
                    
                    stretched_path = os.path.join(temp_dir, "stretched_narration.mp3")
                    sf.write(stretched_path, y_stretched, sr)
                    temp_files.append(stretched_path)
                    
                    narration_audio = AudioFileClip(stretched_path)
                
                # Apply audio to video with a subtle background music if possible
                final_clip = final_clip.set_audio(narration_audio)
            
            # Write the final video with high quality settings
            final_clip.write_videofile(
                output_path, 
                codec='libx264', 
                audio_codec='aac', 
                fps=24,
                bitrate='8000k',  # Higher bitrate for better quality
                audio_bitrate='320k',  # Higher audio bitrate
                threads=4,
                preset='slow'  # Slower encoding for better quality
            )
            
            print(f"Enhanced dynamic summary video created with {len(video_clips)} segments")
            return output_path
            
        finally:
            # Clean up temporary files
            for temp_file in temp_files:
                if os.path.exists(temp_file):
                    try:
                        os.remove(temp_file)
                    except Exception:
                        pass
                    
            if os.path.exists(temp_dir):
                try:
                    os.rmdir(temp_dir)
                except OSError:
                    pass
                    
    def _is_static_clip(self, clip, threshold=0.015):
        """Detect if a clip is mostly static (little movement).
        
        Args:
            clip: VideoClip to analyze
            threshold: Motion threshold
            
        Returns:
            True if clip is static, False otherwise
        """
        try:
            # Sample a few frames to check for motion
            duration = clip.duration
            frames = []
            
            # Get 3 sample frames
            for t in np.linspace(0, duration, min(3, int(duration) + 1)):
                frame = clip.get_frame(t)
                frames.append(cv2.cvtColor(frame.astype(np.uint8), cv2.COLOR_RGB2GRAY))
            
            # If we have less than 2 frames, can't determine motion
            if len(frames) < 2:
                return False
            
            # Calculate average frame difference
            diffs = []
            for i in range(1, len(frames)):
                diff = np.abs(frames[i].astype(float) - frames[i-1].astype(float))
                diffs.append(np.mean(diff) / 255.0)  # Normalize to 0-1
            
            avg_diff = np.mean(diffs)
            return avg_diff < threshold
            
        except Exception as e:
            print(f"Error checking for static clip: {e}")
            return False
            
    def _detect_motion_scenes(self, count=5):
        """Detect scenes with significant motion for better coverage.
        
        Args:
            count: Number of motion scenes to detect
            
        Returns:
            List of timestamps with significant motion
        """
        try:
            # Open video for processing
            cap = cv2.VideoCapture(self.video_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            # Sample frames at regular intervals
            sample_rate = max(1, frame_count // 100)
            frames = []
            timestamps = []
            
            for i in range(0, frame_count, sample_rate):
                cap.set(cv2.CAP_PROP_POS_FRAMES, i)
                ret, frame = cap.read()
                if not ret:
                    break
                    
                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
                timestamps.append(i / fps)
            
            # Calculate frame differences
            motion_scores = []
            for i in range(1, len(frames)):
                diff = cv2.absdiff(frames[i], frames[i-1])
                score = np.mean(diff)
                motion_scores.append((timestamps[i], score))
            
            # Sort by motion score (highest first)
            motion_scores.sort(key=lambda x: x[1], reverse=True)
            
            # Return top timestamps with highest motion
            return [t for t, _ in motion_scores[:count]]
            
        except Exception as e:
            print(f"Error detecting motion scenes: {e}")
            return []

    def _generate_narration(self, text, output_path, voice_profile=None, speed=1.2):
        """Generate better quality narration with improved text preparation.
        
        Args:
            text: Text to convert to speech
            output_path: Path to save the audio file
            voice_profile: Voice profile to use (None for default)
            speed: Speech speed factor (1.0 = normal speed)
        """
        try:
            # Better clean up of text for narration
            clean_text = self._prepare_text_for_narration(text)
            
            # Split text into natural segments for better flow
            segments = re.split(r'(?<=[.!?])\s+', clean_text)
            segment_groups = []
            current_group = []
            
            # Group segments into chunks for better processing
            for segment in segments:
                current_group.append(segment)
                if len(' '.join(current_group).split()) > 30:
                    segment_groups.append(' '.join(current_group))
                    current_group = []
            
            if current_group:
                segment_groups.append(' '.join(current_group))
            
            # Try using ElevenLabs for high quality voice
            if hasattr(elevenlabs, 'generate'):
                try:
                    # Set your ElevenLabs API key
                    elevenlabs.set_api_key("sk_caf3c55a9cd7d5211cbde758b98c15241c13836e90328bf3")
                    
                    # Use a consistent voice for better results
                    voice = "Rachel"  # Female voice with good clarity
                    
                    # Generate audio for each segment group with proper pauses
                    audio_segments = []
                    for i, group in enumerate(segment_groups):
                        modified_text = group
                        
                        # Add SSML-like markers for better pacing (ElevenLabs might support some SSML)
                        modified_text = modified_text.replace(". ", "... ")
                        
                        # Generate speech with ElevenLabs
                        audio = elevenlabs.generate(
                            text=modified_text,
                            voice=voice,
                            model="eleven_monolingual_v1",
                            stability=0.5,  # More expressive
                            similarity_boost=0.75,  # Balance between stability and similarity
                            style=0.25  # Slight style variation for more natural speech
                        )
                        audio_segments.append(audio)
                    
                    # Combine audio segments with proper pauses
                    combined_audio = AudioSegment.empty()
                    for audio_data in audio_segments:
                        # Convert bytes to AudioSegment
                        segment = AudioSegment.from_file(BytesIO(audio_data))
                        
                        # Add small pause between segments
                        if len(combined_audio) > 0:
                            combined_audio += AudioSegment.silent(duration=300)
                        
                        combined_audio += segment
                    
                    # Save to file
                    combined_audio.export(output_path, format="mp3")
                    
                    # Apply speed adjustment
                    self._adjust_audio_speed(output_path, speed)
                    
                    # Apply audio enhancement for clarity
                    self._enhance_audio(output_path)
                    
                    print(f"Generated enhanced narration with natural speech patterns")
                    return output_path
                    
                except Exception as e:
                    print(f"Error using ElevenLabs, falling back to gTTS: {e}")
            
            # Fallback to gTTS with improvements
            combined_audio = AudioSegment.empty()
            
            for i, group in enumerate(segment_groups):
                # Create temporary file for this segment
                temp_file = os.path.join(os.path.dirname(output_path), f"temp_segment_{i}.mp3")
                
                # Generate speech for this segment
                tts = gtts.gTTS(text=group, lang='en', slow=False)
                tts.save(temp_file)
                
                # Load segment
                segment = AudioSegment.from_file(temp_file)
                
                # Add small pause between segments
                if len(combined_audio) > 0:
                    combined_audio += AudioSegment.silent(duration=300)
                
                combined_audio += segment
                
                # Clean up temporary file
                if os.path.exists(temp_file):
                    os.remove(temp_file)
            
            # Save combined audio
            combined_audio.export(output_path, format="mp3")
            
            # Apply speed adjustment
            self._adjust_audio_speed(output_path, speed)
            
            # Apply audio enhancement for clarity
            self._enhance_audio(output_path)
            
            print(f"Generated narration with improved segmenting and natural pauses")
            return output_path
            
        except Exception as e:
            print(f"Error generating narration: {e}")
            # Create an empty audio file as fallback
            silent_audio = AudioSegment.silent(duration=5000)
            silent_audio.export(output_path, format="mp3")
            return output_path

    def _prepare_text_for_narration(self, text):
        """Prepare text for narration by removing unnecessary phrases and improving flow.
        Enhanced version with better text cleaning and audio-friendly formatting.
        
        Args:
            text: Original text
            
        Returns:
            Clean text optimized for narration
        """
        # Remove phrases like "in this video", "this video shows", etc.
        phrases_to_remove = [
            "in this video", "this video shows", "the video depicts", 
            "as seen in the video", "the video demonstrates",
            "throughout the video", "the video presents", "the video contains",
            "the video includes", "the video features", "the video highlights",
            "the video focuses on", "the video emphasizes", "the video illustrates",
            "the video portrays", "the video represents", "the video displays",
            "the video reveals", "the video uncovers", "the video exposes",
            "the video uncovers", "the video discloses", "the video divulges",
            "the video unveils", "the video unmasks", "the video unearths",
            "the video uncovers", "the video discloses", "the video divulges",
            "in this scene", "this scene shows", "the scene depicts",
            "as seen in this scene", "the scene demonstrates",
            "throughout the scene", "the scene presents", "the scene contains",
            "the scene includes", "the scene features", "the scene highlights",
            "the scene focuses on", "the scene emphasizes", "the scene illustrates",
            "the scene portrays", "the scene represents", "the scene displays",
            "the scene reveals", "the scene uncovers", "the scene exposes",
            "the scene uncovers", "the scene discloses", "the scene divulges",
            "the scene unveils", "the scene unmasks", "the scene unearths",
            "the scene uncovers", "the scene discloses", "the scene divulges"
        ]
        
        clean_text = text
        for phrase in phrases_to_remove:
            clean_text = re.sub(phrase, "", clean_text, flags=re.IGNORECASE)
        
        # Simplify and improve flow for narration
        clean_text = clean_text.replace("\n\n", ". ").replace("\n", ". ")
        clean_text = re.sub(r'\s+', ' ', clean_text)  # Remove excess whitespace
        clean_text = re.sub(r'\.+', '.', clean_text)  # Fix multiple periods
        clean_text = re.sub(r'\.\s+\.', '.', clean_text)  # Fix period space period
        
        # Add pauses for better pacing
        clean_text = re.sub(r'([.!?])\s+', r'\1... ', clean_text)
        
        # Fix capitalization after periods
        clean_text = re.sub(r'\.\.\.\s+([a-z])', lambda m: f"... {m.group(1).upper()}", clean_text)
        
        # Add emphasis markers for important words (will be interpreted by TTS)
        # This is a simple approach - for more advanced prosody, consider using SSML
        important_words = [
            "first", "second", "third", "finally", "important", "key", "crucial",
            "significant", "essential", "critical", "vital", "fundamental",
            "primary", "main", "major", "principal", "chief", "leading",
            "foremost", "preeminent", "supreme", "paramount", "premier"
        ]
        
        for word in important_words:
            # Add emphasis to standalone words (not parts of other words)
            clean_text = re.sub(r'\b' + word + r'\b', f"<emphasis>{word}</emphasis>", clean_text, flags=re.IGNORECASE)
        
        return clean_text
                    
    def _get_keyframe_timestamps(self):
        """Get the timestamps for each keyframe in the original video.
        More efficient version using frame hashing.
        
        Returns:
            List of timestamps in seconds
        """
        # Open the video
        cap = cv2.VideoCapture(self.video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Extract hash signatures for our keyframes
        keyframe_hashes = []
        for keyframe in self.storyboard_frames:  # Using storyboard frames which are sorted by position
            # Resize and convert to grayscale for faster comparison
            small_frame = cv2.resize(keyframe, (32, 32))
            gray = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)
            # Simple hash: flatten and get the binary pattern of values above/below mean
            avg = gray.mean()
            hash_sig = (gray > avg).flatten()
            keyframe_hashes.append(hash_sig)
        
        # Find closest matches in video
        timestamps = []
        
        # Sample at a reasonable rate to speed up processing
        sample_rate = max(1, frame_count // 500)  # More samples for better accuracy
        
        for i in range(0, frame_count, sample_rate):
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()
            if not ret:
                break
                
            # Calculate hash for current frame
            small_frame = cv2.resize(frame, (32, 32))
            gray = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)
            avg = gray.mean()
            hash_sig = (gray > avg).flatten()
            
            # Compare with all keyframe hashes
            for j, keyframe_hash in enumerate(keyframe_hashes):
                # If we already found a timestamp for this keyframe, skip
                if j < len(timestamps):
                    continue
                    
                # Calculate hamming distance (number of different bits)
                hash_diff = np.sum(hash_sig != keyframe_hash)
                
                # If hash is close enough, consider it a match
                if hash_diff < 100:  # Threshold can be adjusted
                    timestamps.append(i / fps)
                    # If we've found all keyframes, exit early
                    if len(timestamps) == len(keyframe_hashes):
                        break
        
        # Sort timestamps (should already be in order, but just to be safe)
        timestamps.sort()
        
        cap.release()
        return timestamps
    
    def synchronize_narration_with_visuals(self, narration_path, segment_count, output_path):
        """Improved synchronization ensuring narration matches visual segments.
        
        Args:
            narration_path: Path to the original narration audio
            segment_count: Number of video segments to synchronize with
            output_path: Path to save the adjusted narration
        """
        try:
            # Load original narration
            narration = AudioSegment.from_file(narration_path)
            narration_duration = len(narration) / 1000  # in seconds
            
            # Get video data for timing
            video = VideoFileClip(self.video_path)
            video_duration = video.duration
            keyframe_timestamps = self._get_keyframe_timestamps()
            
            # Parse timestamps from narration text if available
            # This extracts any timestamp markers like "[0:45]" that might be in the script
            timestamp_pattern = r'\[(\d+:?\d*)\]'
            script_timestamps = []
            if hasattr(self, 'combined_summary') and self.combined_summary:
                matches = re.findall(timestamp_pattern, self.combined_summary)
                for match in matches:
                    try:
                        if ':' in match:
                            mins, secs = match.split(':')
                            timestamp = int(mins) * 60 + float(secs)
                        else:
                            timestamp = float(match)
                        script_timestamps.append(timestamp)
                    except ValueError:
                        continue
            
            # Divide narration if we have reliable timestamps
            if len(script_timestamps) >= 2:
                # Calculate the segment boundaries in the narration audio
                narration_segments = []
                for i in range(len(script_timestamps) - 1):
                    start_pct = script_timestamps[i] / video_duration
                    end_pct = script_timestamps[i+1] / video_duration
                    start_ms = int(start_pct * len(narration))
                    end_ms = int(end_pct * len(narration))
                    narration_segments.append(narration[start_ms:end_ms])
                
                # Add the last segment
                last_start_pct = script_timestamps[-1] / video_duration
                last_start_ms = int(last_start_pct * len(narration))
                narration_segments.append(narration[last_start_ms:])
                
                # Stretch/compress each segment to match video segment timing
                adjusted_segments = []
                for i in range(min(len(narration_segments), len(keyframe_timestamps)-1)):
                    segment = narration_segments[i]
                    segment_duration = len(segment) / 1000  # seconds
                    
                    # Calculate target duration from video
                    if i < len(keyframe_timestamps) - 1:
                        target_duration = keyframe_timestamps[i+1] - keyframe_timestamps[i]
                    else:
                        target_duration = video_duration - keyframe_timestamps[i]
                    
                    # Adjust speed if needed (limit to 0.8-1.5x for natural sound)
                    if segment_duration > 0 and target_duration > 0:
                        speed_factor = segment_duration / target_duration
                        speed_factor = min(1.5, max(0.8, speed_factor))
                        
                        # Use librosa for better quality stretching
                        y, sr = librosa.load(BytesIO(segment.raw_data), sr=segment.frame_rate)
                        y_stretched = librosa.effects.time_stretch(y, rate=speed_factor)
                        
                        # Convert back to AudioSegment
                        buffer = BytesIO()
                        sf.write(buffer, y_stretched, sr, format='wav')
                        buffer.seek(0)
                        adjusted_segment = AudioSegment.from_wav(buffer)
                        adjusted_segments.append(adjusted_segment)
                    else:
                        adjusted_segments.append(segment)
                
                # Combine adjusted segments
                if adjusted_segments:
                    combined = adjusted_segments[0]
                    for segment in adjusted_segments[1:]:
                        combined += segment
                    combined.export(output_path, format="mp3")
                    return output_path
            
            # Fallback to basic time stretching if timestamps aren't available
            y, sr = librosa.load(narration_path, sr=None)
            
            # Calculate appropriate duration
            total_segments_duration = sum(keyframe_timestamps) if keyframe_timestamps else video_duration
            if total_segments_duration > 0 and narration_duration > 0:
                stretch_factor = total_segments_duration / narration_duration
                # Limit stretch factor for natural sound
                stretch_factor = min(1.3, max(0.85, stretch_factor))
                
                # Time stretch while preserving pitch
                y_stretched = librosa.effects.time_stretch(y, rate=(1/stretch_factor))
                
                # Save the stretched audio
                sf.write(output_path, y_stretched, sr)
                print(f"Adjusted narration timing to match video segments")
                return output_path
            
            # If all else fails, just use original
            narration.export(output_path, format="mp3")
            return output_path
                
        except Exception as e:
            print(f"Error adjusting narration timing: {e}")
            # Return original if there's an error
            return narration_path

    def _adjust_audio_speed(self, audio_path, speed_factor=1.25):
        """Adjust audio speed while preserving pitch quality.
        
        Args:
            audio_path: Path to the audio file
            speed_factor: Speed factor (1.0 = normal speed)
        """
        try:
            audio = AudioSegment.from_file(audio_path)
            
            # Normalize audio first
            audio = normalize(audio)
            
            # For better quality with specific speed factors, use librosa
            if 1.1 <= speed_factor <= 1.5:
                # Use pydub's speedup which preserves pitch better
                modified_audio = speedup(audio, speed_factor, 150)
            else:
                # Use simple speed change for other factors
                modified_audio = audio._spawn(audio.raw_data, overrides={
                    "frame_rate": int(audio.frame_rate * speed_factor)
                })
                modified_audio = modified_audio.set_frame_rate(audio.frame_rate)
            
            # Apply subtle compression for clarity
            modified_audio = modified_audio.compress_dynamic_range(threshold=-20, ratio=4.0)
            
            # Export with high quality
            modified_audio.export(audio_path, format="mp3", bitrate="192k")
            return True
        except Exception as e:
            print(f"Error adjusting audio speed: {e}")
            return False
            
    def _enhance_audio(self, audio_path):
        """Apply audio enhancements to improve quality.
        
        Args:
            audio_path: Path to audio file
        """
        try:
            # Load audio file
            y, sr = librosa.load(audio_path, sr=None)
            
            # Apply audio enhancements
            
            # 1. Normalize audio
            y = librosa.util.normalize(y)
            
            # 2. Apply compression to even out volume
            y = np.clip(y, -0.9, 0.9)  # Soft clipping
            
            # 3. Apply high-pass filter to remove low-frequency noise
            y = librosa.effects.preemphasis(y)
            
            # 4. Apply subtle reverb for better sound
            y = librosa.effects.preemphasis(y, coef=0.97)
            
            # 5. Apply subtle EQ boost to mid-range for clarity
            # This is a simplified EQ - for more advanced EQ, consider using a dedicated audio library
            y = y * 1.2  # Boost mid-range
            
            # Save enhanced audio
            sf.write(audio_path, y, sr)
            
            return True
        except Exception as e:
            print(f"Error enhancing audio: {e}")
            return False
    
    def _analyze_original_voice(self):
        """Analyze the original video's voice characteristics to match with TTS.
        
        Returns:
            Dictionary with voice characteristics
        """
        try:
            # Extract audio from video
            temp_dir = tempfile.mkdtemp()
            temp_audio = os.path.join(temp_dir, "temp_audio.wav")
            
            video = VideoFileClip(self.video_path)
            video.audio.write_audiofile(temp_audio, verbose=False, logger=None)
            
            # Load audio for analysis
            y, sr = librosa.load(temp_audio, sr=None)
            
            # Analyze pitch
            pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
            pitch_values = pitches[magnitudes > np.max(magnitudes) * 0.1]
            avg_pitch = np.mean(pitch_values) if len(pitch_values) > 0 else 0
            
            # Analyze speech rate
            onset_env = librosa.onset.onset_strength(y=y, sr=sr)
            tempo = librosa.beat.tempo(onset_envelope=onset_env, sr=sr)[0]
            
            # Analyze voice gender (simplified approach)
            # Higher pitch typically indicates female voice
            is_female = avg_pitch > 150 if avg_pitch > 0 else True
            
            # Clean up
            if os.path.exists(temp_audio):
                os.remove(temp_audio)
            if os.path.exists(temp_dir):
                os.rmdir(temp_dir)
                
            # Return voice characteristics
            return {
                "avg_pitch": avg_pitch,
                "tempo": tempo,
                "is_female": is_female
            }
        except Exception as e:
            print(f"Error analyzing original voice: {e}")
            # Return default values
            return {
                "avg_pitch": 0,
                "tempo": 120,
                "is_female": True
            }

    def _adjust_audio_pitch(self, audio_path, target_pitch):
        """Adjust audio pitch to match the target pitch.
        
        Args:
            audio_path: Path to the audio file
            target_pitch: Target pitch in Hz
        """
        try:
            # Load audio
            y, sr = librosa.load(audio_path, sr=None)
            
            # Get current pitch
            pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
            pitch_values = pitches[magnitudes > np.max(magnitudes) * 0.1]
            current_pitch = np.mean(pitch_values) if len(pitch_values) > 0 else 0
            
            if current_pitch > 0 and target_pitch > 0:
                # Calculate pitch shift factor
                pitch_shift = target_pitch / current_pitch
                
                # Limit pitch shift to avoid unnatural sound
                pitch_shift = min(1.5, max(0.7, pitch_shift))
                
                # Apply pitch shift
                y_shifted = librosa.effects.pitch_shift(y, sr=sr, n_steps=12 * np.log2(pitch_shift))
                
                # Save adjusted audio
                sf.write(audio_path, y_shifted, sr)
                
                print(f"Adjusted pitch from {current_pitch:.1f}Hz to {target_pitch:.1f}Hz")
                return True
            else:
                print("Could not determine pitch for adjustment")
                return False
        except Exception as e:
            print(f"Error adjusting audio pitch: {e}")
            return False

# Example usage
if __name__ == "__main__":
    # Replace with your video path
    video_path = "/Users/krishilparikh/Downloads/videoplayback (1).mp4"
    
    # Replace with your API key
    api_key = "AIzaSyBfIWKs0ffmQGdx7suNBUBgM4C_0Ss3LMs"
    
    # Using Gemini 2.0 Flash
    model_provider = "gemini"
    
    # Create summarizer
    summarizer = VideoSummarizer(video_path, api_key, model_provider)
    
    # Process video
    output_files = summarizer.process_video()
    
    print("Video processing complete!")
    print(f"Summary saved to: {output_files['summary']}")
    print(f"Storyboard image saved to: {output_files['storyboard_image']}")
    print(f"Storyboard PDF saved to: {output_files['storyboard_pdf']}")