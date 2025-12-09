import os
import subprocess
import uuid
import json
import re
from typing import List, Dict, Any
from pathlib import Path
from dotenv import load_dotenv
from google import generativeai as genai
from starlette.concurrency import run_in_threadpool

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.0-flash-exp")
UPLOAD_DIR = Path(os.getenv("UPLOAD_DIR", "./tmp_uploads"))
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# configure Gemini
if GEMINI_API_KEY:
    try:
        genai.configure(api_key=GEMINI_API_KEY)
    except Exception:
        pass

# Batch prompt for all sentences at once
BATCH_PROMPT_TEMPLATE = """You are an expert technical educator reviewing a lecture transcript about programming, algorithms, and data structures.

Below is a list of sentences from a video lecture, each with an index number and timestamp.

For EACH sentence, analyze if the explanation could be improved for better viewer understanding. Consider:
- Missing key concepts (time/space complexity, definitions, prerequisites)
- Unclear explanations that need more context
- Technical terms used without explanation
- Incomplete thoughts or logical gaps

For each sentence, provide your analysis in this EXACT format:
1. "needs_improvement": true or false - Does this sentence need improvement?
2. "reason": If needs_improvement is true, explain what's missing or unclear (1-2 lines). If false, leave empty string "".
3. "improved_sentence": If needs_improvement is true, provide a better version that includes the missing information. If false, leave empty string "".

CRITICAL INSTRUCTIONS:
- Respond ONLY with a valid JSON array
- No markdown code blocks (no ```)
- No explanation text before or after the JSON
- Each object must have "index", "needs_improvement", "reason", and "improved_sentence" fields
- The array must have exactly {num_sentences} objects matching the input order

Input sentences:
{sentences_json}

Example response format:
[
  {{"index": 0, "needs_improvement": true, "reason": "Missing explanation of what OOP stands for", "improved_sentence": "Today we will be learning about OOP (Object-Oriented Programming), a programming paradigm based on objects and classes."}},
  {{"index": 1, "needs_improvement": false, "reason": "", "improved_sentence": ""}}
]

Respond with only the JSON array, nothing else:"""

def extract_audio_from_video(video_path: str, out_audio_path: str) -> None:
    """Extracts audio to 16k mono wav using ffmpeg CLI."""
    ffmpeg_binary = os.getenv("FFMPEG_BINARY", "ffmpeg")
    cmd = [
        ffmpeg_binary, "-y",
        "-i", video_path,
        "-vn",
        "-acodec", "pcm_s16le",
        "-ar", "16000",
        "-ac", "1",
        out_audio_path
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

def merge_short_segments(segments: List[Dict[str, Any]], min_duration: float = 2.0) -> List[Dict[str, Any]]:
    """
    Merge very short segments to create more meaningful sentences.
    Whisper sometimes splits on every pause, this helps create coherent thoughts.
    """
    if not segments:
        return []
    
    merged = []
    current = None
    
    for seg in segments:
        duration = seg["end"] - seg["start"]
        text = seg["text"].strip()
        
        if not text:  # Skip empty segments
            continue
            
        if current is None:
            current = {
                "start": seg["start"],
                "end": seg["end"],
                "text": text
            }
        else:
            # Merge if current segment is too short or if the new one would create a better sentence
            current_duration = current["end"] - current["start"]
            
            # Merge conditions:
            # 1. Current is very short (< min_duration)
            # 2. Current doesn't end with sentence-ending punctuation
            should_merge = (
                current_duration < min_duration or
                not current["text"].rstrip().endswith(('.', '!', '?', '।'))
            )
            
            if should_merge and (seg["start"] - current["end"]) < 1.5:  # Max 1.5s gap
                current["text"] += " " + text
                current["end"] = seg["end"]
            else:
                merged.append(current)
                current = {
                    "start": seg["start"],
                    "end": seg["end"],
                    "text": text
                }
    
    if current:
        merged.append(current)
    
    return merged

def build_batch_prompt(sentences: List[Dict[str, Any]]) -> str:
    """Build a prompt with all sentences for batch processing."""
    sentences_list = []
    for idx, sent in enumerate(sentences):
        sentences_list.append({
            "index": idx,
            "timestamp": f"{sent['start']:.2f}s - {sent['end']:.2f}s",
            "text": sent["text"]
        })
    
    sentences_json = json.dumps(sentences_list, indent=2, ensure_ascii=False)
    return BATCH_PROMPT_TEMPLATE.format(
        sentences_json=sentences_json,
        num_sentences=len(sentences)
    )

def clean_json_response(text: str) -> str:
    """Remove markdown code blocks and extract clean JSON."""
    text = text.strip()
    
    # Remove markdown code blocks if present
    if text.startswith("```"):
        lines = text.split('\n')
        # Remove first line if it's ```json or ```
        if lines[0].startswith("```"):
            lines = lines[1:]
        # Remove last line if it's ```
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = '\n'.join(lines).strip()
    
    # Try to find JSON array boundaries
    start = text.find('[')
    end = text.rfind(']')
    
    if start != -1 and end != -1:
        text = text[start:end+1]
    
    return text

async def call_gemini_batch(sentences: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Calls Gemini once with all sentences and returns improvements for each.
    Returns list matching input order with improvement data.
    """
    def _sync_generate():
        try:
            model_name = GEMINI_MODEL
            if model_name.startswith("models/"):
                model_name = model_name.replace("models/", "")
            
            model = genai.GenerativeModel(model_name)
            prompt = build_batch_prompt(sentences)
            
            # Use generate_content with basic config (without response_mime_type for compatibility)
            response = model.generate_content(
                prompt,
                generation_config={
                    "max_output_tokens": 4096,
                    "temperature": 0.3,
                }
            )
            
            text = response.text.strip()
            print(f"Raw Gemini response (first 500 chars): {text[:500]}")
            
            # Clean and parse JSON
            cleaned_text = clean_json_response(text)
            print(f"Cleaned JSON (first 500 chars): {cleaned_text[:500]}")
            
            improvements = json.loads(cleaned_text)
            
            # Validate it's a list
            if not isinstance(improvements, list):
                print(f"Response is not a list, it's: {type(improvements)}")
                raise ValueError("Response is not a JSON array")
            
            # Ensure we have the right number of improvements
            if len(improvements) != len(sentences):
                print(f"Warning: Got {len(improvements)} improvements but expected {len(sentences)}")
                # Try to match by index field if present
                if all("index" in imp for imp in improvements):
                    sorted_improvements = sorted(improvements, key=lambda x: x.get("index", 0))
                    improvements = sorted_improvements
                
                # Still mismatch? Pad or truncate
                if len(improvements) < len(sentences):
                    # Pad with defaults
                    for i in range(len(improvements), len(sentences)):
                        improvements.append({
                            "needs_improvement": False,
                            "reason": "",
                            "improved_sentence": ""
                        })
                elif len(improvements) > len(sentences):
                    # Truncate
                    improvements = improvements[:len(sentences)]
            
            print(f"Successfully parsed {len(improvements)} improvements")
            return improvements
            
        except json.JSONDecodeError as e:
            error_msg = f"JSON parse error: {str(e)}"
            print(f"Gemini batch processing error: {error_msg}")
            print(f"Failed to parse: {text[:1000] if 'text' in locals() else 'No text captured'}")
            
            # Return safe fallback for all sentences
            return [{
                "needs_improvement": False,
                "reason": f"JSON parsing failed: {str(e)[:100]}",
                "improved_sentence": ""
            } for _ in sentences]
            
        except Exception as e:
            error_msg = str(e)
            print(f"Gemini batch processing error: {error_msg}")
            
            # Return safe fallback for all sentences
            return [{
                "needs_improvement": False,
                "reason": f"Error: {error_msg[:100]}",
                "improved_sentence": ""
            } for _ in sentences]
    
    result = await run_in_threadpool(_sync_generate)
    return result

async def transcribe_with_whisper(whisper_model, audio_path: str) -> List[Dict[str, Any]]:
    """
    Run whisper.transcribe in threadpool.
    Returns list of segments with 'start', 'end', 'text'.
    """
    def _sync_transcribe():
        res = whisper_model.transcribe(
            audio_path, 
            language=None, 
            task="transcribe",
            word_timestamps=False  # Faster without word-level timestamps
        )
        segments = res.get("segments", [])
        return segments

    segments = await run_in_threadpool(_sync_transcribe)
    
    # Normalize segments
    out = []
    for seg in segments:
        out.append({
            "start": float(seg.get("start", 0.0)),
            "end": float(seg.get("end", 0.0)),
            "text": str(seg.get("text", "")).strip()
        })
    return out

async def process_video_file(file_path: str, whisper_model, db) -> Dict[str, Any]:
    """
    Full pipeline:
      1. Extract audio
      2. Transcribe with Whisper
      3. Merge short segments into coherent sentences
      4. Batch process all sentences with Gemini (single API call)
      5. Save to MongoDB
    """
    tmp_audio = str(UPLOAD_DIR / f"{uuid.uuid4().hex}.wav")
    try:
        # Step 1: Extract audio
        print("Extracting audio...")
        await run_in_threadpool(extract_audio_from_video, file_path, tmp_audio)

        # Step 2: Transcribe
        print("Transcribing audio...")
        raw_segments = await transcribe_with_whisper(whisper_model, tmp_audio)
        
        # Step 3: Merge short segments to create better sentences
        print(f"Merging {len(raw_segments)} segments into coherent sentences...")
        merged_sentences = merge_short_segments(raw_segments, min_duration=3.0)
        print(f"Created {len(merged_sentences)} sentences after merging")

        # Step 4: Batch process with Gemini (single API call!)
        print("Analyzing all sentences with Gemini (batch processing)...")
        improvements = await call_gemini_batch(merged_sentences)

        # Step 5: Combine results
        sentences = []
        for idx, sent in enumerate(merged_sentences):
            improvement = improvements[idx] if idx < len(improvements) else {
                "needs_improvement": False,
                "reason": "",
                "improved_sentence": ""
            }
            
            sentences.append({
                "text": sent["text"],
                "start": sent["start"],
                "end": sent["end"],
                "duration": round(sent["end"] - sent["start"], 2),
                "needs_improvement": improvement.get("needs_improvement", False),
                "improvement": {
                    "reason": improvement.get("reason", ""),
                    "suggestion": improvement.get("improved_sentence", "")
                }
            })

        # Step 6: Save to MongoDB
        doc = {
            "file_name": Path(file_path).name,
            "total_sentences": len(sentences),
            "sentences_needing_improvement": sum(1 for s in sentences if s["needs_improvement"]),
            "sentences": sentences
        }
        inserted_id = await db.insert_analysis("video_analyses", doc)

        print(f"Processing complete! {doc['sentences_needing_improvement']}/{doc['total_sentences']} sentences need improvement")

        return {
            "id": inserted_id,
            "total_sentences": len(sentences),
            "sentences_needing_improvement": doc["sentences_needing_improvement"],
            "sentences": sentences
        }
    finally:
        # Cleanup
        try:
            if os.path.exists(tmp_audio):
                os.remove(tmp_audio)
        except Exception:
            pass