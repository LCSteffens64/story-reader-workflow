# Story Reader Workflow

An automated video generation pipeline that transforms narration audio (or text scripts) into visual "story reader" videos with AI-generated imagery and optional background music.

---
## USAGE
```
python -m story_reader -i narration.wav -o output/
# or
python main.py -i narration.wav -o output/
```

## Overview

This tool takes spoken narration and automatically:
1. Transcribes the audio using OpenAI Whisper
2. Segments the transcript into logical paragraphs
3. Generates AI images for each paragraph using Stable Diffusion
4. Applies Ken Burns (zoom/pan) effects to bring images to life
5. Assembles everything into a final video with synchronized audio

---

## Current Pipeline

```
┌─────────────────┐
│  narration.wav  │  (Input: audio file with spoken narration)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│    Whisper      │  Speech-to-text transcription with timestamps
│  Transcription  │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   Paragraph     │  Groups sentences by silence gaps, duration,
│  Segmentation   │  and sentence count
└────────┬────────┘
         │
         ▼
┌──────────────────────────────────────────────────────────────┐
│                     Image Source Select                      │
│   • Default: Stable Diffusion (local)                         │
│   • Optional: Pexels (stock photos)                           │
│   • Optional: Legnext (remote diffusion)                      │
│   Note: --use-pexels and --use-legnext are mutually exclusive  │
└────────┬─────────────────────────────────────────────────────┘
         │
         ├───────────────┬───────────────────────────────┬───────────────┐
         │               │                               │               │
         ▼               ▼                               ▼               │
┌─────────────────┐  ┌─────────────────┐          ┌─────────────────┐    │
│ Stable Diffusion│  │ LLM Keyword     │          │   Legnext API   │    │
│ Image Generation│  │ Extraction      │          │  (Diffusion Job)│    │
└────────┬────────┘  └────────┬────────┘          └────────┬────────┘    │
         │                    │                              │           │
         │                    ▼                              ▼           │
         │           ┌─────────────────┐             ┌─────────────────┐  │
         │           │   Pexels API    │             │  Job Polling    │  │
         │           │  Search Query   │             │  + Image Fetch  │  │
         │           └────────┬────────┘             └────────┬────────┘  │
         │                    │                              │           │
         │                    ▼                              ▼           │
         │           ┌─────────────────┐                     │           │
         │           │  Pexels Fetch   │                     │           │
         │           └────────┬────────┘                     │           │
         │                    │                              │           │
         │                    ▼                              │           │
         │           ┌─────────────────┐                     │           │
         │           │  Optional SD    │                     │           │
         │           │   Fallback      │                     │           │
         │           └─────────────────┘                     │           │
         │                                                   │           │
         └───────────────────────────────┬───────────────────┴───────────┘
                                         │
                                         ▼
┌─────────────────┐
│   Ken Burns     │  Applies slow zoom/pan animation to each
│  Video Effect   │  image, matching paragraph duration
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   Audio Muxing  │  Combines visual video with original
│    (FFmpeg)     │  narration audio
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ final_video.mp4 │  Output: complete narrated slideshow
└─────────────────┘
```

---

## Planned Expansions

### 1. AI-Generated Narration Audio (Text-to-Speech)

**Goal:** Accept a text script instead of pre-recorded audio, and generate the narration automatically.

```
┌─────────────────┐
│   story.txt     │  (NEW Input: plain text script)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   TTS Engine    │  Converts text to speech using:
│  (Piper/Coqui)  │  • Piper TTS (fast, CPU-friendly)
└────────┬────────┘  • Coqui TTS (higher quality, GPU)
         │           • Edge TTS (cloud, Microsoft voices)
         ▼
┌─────────────────┐
│  narration.wav  │  Generated audio file
└────────┬────────┘
         │
         ▼
    (continues to existing pipeline...)
```

**Proposed CLI Options:**
```bash
--script story.txt      # Input text file instead of audio
--voice "en_US-amy"     # Select TTS voice
--tts-engine piper      # Select TTS engine (piper/coqui/edge)
```

**TTS Engine Comparison:**

| Engine | Speed | Quality | Requirements | Cost |
|--------|-------|---------|--------------|------|
| Piper TTS | ⚡ Very Fast | Good | CPU only | Free |
| Coqui TTS | Medium | Excellent | GPU recommended | Free |
| Bark | Slow | Natural | GPU required | Free |
| Edge TTS | Fast | Good | Internet | Free |

---

### 2. Automatic Background Music

**Goal:** Automatically add public-domain background music that matches the content's mood.

#### Approach A: Curated Library (Simpler)

```
┌─────────────────┐
│  music_library/ │  Local folder containing pre-downloaded
│   ├── calm/     │  public domain tracks, organized by mood
│   ├── dramatic/ │
│   └── upbeat/   │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Random/Manual   │  Select track based on:
│   Selection     │  • Random choice from mood folder
└────────┬────────┘  • Manual specification via CLI
         │
         ▼
    (mixed with narration at lower volume)
```

**Music Sources:**
- [Free Music Archive (FMA)](https://freemusicarchive.org/)
- [Incompetech (Kevin MacLeod)](https://incompetech.com/)
- [Musopen (Classical)](https://musopen.org/)
- [ccMixter](http://ccmixter.org/)

#### Approach B: AI-Powered Selection (Smarter)

```
┌─────────────────┐
│   Transcript    │  Analyzed for mood and themes
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   LLM Analysis  │  Local (Ollama) or API-based
│  (Mood/Theme)   │  Outputs: "calm", "dramatic", "mysterious"
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Music Search   │  Query APIs:
│   (FMA/Jamendo) │  • Free Music Archive API
└────────┬────────┘  • Jamendo API (Creative Commons)
         │
         ▼
┌─────────────────┐
│ Auto-Download   │  Fetch matching track, cache locally
└────────┬────────┘
         │
         ▼
    (mixed with narration at ~30% volume)
```

**Proposed CLI Options:**
```bash
--music track.mp3       # Manually specify background music
--music-library ./music # Folder of tracks to choose from
--music-mood calm       # Specify mood for auto-selection
--auto-music            # AI analyzes content and selects music
--no-music              # Disable background music entirely
--music-volume 0.3      # Background music volume (0.0-1.0)
```

---

## Full Expanded Pipeline

```
┌─────────────────┐     ┌─────────────────┐
│   story.txt     │ OR  │  narration.wav  │
│  (text script)  │     │ (existing audio)│
└────────┬────────┘     └────────┬────────┘
         │                       │
         ▼                       │
┌─────────────────┐              │
│ [NEW] TTS       │              │
│ Generation      │──────────────┤
└─────────────────┘              │
                                 │
         ┌───────────────────────┘
         │
         ▼
┌─────────────────┐
│    Whisper      │  (skipped if script provided with
│  Transcription  │   timestamps already known)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   Paragraph     │
│  Segmentation   │
└────────┬────────┘
         │
         ├──────────────────────────────┐
         │                              │
         ▼                              ▼
┌─────────────────┐            ┌─────────────────┐
│ Stable Diffusion│            │ [NEW] Music     │
│ Image Generation│            │ Selection       │
└────────┬────────┘            └────────┬────────┘
         │                              │
         ▼                              │
┌─────────────────┐                     │
│   Ken Burns     │                     │
│  Video Effect   │                     │
└────────┬────────┘                     │
         │                              │
         ▼                              │
┌─────────────────┐                     │
│  Video + Audio  │◀────────────────────┘
│    Muxing       │  (narration + optional background music)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ final_video.mp4 │
└─────────────────┘
```

---

## Installation

```bash
# Clone or download this repository
cd "story reader workflow"

# Install Python dependencies
pip install -r requirements.txt

# Ensure ffmpeg is installed
sudo apt install ffmpeg  # Debian/Ubuntu
```

---

## Usage

### Basic Usage (with pre-recorded audio)
```bash
python main.py -i narration.wav -o output/
```

### With Background Music
```bash
python main.py -i narration.wav -m background_music.mp3 -o output/
```

### Clear Cache and Regenerate
```bash
python main.py -i narration.wav --clear-cache -o output/
```

### Visual-Only Output (no audio)
```bash
python main.py -i narration.wav --no-audio -o output/
```

### Pexels Integration

Use real stock photos from Pexels instead of AI-generated images:

```bash
# Use Pexels with Stable Diffusion fallback (recommended)
python -m story_reader -i narration.wav --use-pexels -o output/

# Pure Pexels mode (no AI generation)
python -m story_reader -i narration.wav --use-pexels -o output/

# Pexels with custom settings
python -m story_reader -i narration.wav --use-pexels --pexels-per-page 10 --pexels-min-width 1920 -o output/
```

**Pexels API Key Setup:**
1. Get a free API key from [Pexels API](https://www.pexels.com/api/)
2. Set as environment variable: `export PEXELS_API_KEY="your_key_here"`
3. Or create `pexels.txt` and run `python setup_env.py`

**How It Works:**
- **Smart Keyword Extraction**: Uses NLP to extract 2-3 key concepts from both your text AND the Stable Diffusion prompt
- **Optimized Search**: Converts keywords into targeted Pexels search queries
- **Automatic Fallback**: If Pexels fails, automatically falls back to Stable Diffusion
- **Mixed Results**: Can combine real photos and AI images in the same video

**Example:**
```
Input: "The old lighthouse stood proudly against the stormy sea"
NLP Extracts: ["lighthouse", "stormy", "sea", "beam"]
Pexels Search: "lighthouse stormy sea"
Result: Real lighthouse photo OR AI-generated image if Pexels fails
```

**Pexels Configuration Options:**
| Option | Default | Description |
|--------|---------|-------------|
| `--use-pexels` | `false` | Enable Pexels integration |
| `--sd-fallback` | `false` | Use SD if Pexels fails |
| `--pexels-per-page` | `15` | Search results per query |
| `--pexels-min-width` | `1280` | Minimum image width |
| `--pexels-min-height` | `720` | Minimum image height |

### Legnext Integration

Generate images from Legnext instead of local Stable Diffusion:

```bash
# Use Legnext images
python -m story_reader -i narration.wav --use-legnext -o output/

# Tune Legnext polling behavior
python -m story_reader -i narration.wav --use-legnext --legnext-poll-interval 1.5 --legnext-timeout 240 -o output/
```

**Legnext API Key Setup:**
1. Get an API key from Legnext
2. Set as environment variable: `export LEGNEXT_API_KEY="your_key_here"`
3. Or create `legnext.txt` and run `python setup_env.py`

**Legnext Configuration Options:**
| Option | Default | Description |
|--------|---------|-------------|
| `--use-legnext` | `false` | Enable Legnext integration |
| `--legnext-poll-interval` | `2.0` | Polling interval in seconds |
| `--legnext-timeout` | `180.0` | Job timeout in seconds |

**Note:** `--use-pexels` and `--use-legnext` are mutually exclusive.

---

## Configuration

Key settings in `main.py`:

| Setting | Default | Description |
|---------|---------|-------------|
| `WHISPER_MODEL` | `"tiny"` | Whisper model size (tiny/base/small/medium/large) |
| `SD_MODEL` | `"runwayml/stable-diffusion-v1-5"` | Stable Diffusion model |
| `IMAGE_SIZE` | `(384, 384)` | Generated image dimensions |
| `MAX_PARAGRAPH_DURATION` | `15.0` | Max seconds per paragraph |
| `MIN_SILENCE` | `1.5` | Silence gap to split paragraphs |
| `MAX_SENTENCES` | `3` | Max sentences per paragraph |
| `FPS` | `25` | Video frame rate |

---

## Output Files

After running, the `output/` directory contains:

```
output/
├── .cache/                    # Cached transcriptions and images
│   ├── cache_index.json
│   └── transcription_*.json
├── 000.png, 001.png, ...      # Generated images
├── scene_000.mp4, ...         # Individual Ken Burns clips
├── paragraphs.json            # Segmented transcript with timestamps
├── jobs.json                  # Pipeline job status tracking
├── visuals.mp4                # Video without audio
└── final_video.mp4            # Complete video with audio
```

---

## License

This project uses open-source components:
- [OpenAI Whisper](https://github.com/openai/whisper) - MIT License
- [Stable Diffusion](https://github.com/CompVis/stable-diffusion) - CreativeML Open RAIL-M
- [FFmpeg](https://ffmpeg.org/) - LGPL/GPL

---

## Roadmap

- [x] Basic pipeline: audio → images → video
- [x] Caching for transcription and images
- [x] Background music muxing support
- [x] Photojournalistic image prompts with negative prompts
- [ ] Text-to-speech integration (Piper/Coqui/Edge)
- [ ] Automatic music selection based on content mood
- [ ] Web UI for easier interaction
- [ ] Batch processing multiple stories
