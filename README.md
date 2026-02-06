# Media Compressor

A powerful, universal batch media compressor for **Video** and **Images**.  
Optimizes your media library to save space while maintaining high visual quality.

## ✨ Features

### 🎥 Video Compression
- **High Efficiency**: Uses **H.265 (HEVC)** for maximum compression
- **Smart Codec Detection**: Automatically skips videos already using efficient codecs (AV1, H.265/HEVC)
- **Pre-Compression Estimation**: Samples multiple points to predict compression ratio; skips videos that won't benefit
- **Safe Compression**: If compressed file is larger than original, the original is kept (prevents data loss)

### 🖥️ Multi-Hardware Acceleration
Automatically detects and uses the best available encoder:

| Hardware | Encoder | Detection |
|----------|---------|-----------|
| **NVIDIA GPU** | `hevc_nvenc` | nvidia-smi |
| **AMD GPU** | `hevc_vaapi` | VAAPI (Mesa) |
| **Intel GPU/iGPU** | `hevc_qsv` / `hevc_vaapi` | Intel Media SDK / VAAPI |
| **CPU** | `libx265` | Always available |

- **Hybrid Processing**: Simultaneous CPU and GPU workers for maximum throughput
- **Auto-Config**: Automatically detects CPU threads and VRAM to set optimal worker counts
- **Fallback**: Gracefully falls back to CPU if no GPU is available

### 🖼️ Image Compression
- **Smart Format Handling**: 
  - PNG/BMP/TIFF → **Lossless WebP** (zero quality loss)
  - JPEG → **Lossy WebP** (quality=90, visually lossless)
- **Format Support**: `.jpg`, `.jpeg`, `.png`, `.bmp`, `.tiff`, `.webp`, `.gif`
- **Safe Compression**: Keeps original if compression makes file larger
- **Flexible**: Option to keep original format with `--keep-format`

### 🛡️ Safety & Robustness
- **Never lose quality**: Skips files that would get larger after compression
- **Smart skip logic**: Avoids re-encoding already efficient formats
- **Pre-flight checks**: Estimates compression ratio before full encode (videos > 60s)
- **Corrupted file detection**: Validates files before processing
- **Input validation**: Validates all parameters before starting
- **Timeout support**: Optional per-file timeout to skip stuck files
- **Symlink handling**: Optional `--follow-symlinks` flag

### 📊 Reporting
- **Processing Summary**: Shows files processed, skipped, failed
- **Space Saved**: Total compression savings at the end
- **Logging**: Optional `--log-file` for detailed logs

## 📦 Installation

### Prerequisites
- **FFmpeg**: Must be installed and accessible in your system PATH
- **Python**: 3.12+
- **uv**: Python package manager (recommended)

### Optional (for hardware acceleration)
- **NVIDIA**: NVIDIA drivers with NVENC support
- **AMD/Intel VAAPI**: `vainfo` tool, Mesa VAAPI drivers
- **Intel QSV**: Intel Media SDK

### Install
```bash
# Clone the repository
git clone https://github.com/Initsnow/media-compressor.git
cd media-compressor

# Install dependencies
uv sync
```

## 🚀 Usage

### Basic Usage
```bash
# Compress all videos and images in a folder (auto-detect encoder)
uv run main.py /path/to/media_folder

# Automatic hardware detection (Recommended)
uv run main.py --auto-config /path/to/media_folder

# List available encoders
uv run main.py --list-encoders
```

### Encoder Selection
```bash
# Use specific encoder
uv run main.py --encoder nvenc /videos     # NVIDIA NVENC
uv run main.py --encoder vaapi /videos     # AMD/Intel VAAPI  
uv run main.py --encoder qsv /videos       # Intel Quick Sync
uv run main.py --encoder cpu /videos       # CPU only (libx265)
```

### Common Examples

```bash
# High quality video compression (CRF 20)
uv run main.py --crf 20 /movies

# Keep original image format (don't convert to WebP)
uv run main.py --keep-format /photos

# Full automation with source deletion
uv run main.py --auto-config --delete-source /downloads

# CPU-only processing (2 workers)
uv run main.py --cpu-workers 2 /videos

# GPU-only processing (3 workers)
uv run main.py --gpu-workers 3 /videos

# Quiet mode with logging
uv run main.py --quiet --log-file compress.log /videos

# Set timeout (skip files taking longer than 1 hour)
uv run main.py --timeout 3600 /videos

# Disable resume functionality (re-scan all files)
uv run main.py --no-resume /videos
```

### All Options

| Flag | Description | Default |
|------|-------------|---------|
| `--output-path`, `-o` | Output directory or file path | `{input}_compressed` |
| `--delete-source` | Delete original file after successful compression | `False` |
| `--auto-config` | Auto-detect optimal CPU/GPU worker counts | `False` |
| `--cpu-workers` | Number of CPU encoding workers | `0` |
| `--gpu-workers` | Number of GPU/hardware encoding workers | `0` |
| `--encoder` | Encoder: `auto`, `nvenc`, `vaapi`, `qsv`, `cpu` | `auto` |
| `--list-encoders` | List available encoders and exit | - |

#### Video Options
| Flag | Description | Default |
|------|-------------|---------|
| `--crf` | CRF/CQ/QP value (lower = better quality, larger file) | `23` |
| `--preset` | Encoding preset (ultrafast → veryslow) | `medium` |
| `--timeout` | Timeout per file in seconds (0 = no timeout) | `0` |

#### Image Options
| Flag | Description | Default |
|------|-------------|---------|
| `--image-quality` | JPEG/WebP quality (1-100) | `90` |
| `--keep-format` | Keep original format instead of converting to WebP | `False` |

#### Other Options
| Flag | Description | Default |
|------|-------------|---------|
| `--no-scan-duration` | Skip pre-scanning video durations (faster startup) | `False` |
| `--no-resume` | Disable resume state tracking | `False` |
| `--follow-symlinks` | Follow symbolic links when scanning directories | `False` |
| `--log-file` | Write detailed logs to file | - |
| `--quiet`, `-q` | Reduce output verbosity | `False` |
| `--verbose`, `-v` | Increase output verbosity | `False` |

## 📊 How It Works

```
Input File
    ↓
[1] Check if output exists → Skip
    ↓
[2] Check file permissions → Error if unreadable
    ↓
[3] Validate file integrity (ffprobe) → Skip if corrupted
    ↓
[4] Check codec (AV1/H.265?) → Move/Copy original → Skip
    ↓
[5] Video > 60s? → Sample 3 points → Estimate ratio
    ↓
    Ratio ≥ 0.95? → Move/Copy original → Skip
    ↓
[6] Compress with best available encoder
    ↓
[7] Compare sizes
    ↓
    Larger? → Discard, keep original
    ↓
    Smaller? → Use compressed ✓
    ↓
[8] Update statistics, print summary
```

## 🔧 Troubleshooting

### No hardware acceleration detected
```bash
# Check available encoders
uv run main.py --list-encoders

# For NVIDIA: ensure nvidia-smi works
nvidia-smi

# For AMD/Intel VAAPI: check vainfo
vainfo

# Ensure FFmpeg has required encoder support
ffmpeg -encoders | grep hevc
```

### Permission denied for VAAPI
```bash
# Add user to video group
sudo usermod -aG video $USER
# Log out and log back in
```

### File appears corrupted
The tool uses ffprobe to validate files. If a file is marked as corrupted:
- Try playing the file in a media player
- Use `ffprobe -v error file.mp4` to check for errors
- Re-download or recover the file if possible

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

GPL-3.0 License - see [LICENSE](LICENSE) for details.
