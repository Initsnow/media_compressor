# Media Compressor

A powerful, universal batch media compressor for **Video** and **Images**.  
Optimizes your media library to save space while maintaining high visual quality.

## ✨ Features

### 🎥 Video Compression
- **High Efficiency**: Uses **H.265 (HEVC)** for maximum compression
- **Smart Codec Detection**: Automatically skips videos already using efficient codecs (AV1, H.265/HEVC)
- **Pre-Compression Estimation**: Samples multiple points to predict compression ratio; skips videos that won't benefit
- **Safe Compression**: If compressed file is larger than original, the original is kept (prevents data loss)
- **Hardware Acceleration**: Automatic **NVIDIA GPU (NVENC)** detection and usage
- **Hybrid Processing**: Simultaneous CPU and GPU workers for maximum throughput
- **Auto-Config**: Automatically detects CPU threads and VRAM to set optimal worker counts

### 🖼️ Image Compression
- **Smart Format Handling**: 
  - PNG/BMP/TIFF → **Lossless WebP** (zero quality loss)
  - JPEG → **Lossy WebP** (quality=90, visually lossless)
- **Format Support**: `.jpg`, `.jpeg`, `.png`, `.bmp`, `.tiff`, `.webp`
- **Safe Compression**: Keeps original if compression makes file larger
- **Flexible**: Option to keep original format with `--keep-format`

### 🛡️ Safety Features
- **Never lose quality**: Skips files that would get larger after compression
- **Smart skip logic**: Avoids re-encoding already efficient formats
- **Pre-flight checks**: Estimates compression ratio before full encode (videos > 60s)

## 📦 Installation

### Prerequisites
- **FFmpeg**: Must be installed and accessible in your system PATH
- **Python**: 3.12+
- **uv**: Python package manager (recommended)

### Install
```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/media-compressor.git
cd media-compressor

# Install dependencies
uv sync
```

## 🚀 Usage

### Basic Usage
```bash
# Compress all videos and images in a folder
uv run main.py /path/to/media_folder

# Automatic hardware detection (Recommended)
uv run main.py --auto-config /path/to/media_folder
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
```

### All Options

| Flag | Description | Default |
|------|-------------|---------|
| `--output-path`, `-o` | Output directory or file path | `{input}_compressed` |
| `--delete-source` | Delete original file after successful compression | `False` |
| `--auto-config` | Auto-detect optimal CPU/GPU worker counts | `False` |
| `--cpu-workers` | Number of CPU encoding workers | `0` |
| `--gpu-workers` | Number of GPU encoding workers (NVENC) | `0` |

#### Video Options
| Flag | Description | Default |
|------|-------------|---------|
| `--crf` | CRF/CQ value (lower = better quality, larger file) | `23` |
| `--preset` | Encoding preset (ultrafast → veryslow) | `medium` |

#### Image Options
| Flag | Description | Default |
|------|-------------|---------|
| `--image-quality` | JPEG/WebP quality (1-100) | `90` |
| `--keep-format` | Keep original format instead of converting to WebP | `False` |

#### Other Options
| Flag | Description | Default |
|------|-------------|---------|
| `--no-scan-duration` | Skip pre-scanning video durations (faster startup) | `False` |

## 📊 How It Works

```
Input File
    ↓
[1] Check if output exists → Skip
    ↓
[2] Check codec (AV1/H.265?) → Move/Copy original → Skip
    ↓
[3] Video > 60s? → Sample 3 points → Estimate ratio
    ↓
    Ratio ≥ 0.95? → Move/Copy original → Skip
    ↓
[4] Compress
    ↓
[5] Compare sizes
    ↓
    Larger? → Discard, keep original
    ↓
    Smaller? → Use compressed ✓
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

MIT License - see [LICENSE](LICENSE) for details.
