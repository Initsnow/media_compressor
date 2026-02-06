import os
import subprocess
import sys
import re
import shutil
import signal
import tempfile
import logging
from pathlib import Path
from enum import Enum
from dataclasses import dataclass
from typing import Optional, List, Dict, Tuple
import click
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn, TimeRemainingColumn
from rich.prompt import Confirm
from rich.table import Table
import concurrent.futures
import queue
import threading
import time
import json
import hashlib
from PIL import Image

console = Console()
shutdown_event = threading.Event()
pause_event = threading.Event()
interrupt_count = 0
last_interrupt_time = 0

# Configure logging
logger = logging.getLogger("media_compressor")

# =============================================================================
# Constants and Enums
# =============================================================================

class EncoderType(Enum):
    """Available video encoder types."""
    CPU = "cpu"          # libx265
    NVENC = "nvenc"      # NVIDIA
    VAAPI = "vaapi"      # AMD/Intel VAAPI
    QSV = "qsv"          # Intel Quick Sync
    AUTO = "auto"        # Auto-detect best

@dataclass
class EncoderInfo:
    """Information about an available encoder."""
    encoder_type: EncoderType
    ffmpeg_encoder: str
    display_name: str
    available: bool = False
    device_path: Optional[str] = None  # For VAAPI

# Supported video extensions
VIDEO_EXTENSIONS = {'.mp4', '.mkv', '.mov', '.avi', '.flv', '.wmv', '.ts', '.webm', '.m4v', '.3gp', '.ogv'}

# Supported image extensions
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp', '.gif'}

# Valid x265 presets
VALID_PRESETS = ['ultrafast', 'superfast', 'veryfast', 'faster', 'fast', 'medium', 'slow', 'slower', 'veryslow', 'placebo']

# =============================================================================
# Hardware Detection
# =============================================================================

def check_command_exists(cmd: str) -> bool:
    """Check if a command exists in PATH."""
    return shutil.which(cmd) is not None

def check_ffmpeg_encoder_support(encoder: str) -> bool:
    """Check if FFmpeg was compiled with support for a specific encoder."""
    try:
        result = subprocess.run(
            ["ffmpeg", "-hide_banner", "-encoders"],
            capture_output=True, text=True, timeout=10
        )
        return encoder in result.stdout
    except Exception:
        return False

def detect_nvidia_gpu() -> bool:
    """Detect if NVIDIA GPU is available."""
    if not check_command_exists("nvidia-smi"):
        return False
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10
        )
        return result.returncode == 0 and result.stdout.strip() != ""
    except Exception:
        return False

def detect_vaapi_device() -> Optional[str]:
    """
    Detect VAAPI device (AMD or Intel).
    Returns device path like /dev/dri/renderD128 or None.
    """
    # Check for render nodes
    dri_path = Path("/dev/dri")
    if not dri_path.exists():
        return None
    
    render_devices = sorted(dri_path.glob("renderD*"))
    if not render_devices:
        return None
    
    # Check if vainfo works on the device
    for device in render_devices:
        device_str = str(device)
        try:
            # Check if we have permission
            if not os.access(device_str, os.R_OK | os.W_OK):
                continue
                
            # Try vainfo if available
            if check_command_exists("vainfo"):
                result = subprocess.run(
                    ["vainfo", "--display", "drm", "--device", device_str],
                    capture_output=True, text=True, timeout=10
                )
                # Check if HEVC encoding is supported
                if result.returncode == 0 and "VAProfileHEVC" in result.stdout:
                    return device_str
            else:
                # If vainfo not available, check via ffmpeg
                result = subprocess.run(
                    ["ffmpeg", "-hide_banner", "-init_hw_device", f"vaapi=va:{device_str}",
                     "-f", "lavfi", "-i", "nullsrc", "-t", "0.1", "-c:v", "hevc_vaapi", "-f", "null", "-"],
                    capture_output=True, text=True, timeout=10
                )
                if result.returncode == 0:
                    return device_str
        except Exception:
            continue
    
    return None

def detect_qsv() -> bool:
    """Detect if Intel QSV is available."""
    if not check_ffmpeg_encoder_support("hevc_qsv"):
        return False
    
    # Try a quick encode to verify QSV works
    try:
        result = subprocess.run(
            ["ffmpeg", "-hide_banner", "-f", "lavfi", "-i", "nullsrc=s=64x64", 
             "-t", "0.1", "-c:v", "hevc_qsv", "-f", "null", "-"],
            capture_output=True, text=True, timeout=10
        )
        return result.returncode == 0
    except Exception:
        return False

def detect_available_encoders() -> Dict[EncoderType, EncoderInfo]:
    """
    Detect all available hardware encoders.
    Returns a dict of EncoderType -> EncoderInfo.
    """
    encoders = {}
    
    # CPU is always available if libx265 is compiled in
    cpu_available = check_ffmpeg_encoder_support("libx265")
    encoders[EncoderType.CPU] = EncoderInfo(
        encoder_type=EncoderType.CPU,
        ffmpeg_encoder="libx265",
        display_name="CPU (libx265)",
        available=cpu_available
    )
    
    # Check NVIDIA NVENC
    nvenc_available = detect_nvidia_gpu() and check_ffmpeg_encoder_support("hevc_nvenc")
    encoders[EncoderType.NVENC] = EncoderInfo(
        encoder_type=EncoderType.NVENC,
        ffmpeg_encoder="hevc_nvenc",
        display_name="NVIDIA NVENC",
        available=nvenc_available
    )
    
    # Check VAAPI (AMD/Intel)
    vaapi_device = detect_vaapi_device()
    vaapi_available = vaapi_device is not None and check_ffmpeg_encoder_support("hevc_vaapi")
    encoders[EncoderType.VAAPI] = EncoderInfo(
        encoder_type=EncoderType.VAAPI,
        ffmpeg_encoder="hevc_vaapi",
        display_name="VAAPI (AMD/Intel)",
        available=vaapi_available,
        device_path=vaapi_device
    )
    
    # Check Intel QSV
    qsv_available = detect_qsv()
    encoders[EncoderType.QSV] = EncoderInfo(
        encoder_type=EncoderType.QSV,
        ffmpeg_encoder="hevc_qsv",
        display_name="Intel Quick Sync",
        available=qsv_available
    )
    
    return encoders

def select_best_encoder(encoders: Dict[EncoderType, EncoderInfo]) -> EncoderType:
    """
    Select the best available encoder.
    Priority: NVENC > VAAPI > QSV > CPU
    """
    priority_order = [EncoderType.NVENC, EncoderType.VAAPI, EncoderType.QSV, EncoderType.CPU]
    
    for encoder_type in priority_order:
        if encoders.get(encoder_type, EncoderInfo(EncoderType.CPU, "", "", False)).available:
            return encoder_type
    
    return EncoderType.CPU

def print_encoder_table(encoders: Dict[EncoderType, EncoderInfo]):
    """Print a table of available encoders."""
    table = Table(title="Available Encoders")
    table.add_column("Encoder", style="cyan")
    table.add_column("Status", style="green")
    table.add_column("FFmpeg Encoder")
    table.add_column("Notes")
    
    for encoder_type in [EncoderType.NVENC, EncoderType.VAAPI, EncoderType.QSV, EncoderType.CPU]:
        info = encoders.get(encoder_type)
        if info:
            status = "✓ Available" if info.available else "✗ Not Available"
            status_style = "green" if info.available else "red"
            notes = ""
            if info.device_path:
                notes = f"Device: {info.device_path}"
            table.add_row(
                info.display_name,
                f"[{status_style}]{status}[/{status_style}]",
                info.ffmpeg_encoder,
                notes
            )
    
    console.print(table)

# =============================================================================
# Pre-flight Checks
# =============================================================================

def check_dependencies() -> List[str]:
    """
    Check for required dependencies.
    Returns list of error messages, empty if all OK.
    """
    errors = []
    
    if not check_command_exists("ffmpeg"):
        errors.append("FFmpeg not found in PATH. Please install FFmpeg.")
    
    if not check_command_exists("ffprobe"):
        errors.append("FFprobe not found in PATH. Please install FFmpeg.")
    
    return errors

def validate_crf(crf: int, encoder_type: EncoderType) -> Tuple[bool, str]:
    """Validate CRF value for the encoder type."""
    if encoder_type == EncoderType.CPU:
        if not 0 <= crf <= 51:
            return False, f"CRF must be between 0 and 51 for libx265 (got {crf})"
    elif encoder_type == EncoderType.NVENC:
        if not 0 <= crf <= 51:
            return False, f"CQ must be between 0 and 51 for NVENC (got {crf})"
    elif encoder_type == EncoderType.VAAPI:
        if not 0 <= crf <= 52:
            return False, f"QP must be between 0 and 52 for VAAPI (got {crf})"
    elif encoder_type == EncoderType.QSV:
        if not 1 <= crf <= 51:
            return False, f"Global quality must be between 1 and 51 for QSV (got {crf})"
    return True, ""

def validate_preset(preset: str) -> Tuple[bool, str]:
    """Validate preset value."""
    if preset.lower() not in VALID_PRESETS:
        return False, f"Invalid preset '{preset}'. Valid options: {', '.join(VALID_PRESETS)}"
    return True, ""

def validate_image_quality(quality: int) -> Tuple[bool, str]:
    """Validate image quality value."""
    if not 1 <= quality <= 100:
        return False, f"Image quality must be between 1 and 100 (got {quality})"
    return True, ""

def check_file_readable(path: Path) -> bool:
    """Check if a file is readable."""
    return path.exists() and os.access(path, os.R_OK)

def check_directory_writable(path: Path) -> bool:
    """Check if a directory is writable."""
    if path.exists():
        return os.access(path, os.W_OK)
    # Check parent directory
    parent = path.parent
    return parent.exists() and os.access(parent, os.W_OK)

def get_available_disk_space(path: Path) -> int:
    """Get available disk space in bytes for the given path."""
    try:
        stat = os.statvfs(path if path.exists() else path.parent)
        return stat.f_frsize * stat.f_bavail
    except Exception:
        return 0

def is_file_corrupted(file_path: Path) -> bool:
    """
    Check if a media file is corrupted.
    Uses PIL for images (verifies integrity) and ffprobe for videos.
    Returns True if corrupted, False if OK.
    """
    try:
        if file_path.stat().st_size == 0:
            return True
            
        # Use PIL for images
        if file_path.suffix.lower() in IMAGE_EXTENSIONS:
            try:
                with Image.open(file_path) as img:
                    img.verify()
                return False
            except Exception:
                return True

        # Use ffprobe for videos
        result = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0", 
             "-show_entries", "stream=codec_name", "-of", "csv=p=0", str(file_path)],
            capture_output=True, text=True, timeout=30
        )
        return result.returncode != 0 or not result.stdout.strip()
    except Exception:
        return True

def verify_file_magic(file_path: Path) -> Optional[str]:
    """
    Verify file type using magic bytes.
    Returns detected extension or None if unknown.
    """
    magic_signatures = {
        b'\xff\xd8\xff': '.jpg',
        b'\x89PNG\r\n\x1a\n': '.png',
        b'GIF87a': '.gif',
        b'GIF89a': '.gif',
        b'RIFF': '.webp',  # Need to check for WEBP after RIFF
        b'BM': '.bmp',
        b'\x00\x00\x00\x1cftyp': '.mp4',
        b'\x00\x00\x00\x20ftyp': '.mp4',
        b'\x1aE\xdf\xa3': '.mkv',
        b'\x00\x00\x01\xba': '.mpg',
        b'\x00\x00\x01\xb3': '.mpg',
    }
    
    try:
        with open(file_path, 'rb') as f:
            header = f.read(32)
        
        for magic, ext in magic_signatures.items():
            if header.startswith(magic):
                # Special check for WebP (RIFF....WEBP)
                if magic == b'RIFF' and b'WEBP' in header[:12]:
                    return '.webp'
                elif magic == b'RIFF':
                    continue  # Not WebP, might be AVI or WAV
                return ext
        
        # Check ftyp for various video formats
        if b'ftyp' in header[:12]:
            return '.mp4'  # Generic MP4 container
            
    except Exception:
        pass
    
    return None

# =============================================================================
# Statistics Tracking
# =============================================================================

@dataclass
class ProcessingStats:
    """Track processing statistics."""
    files_processed: int = 0
    files_skipped: int = 0
    files_failed: int = 0
    files_resumed: int = 0  # Added
    bytes_saved: int = 0
    original_size: int = 0
    compressed_size: int = 0
    
    lock: threading.Lock = None
    
    def __post_init__(self):
        self.lock = threading.Lock()
    
    def add_success(self, original: int, compressed: int):
        with self.lock:
            self.files_processed += 1
            self.original_size += original
            self.compressed_size += compressed
            self.bytes_saved += (original - compressed)
    
    def add_skipped(self):
        with self.lock:
            self.files_skipped += 1

    def add_resumed(self):
        with self.lock:
            self.files_resumed += 1
    
    def add_failed(self):
        with self.lock:
            self.files_failed += 1

# Global stats instance
stats = ProcessingStats()

# =============================================================================
# Utility Functions
# =============================================================================

def get_gpu_free_memory():
    """Get free VRAM in MiB using nvidia-smi. Returns 0 if failed."""
    try:
        cmd = ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            return int(result.stdout.strip().split('\n')[0])
    except Exception:
        pass
    return 0

def get_media_info_json(input_file: Path) -> Dict:
    """
    Get both duration and codec info in one pass using ffprobe json output.
    Returns dict with keys: 'duration' (float), 'codec' (str).
    """
    cmd = [
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=codec_name:format=duration",
        "-of", "json", str(input_file)
    ]
    try:
        # Timeout slightly longer as we are checking more
        result = subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=60)
        data = json.loads(result.stdout)
        
        info = {'duration': 0.0, 'codec': None}
        
        # Get duration (try format, then stream)
        try:
            if 'format' in data and 'duration' in data['format']:
                info['duration'] = float(data['format']['duration'])
        except Exception:
            pass
            
        # Get codec
        try:
            if 'streams' in data and len(data['streams']) > 0:
                info['codec'] = data['streams'][0].get('codec_name', '').lower()
        except Exception:
            pass
            
        return info
    except Exception:
        return {'duration': 0.0, 'codec': None}

def get_duration(input_file: Path) -> Optional[float]:
    """Get video duration in seconds using ffprobe."""
    # Wrapper for legacy or specific calls, but optimally we use the cached info now
    info = get_media_info_json(input_file)
    return info['duration'] if info['duration'] > 0 else None

def get_video_codec(input_file: Path) -> Optional[str]:
    """Get video codec name using ffprobe."""
    info = get_media_info_json(input_file)
    return info['codec']

def convert_time_to_seconds(time_str: str) -> float:
    """Convert HH:MM:SS.ms to seconds."""
    try:
        h, m, s = time_str.split(':')
        return int(h) * 3600 + int(m) * 60 + float(s)
    except (ValueError, AttributeError):
        return 0.0

def format_bytes(size: int) -> str:
    """Return human readable file size string."""
    power = 2**10
    n = 0
    power_labels = {0: '', 1: 'K', 2: 'M', 3: 'G', 4: 'T'}
    while size > power and n < 4:
        size /= power
        n += 1
    return f"{size:.2f} {power_labels[n]}B"

def calculate_saving(old_size: int, new_size: int) -> str:
    """Return saving string and percentage."""
    if old_size == 0:
        return "0 B (0%)"
    saved = old_size - new_size
    percent = (saved / old_size) * 100
    return f"{format_bytes(saved)} ({percent:.1f}%)"

def clean_partial_files(directory: Path):
    """Clean up any leftover .part files from previous runs."""
    if not directory.exists():
        return
    
    part_files = list(directory.rglob("*.part"))
    if part_files:
        console.print(f"[yellow]Cleaning up {len(part_files)} leftover temporary files...[/yellow]")
        for f in part_files:
            try:
                f.unlink()
            except Exception:
                pass

# =============================================================================
# Compression Estimation
# =============================================================================

def estimate_compression_ratio(input_file: Path, crf: int, preset: str, encoder_info: EncoderInfo, sample_seconds: int = 10) -> Optional[float]:
    """
    Encode short samples from multiple positions to estimate compression ratio.
    Returns estimated ratio (new_size / old_size). Values < 1.0 mean compression saves space.
    Returns None if estimation fails.
    """
    duration = get_duration(input_file)
    if not duration or duration < sample_seconds * 2:
        return None
    
    original_size = input_file.stat().st_size
    original_bitrate = original_size / duration
    
    sample_positions = [duration * 0.25, duration * 0.50, duration * 0.75]
    ratios = []
    
    for start_pos in sample_positions:
        if start_pos + sample_seconds > duration:
            start_pos = max(0, duration - sample_seconds)
        
        expected_sample_size = original_bitrate * sample_seconds
        
        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as tmp:
            temp_path = tmp.name
        
        try:
            cmd = build_ffmpeg_command(
                input_file, Path(temp_path), crf, preset, encoder_info,
                start_time=start_pos, duration=sample_seconds, no_audio=True
            )
            
            result = subprocess.run(cmd, capture_output=True, timeout=60)
            
            if result.returncode == 0 and Path(temp_path).exists():
                sample_size = Path(temp_path).stat().st_size
                ratio = sample_size / expected_sample_size
                ratios.append(ratio)
        except Exception:
            pass
        finally:
            try:
                Path(temp_path).unlink()
            except Exception:
                pass
    
    if ratios:
        return sum(ratios) / len(ratios)
    
    return None

# =============================================================================
# FFmpeg Command Building
# =============================================================================

def build_ffmpeg_command(
    input_file: Path,
    output_file: Path,
    crf: int,
    preset: str,
    encoder_info: EncoderInfo,
    start_time: Optional[float] = None,
    duration: Optional[float] = None,
    no_audio: bool = False
) -> List[str]:
    """Build FFmpeg command for the specified encoder."""
    cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-stats"]
    
    # VAAPI requires device initialization before input
    if encoder_info.encoder_type == EncoderType.VAAPI and encoder_info.device_path:
        cmd.extend(["-vaapi_device", encoder_info.device_path])
    
    # Time seeking (before input for speed)
    if start_time is not None:
        cmd.extend(["-ss", str(start_time)])
    if duration is not None:
        cmd.extend(["-t", str(duration)])
    
    cmd.extend(["-i", str(input_file)])
    
    # Encoder-specific settings
    if encoder_info.encoder_type == EncoderType.CPU:
        cmd.extend([
            "-c:v", "libx265",
            "-crf", str(crf),
            "-preset", preset
        ])
    elif encoder_info.encoder_type == EncoderType.NVENC:
        nv_preset = "p4"
        if preset in ["ultrafast", "superfast", "veryfast", "faster", "fast"]:
            nv_preset = "p1"
        elif preset in ["slow", "slower", "veryslow", "placebo"]:
            nv_preset = "p7"
        cmd.extend([
            "-c:v", "hevc_nvenc",
            "-rc", "vbr",
            "-cq", str(crf),
            "-preset", nv_preset
        ])
    elif encoder_info.encoder_type == EncoderType.VAAPI:
        cmd.extend([
            "-vf", "format=nv12,hwupload",
            "-c:v", "hevc_vaapi",
            "-qp", str(crf)
        ])
    elif encoder_info.encoder_type == EncoderType.QSV:
        qsv_preset = "medium"
        if preset in ["ultrafast", "superfast", "veryfast"]:
            qsv_preset = "veryfast"
        elif preset in ["slower", "veryslow", "placebo"]:
            qsv_preset = "veryslow"
        cmd.extend([
            "-c:v", "hevc_qsv",
            "-global_quality", str(crf),
            "-preset", qsv_preset
        ])
    
    # Audio
    if no_audio:
        cmd.append("-an")
    else:
        cmd.extend(["-c:a", "copy"])
    
    cmd.extend(["-f", "mp4", str(output_file)])
    
    return cmd

# =============================================================================
# State Management (Resume Functionality)
# =============================================================================

class StateManager:
    """Manages the state of processed files for resume capability."""
    
    def __init__(self, output_path: Path):
        self.state_file = output_path / ".media_compressor_state.json"
        self.lock = threading.Lock()
        self.processed_files = set()
        self.load()
        
    def load(self):
        """Load state from file."""
        if self.state_file.exists():
            try:
                with open(self.state_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.processed_files = set(data.get("processed_files", []))
                # logger.info(f"Loaded state: {len(self.processed_files)} files previously processed.")
                if self.processed_files:
                    console.print(f"[green]Resume: Loaded {len(self.processed_files)} previously processed files.[/green]")
            except Exception as e:
                logger.warning(f"Failed to load state file: {e}")
                
    def save(self):
        """Save state to file."""
        if not self.state_file.parent.exists():
            return
            
        with self.lock:
            try:
                temp_file = self.state_file.with_suffix('.tmp')
                with open(temp_file, 'w', encoding='utf-8') as f:
                    json.dump({
                        "processed_files": list(self.processed_files),
                        "last_updated": time.time()
                    }, f, indent=2)
                temp_file.replace(self.state_file)
            except Exception as e:
                logger.warning(f"Failed to save state file: {e}")
                
    def mark_processed(self, file_path: str):
        """Mark a file as processed (store relative path string)."""
        with self.lock:
            self.processed_files.add(str(file_path))
            # Auto-save every 50 files? Or rely on periodic save?
            # For simplicity, we might save in the worker loop occasionally,
            # but frequent IO might be bad. Let's just update memory and save on exit/interrupt.
            
    def is_processed(self, file_path: str) -> bool:
        """Check if file has been processed."""
        return str(file_path) in self.processed_files

# =============================================================================
# Video Processing
# =============================================================================

def scan_media_info_parallel(files: List[Path], console: Console) -> Tuple[float, Dict[Path, Dict]]:
    """
    Scan all files for duration and codec concurrently.
    Returns: (total_seconds, media_cache)
    """
    total_seconds = 0.0
    media_cache = {}
    
    with console.status("[bold green]Scanning files for metadata (Parallel)...[/bold green]") as status:
        # Use more workers for IO bound probe
        max_workers = (os.cpu_count() or 4) * 2
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_file = {executor.submit(get_media_info_json, f): f for f in files}
            
            completed_count = 0
            total_files = len(files)
            
            for future in concurrent.futures.as_completed(future_to_file):
                f = future_to_file[future]
                if shutdown_event.is_set():
                    executor.shutdown(wait=False, cancel_futures=True)
                    break
                
                completed_count += 1
                if completed_count % 10 == 0:
                    status.update(f"[bold green]Scanning {completed_count}/{total_files}...[/bold green]")
                
                try:
                    info = future.result()
                    media_cache[f] = info
                    if info['duration']:
                        total_seconds += info['duration']
                except Exception:
                    pass
    
    return total_seconds, media_cache

def process_video(
    in_file: Path,
    input_base_path: Path,
    output_base_path: Path,
    crf: int,
    preset: str,
    delete_source: bool,
    encoder_info: EncoderInfo,
    progress,
    total_task_id,
    global_mode: str = "time",
    timeout: Optional[int] = None,
    state_manager: Optional[StateManager] = None,
    media_cache: Optional[Dict] = None
):
    """Process a single video file."""
    if shutdown_event.is_set():
        return

    worker_type = encoder_info.display_name
    
    # Calculate relative path string for state tracking
    rel_path_str = str(in_file.relative_to(input_base_path)) if input_base_path.is_dir() else in_file.name
    
    # Calculate target output file path
    if input_base_path.is_dir():
        rel_path = in_file.relative_to(input_base_path)
        out_file = output_base_path / rel_path.with_suffix('.mp4')
    else:
        out_file = output_base_path
    
    out_file.parent.mkdir(parents=True, exist_ok=True)

    # Check state first if resume is enabled
    if state_manager and state_manager.is_processed(rel_path_str):
        stats.add_resumed()
        if global_mode == "count":
            progress.advance(total_task_id, advance=1)
        elif global_mode == "time":
             # Try to get duration from cache to advance progress
             cached_duration = 0
             if media_cache and in_file in media_cache:
                 cached_duration = media_cache[in_file].get('duration', 0)
             
             if cached_duration > 0:
                 progress.advance(total_task_id, advance=cached_duration)
             pass
        
        # Check if we should delete source even if skipping
        if delete_source and in_file.exists():
            if out_file.exists():
                try:
                    in_file.unlink()
                    progress.console.print(f"[bold yellow]Deleted source (resume): {in_file.name}[/bold yellow]")
                except Exception as e:
                    progress.console.print(f"[bold red]Failed to delete source: {e}[/bold red]")
            else:
                progress.console.print(f"[bold red]Cannot delete source (resume): Output {out_file.name} missing[/bold red]")
        return

    # Compare resolved paths to handle in-place processing correctly
    try:
        is_same_file = in_file.resolve() == out_file.resolve()
    except Exception:
        is_same_file = False

    if out_file.exists() and not is_same_file:
        logger.debug(f"Skipping {in_file.name} - output exists")
        if state_manager:
            state_manager.mark_processed(rel_path_str)
        stats.add_skipped() # Count as skipped (already exists)
        
        # Check if we should delete source
        if delete_source and in_file.exists():
            try:
                in_file.unlink()
                progress.console.print(f"[bold yellow]Deleted source (exists): {in_file.name}[/bold yellow]")
            except Exception as e:
                progress.console.print(f"[bold red]Failed to delete source: {e}[/bold red]")
        return

    # Check if file is readable
    if not check_file_readable(in_file):
        progress.console.print(f"[bold red]✗ Cannot read {in_file.name} - permission denied[/bold red]")
        stats.add_failed()
        return

    # Check for corrupted files
    if is_file_corrupted(in_file):
        progress.console.print(f"[bold red]✗ Skipping {in_file.name} - file appears corrupted[/bold red]")
        stats.add_failed()
        return

    # Check if video is already using an efficient codec
    codec = None
    if media_cache and in_file in media_cache:
        codec = media_cache[in_file].get('codec')
    
    if not codec:
        codec = get_video_codec(in_file)
        
    if codec in ['hevc', 'h265', 'av1']:
        progress.console.print(f"[bold blue]⊘ Skipping {in_file.name} (Already {codec.upper()} encoded)[/bold blue]")
        old_size = in_file.stat().st_size
        
        # Use existing duration for progress
        duration = 0
        if media_cache and in_file in media_cache:
            duration = media_cache[in_file].get('duration', 0)
        if not duration:
            duration = get_duration(in_file)
        
        # Update progress before skipping
        if global_mode == "time" and duration:
            progress.advance(total_task_id, advance=duration)
        elif global_mode == "count":
            progress.advance(total_task_id, advance=1)
            
        # Use original extension if we keep the original file
        final_out_file = out_file
        if final_out_file.suffix.lower() != in_file.suffix.lower():
            final_out_file = out_file.with_suffix(in_file.suffix)
            
        if final_out_file.exists():
            final_out_file.unlink()

        if delete_source:
            shutil.move(in_file, final_out_file)
        else:
            shutil.copy2(in_file, final_out_file)
        stats.add_success(old_size, old_size)  # No size change
        if global_mode == "count":
            progress.advance(total_task_id, advance=1)
        elif global_mode == "time":
            # Use cached duration if available
            duration = 0
            if media_cache and in_file in media_cache:
                duration = media_cache[in_file].get('duration', 0)
            
            if not duration:
                duration = get_duration(in_file)
                
            if duration:
                progress.advance(total_task_id, advance=duration)
        
        if state_manager:
            state_manager.mark_processed(rel_path_str)
        return

    # Get duration for processing
    duration = 0
    if media_cache and in_file in media_cache:
        duration = media_cache[in_file].get('duration', 0)
    
    if not duration:
        duration = get_duration(in_file)
    
    # Pre-check: estimate compression ratio
    if duration and duration > 60:
        estimated_ratio = estimate_compression_ratio(in_file, crf, preset, encoder_info)
        if estimated_ratio is not None and estimated_ratio >= 0.95:
            progress.console.print(f"[bold yellow]⊘ Skipping {in_file.name} (Estimated ratio: {estimated_ratio:.2f}x - not worth compressing)[/bold yellow]")
            old_size = in_file.stat().st_size
            
            # Use original extension if we keep the original file
            final_out_file = out_file
            if final_out_file.suffix.lower() != in_file.suffix.lower():
                final_out_file = out_file.with_suffix(in_file.suffix)
                
            if final_out_file.exists():
                final_out_file.unlink()

            if delete_source:
                shutil.move(in_file, final_out_file)
            else:
                shutil.copy2(in_file, final_out_file)
            stats.add_skipped()
            if global_mode == "count":
                progress.advance(total_task_id, advance=1)
            elif global_mode == "time":
                progress.advance(total_task_id, advance=duration)
            
            if state_manager:
                state_manager.mark_processed(rel_path_str)
            return

    temp_out_file = out_file.with_suffix('.mp4.part')
    
    cmd = build_ffmpeg_command(in_file, temp_out_file, crf, preset, encoder_info)

    # Determine task color based on encoder type
    color_map = {
        EncoderType.NVENC: "magenta",
        EncoderType.VAAPI: "cyan",
        EncoderType.QSV: "blue",
        EncoderType.CPU: "green"
    }
    task_color = color_map.get(encoder_info.encoder_type, "white")
    
    task_id = progress.add_task(f"[{task_color}]{worker_type}: {in_file.name}", total=duration if duration else None)
    
    error_log = []
    last_reported_seconds = 0.0
    
    try:
        process = subprocess.Popen(
            cmd,
            stderr=subprocess.PIPE,
            stdout=subprocess.PIPE,
            universal_newlines=True,
            encoding="utf-8",
            errors="replace"
        )
        
        time_pattern = re.compile(r"time=(\d{2}:\d{2}:\d{2}\.\d+)")
        start_time = time.time()
        
        while True:
            if shutdown_event.is_set():
                process.terminate()
                break
            
            # Timeout check
            if timeout and (time.time() - start_time) > timeout:
                process.terminate()
                progress.console.print(f"[bold yellow]⏱ Timeout processing {in_file.name}[/bold yellow]")
                stats.add_failed()
                break
            
            while pause_event.is_set():
                time.sleep(0.1)

            line = process.stderr.readline()
            if not line and process.poll() is not None:
                break
            
            if line:
                error_log.append(line)
                if len(error_log) > 20:
                    error_log.pop(0)

                match = time_pattern.search(line)
                if match and duration:
                    current_time_str = match.group(1)
                    current_seconds = convert_time_to_seconds(current_time_str)
                    
                    progress.update(task_id, completed=current_seconds)
                    
                    delta = current_seconds - last_reported_seconds
                    if delta > 0 and global_mode == "time":
                        progress.advance(total_task_id, advance=delta)
                        last_reported_seconds = current_seconds
        
        return_code = process.wait()
        
        if shutdown_event.is_set():
            progress.console.print(f"[yellow]Stopped {in_file.name}[/yellow]")
            if temp_out_file.exists():
                temp_out_file.unlink()
            progress.stop_task(task_id)
            return

        if return_code == 0:
            old_size = in_file.stat().st_size
            new_size = temp_out_file.stat().st_size
            
            # Safety check: if compressed file is larger, discard it
            if new_size >= old_size:
                temp_out_file.unlink()
                
                # Use original extension if we keep the original file
                final_out_file = out_file
                if final_out_file.suffix.lower() != in_file.suffix.lower():
                    final_out_file = out_file.with_suffix(in_file.suffix)
                    
                if final_out_file.exists():
                    final_out_file.unlink()

                if delete_source:
                    shutil.move(in_file, final_out_file)
                else:
                    shutil.copy2(in_file, final_out_file)
                progress.console.print(f"[bold yellow]⚠ Kept original {in_file.name} (Compressed size larger: {format_bytes(old_size)} → {format_bytes(new_size)})[/bold yellow]")
                stats.add_success(old_size, old_size)
                
                if global_mode == "time":
                    final_delta = (duration if duration else 0) - last_reported_seconds
                    if final_delta > 0:
                        progress.advance(total_task_id, advance=final_delta)
                else:
                    progress.advance(total_task_id, advance=1)
                
                progress.update(task_id, completed=duration if duration else 100)
                progress.stop_task(task_id)
                if state_manager:
                    state_manager.mark_processed(rel_path_str)
                return
            
            temp_out_file.rename(out_file)
            stats.add_success(old_size, new_size)
            
            if global_mode == "time":
                final_delta = (duration if duration else 0) - last_reported_seconds
                if final_delta > 0:
                    progress.advance(total_task_id, advance=final_delta)
            else:
                progress.advance(total_task_id, advance=1)
            
            progress.update(task_id, completed=duration if duration else 100)
            progress.stop_task(task_id)
            if state_manager:
                state_manager.mark_processed(rel_path_str)
            
            saving_str = calculate_saving(old_size, new_size)
            progress.console.print(f"[bold green]✓ Finished ({worker_type}): {out_file.name} [dim](Saved: {saving_str})[/dim][/bold green]")
            
            if delete_source:
                try:
                    in_file.unlink()
                    progress.console.print(f"[bold yellow]Deleted source: {in_file.name}[/bold yellow]")
                except Exception as e:
                    progress.console.print(f"[bold red]Failed to delete source: {e}[/bold red]")
        else:
            progress.console.print(f"[bold red]✗ Error processing {in_file.name} (Exit code: {return_code})[/bold red]")
            progress.console.print(f"[red]Last ffmpeg output for {in_file.name}:[/red]")
            for err_line in error_log:
                progress.console.print(f"[dim]{err_line.strip()}[/dim]")
            
            stats.add_failed()
            if temp_out_file.exists():
                temp_out_file.unlink()
            progress.stop_task(task_id)

    except Exception as e:
        progress.console.print(f"[bold red]Exception {in_file.name}: {e}[/bold red]")
        stats.add_failed()
        if temp_out_file.exists():
            temp_out_file.unlink()
        progress.stop_task(task_id)

# =============================================================================
# Image Processing
# =============================================================================

def process_image(
    in_file: Path,
    input_base_path: Path,
    output_base_path: Path,
    image_quality: int,
    keep_format: bool,
    delete_source: bool,
    progress,
    total_task_id,
    global_mode: str = "time",
    state_manager: Optional[StateManager] = None # Added argument
):
    """Process a single image file."""
    if shutdown_event.is_set():
        return
        
    # Calculate relative path string for state tracking
    rel_path_str = str(in_file.relative_to(input_base_path)) if input_base_path.is_dir() else in_file.name
    
    # Calculate target output file path
    if input_base_path.is_dir():
        rel_path = in_file.relative_to(input_base_path)
        if keep_format:
            out_file = output_base_path / rel_path
        else:
            out_file = output_base_path / rel_path.with_suffix('.webp')
    else:
        out_file = output_base_path
    
    out_file.parent.mkdir(parents=True, exist_ok=True)

    # Check state first if resume is enabled
    if state_manager and state_manager.is_processed(rel_path_str):
        stats.add_resumed()
        if global_mode != "time":
            progress.advance(total_task_id, advance=1)
        
        # Check if we should delete source even if skipping
        if delete_source and in_file.exists():
            if out_file.exists():
                try:
                    in_file.unlink()
                    progress.console.print(f"[bold yellow]Deleted source (resume): {in_file.name}[/bold yellow]")
                except Exception as e:
                    progress.console.print(f"[bold red]Failed to delete source: {e}[/bold red]")
            else:
                 # Debug: print why we didn't delete
                 pass
        return

    if out_file.exists():
        if state_manager:
            state_manager.mark_processed(rel_path_str)
        
        # Check if we should delete source
        if delete_source and in_file.exists():
            try:
                in_file.unlink()
                progress.console.print(f"[bold yellow]Deleted source (exists): {in_file.name}[/bold yellow]")
            except Exception as e:
                progress.console.print(f"[bold red]Failed to delete source: {e}[/bold red]")
        return

    # Check if file is readable
    if not check_file_readable(in_file):
        progress.console.print(f"[bold red]✗ Cannot read {in_file.name} - permission denied[/bold red]")
        stats.add_failed()
        return

    task_id = progress.add_task(f"[green]IMG: {in_file.name}", total=None)
    
    try:
        with Image.open(in_file) as img:
            save_kwargs = {}
            output_format = out_file.suffix.lower().lstrip('.')
            source_format = in_file.suffix.lower().lstrip('.')
            
            if output_format == 'webp':
                if source_format in ['png', 'bmp', 'tiff']:
                    save_kwargs = {'lossless': True, 'method': 6}
                else:
                    save_kwargs = {'quality': image_quality, 'method': 6}
                
                # Check for animation
                if getattr(img, "is_animated", False):
                    save_kwargs.update({
                        'save_all': True,
                        'duration': img.info.get('duration', 100),
                        'loop': 0
                    })

            elif output_format in ['jpg', 'jpeg']:
                if img.mode == 'RGBA':
                    img = img.convert('RGB')
                save_kwargs = {'quality': image_quality, 'optimize': True}
            elif output_format == 'png':
                save_kwargs = {'optimize': True}

            img.save(out_file, **save_kwargs)
        
        progress.stop_task(task_id)

        old_size = in_file.stat().st_size
        new_size = out_file.stat().st_size
        
        if new_size >= old_size:
            out_file.unlink()
            
            # Use original extension if we keep the original file
            final_out_file = out_file
            if final_out_file.suffix.lower() != in_file.suffix.lower():
                final_out_file = out_file.with_suffix(in_file.suffix)
                
            # Remove target if it exists (e.g. from a previous failed run or incorrect extension)
            if final_out_file.exists():
                final_out_file.unlink()

            if delete_source:
                shutil.move(in_file, final_out_file)
            else:
                shutil.copy2(in_file, final_out_file)
            
            progress.console.print(f"[bold yellow]⚠ Kept original {in_file.name} (Compressed size larger: {format_bytes(old_size)} → {format_bytes(new_size)})[/bold yellow]")
            stats.add_success(old_size, old_size)
            
            if global_mode != "time":
                progress.advance(total_task_id, advance=1)
            
            if state_manager:
                state_manager.mark_processed(rel_path_str)
            return
        
        stats.add_success(old_size, new_size)
        
        if global_mode != "time":
            progress.advance(total_task_id, advance=1)
        
        if state_manager:
            state_manager.mark_processed(rel_path_str)
        
        saving_str = calculate_saving(old_size, new_size)
        progress.console.print(f"[bold green]✓ Finished (IMG): {out_file.name} [dim](Saved: {saving_str})[/dim][/bold green]")

        if delete_source:
            try:
                in_file.unlink()
                progress.console.print(f"[bold yellow]Deleted source: {in_file.name}[/bold yellow]")
            except Exception as e:
                progress.console.print(f"[bold red]Failed to delete source: {e}[/bold red]")

    except Exception as e:
        progress.console.print(f"[bold red]Exception {in_file.name}: {e}[/bold red]")
        stats.add_failed()
        if out_file.exists():
            try:
                out_file.unlink()
            except Exception:
                pass
        progress.stop_task(task_id)

# =============================================================================
# Worker Loop
# =============================================================================

def worker_loop(
    file_queue: queue.Queue,
    input_path: Path,
    output_path: Path,
    crf: int,
    preset: str,
    image_quality: int,
    keep_format: bool,
    delete_source: bool,
    encoder_info: EncoderInfo,
    progress,
    total_task_id,
    global_mode: str,
    timeout: Optional[int] = None,
    state_manager: Optional[StateManager] = None,
    media_cache: Optional[Dict] = None,
    image_executor: Optional[concurrent.futures.ThreadPoolExecutor] = None
):
    """Continuous worker loop that pulls from queue."""
    # Counter to save state periodically
    tasks_processed_locally = 0
    
    while not shutdown_event.is_set():
        while pause_event.is_set():
            time.sleep(0.1)
        
        try:
            in_file = file_queue.get(timeout=1)
        except queue.Empty:
            if file_queue.empty():
                return
            continue

        if in_file.suffix.lower() in IMAGE_EXTENSIONS:
            if image_executor:
                # Submit to image pool
                # Note: We don't wait for result here strictly to keep main loop fast,
                # but we need to manage queue task_done.
                # Actually, if we submit and continue, we might flood the image pool if queue is huge.
                # But since we pull from queue one by one, we are limited by how fast we pull.
                # To allow proper concurrency, we should submit and go to next.
                # BUT, we need to mark queue.task_done() only when the image task is ACTUALLY done.
                
                def wrapped_process_image():
                    try:
                        process_image(
                            in_file, input_path, output_path, image_quality, keep_format, 
                            delete_source, progress, total_task_id, global_mode, state_manager
                        )
                    finally:
                        file_queue.task_done()

                image_executor.submit(wrapped_process_image)
                # Skip the default task_done at bottom of loop since we do it in wrapper
                continue
            else:
                # Fallback to sync processing
                process_image(
                    in_file, input_path, output_path, image_quality, keep_format, 
                    delete_source, progress, total_task_id, global_mode, state_manager
                )
        else:
            process_video(
                in_file, input_path, output_path, crf, preset, delete_source, 
                encoder_info, progress, total_task_id, global_mode, timeout, state_manager, media_cache
            )
        
        if state_manager:
            # Always save state after a video (they take a long time)
            # For images, save every 50 to avoid IO thrashing
            is_video = in_file.suffix.lower() not in IMAGE_EXTENSIONS
            tasks_processed_locally += 1
            
            if is_video or tasks_processed_locally >= 50:
                state_manager.save()
                tasks_processed_locally = 0
                
        file_queue.task_done()

# =============================================================================
# Summary Report
# =============================================================================

def print_summary_report():
    """Print final processing summary."""
    console.print("\n" + "=" * 60)
    console.print("[bold cyan]Processing Summary[/bold cyan]")
    console.print("=" * 60)
    
    table = Table(show_header=False, box=None)
    table.add_column("Label", style="dim")
    table.add_column("Value", style="bold")
    
    table.add_row("Files Processed:", str(stats.files_processed))
    table.add_row("Files Resumed:", str(stats.files_resumed))
    table.add_row("Files Skipped:", str(stats.files_skipped))
    table.add_row("Files Failed:", str(stats.files_failed))
    table.add_row("", "")
    table.add_row("Original Size:", format_bytes(stats.original_size))
    table.add_row("Compressed Size:", format_bytes(stats.compressed_size))
    
    if stats.bytes_saved > 0:
        percent = (stats.bytes_saved / stats.original_size) * 100 if stats.original_size > 0 else 0
        table.add_row("Space Saved:", f"[green]{format_bytes(stats.bytes_saved)} ({percent:.1f}%)[/green]")
    elif stats.bytes_saved < 0:
        table.add_row("Space Change:", f"[red]+{format_bytes(abs(stats.bytes_saved))}[/red]")
    else:
        table.add_row("Space Saved:", "0 B")
    
    console.print(table)
    console.print("=" * 60 + "\n")

# =============================================================================
# Main Entry Point
# =============================================================================

@click.command()
@click.argument("input_path", type=click.Path(exists=True), required=False)
@click.option("--output-path", "-o", type=click.Path(), help="Output directory (for folders) or file path.")
@click.option("--crf", default=23, help="CRF/CQ/QP value (default: 23). Lower is better quality.")
@click.option("--preset", default="medium", help="Preset (default: medium). Slower = better compression.")
@click.option("--delete-source", is_flag=True, help="Delete source file after successful compression.")
@click.option("--cpu-workers", default=0, help="Number of CPU workers.")
@click.option("--gpu-workers", default=0, help="Number of GPU/hardware workers.")
@click.option("--auto-config", is_flag=True, help="Automatically determine best worker count based on hardware.")
@click.option("--workers", "-w", default=0, help="Legacy: Total workers (defaults to CPU if GPU not specified).")
@click.option("--gpu", is_flag=True, help="Legacy: Use GPU for all workers if specified (same as --encoder nvenc).")
@click.option("--encoder", type=click.Choice(['auto', 'nvenc', 'vaapi', 'qsv', 'cpu'], case_sensitive=False), 
              default='auto', help="Encoder to use (default: auto-detect best).")
@click.option("--list-encoders", is_flag=True, help="List available encoders and exit.")
@click.option("--image-quality", default=90, help="Image quality (1-100). Default 90 (High).")
@click.option("--keep-format", is_flag=True, help="Keep original image format instead of converting to WebP.")
@click.option("--no-scan-duration", is_flag=True, help="Skip pre-calculating total video duration (Faster startup).")
@click.option("--timeout", default=0, help="Timeout per file in seconds (0 = no timeout).")
@click.option("--no-resume", is_flag=True, help="Disable resume functionality (state tracking).")
@click.option("--log-file", type=click.Path(), help="Write logs to file.")
@click.option("--quiet", "-q", is_flag=True, help="Reduce output verbosity.")
@click.option("--verbose", "-v", is_flag=True, help="Increase output verbosity.")
@click.option("--follow-symlinks", is_flag=True, help="Follow symbolic links when scanning directories.")
def main(input_path, output_path, crf, preset, delete_source, cpu_workers, gpu_workers, 
         auto_config, workers, gpu, encoder, list_encoders, image_quality, keep_format, 
         no_scan_duration, timeout, no_resume, log_file, quiet, verbose, follow_symlinks):
    """
    Batch compress videos and images.
    
    Supports multiple hardware encoders: NVIDIA NVENC, AMD/Intel VAAPI, Intel QSV, and CPU (libx265).
    """
    global stats
    stats = ProcessingStats()

    # Resume is default true unless --no-resume
    resume = not no_resume
    
    # Setup logging
    log_level = logging.WARNING if quiet else (logging.DEBUG if verbose else logging.INFO)
    logging.basicConfig(level=log_level, format='%(asctime)s - %(levelname)s - %(message)s')
    
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        logger.addHandler(file_handler)
    
    # Detect available encoders
    console.print("[dim]Detecting available encoders...[/dim]")
    encoders = detect_available_encoders()
    
    # Handle --list-encoders
    if list_encoders:
        print_encoder_table(encoders)
        return
    
    # Check if input_path is provided
    if not input_path:
        console.print("[bold red]Error: Missing argument 'INPUT_PATH'.[/bold red]")
        console.print("Run with --help for usage information.")
        return
    
    # Check dependencies
    dependency_errors = check_dependencies()
    if dependency_errors:
        for error in dependency_errors:
            console.print(f"[bold red]Error: {error}[/bold red]")
        return
    
    # Validate parameters
    valid, msg = validate_preset(preset)
    if not valid:
        console.print(f"[bold red]Error: {msg}[/bold red]")
        return
    
    valid, msg = validate_image_quality(image_quality)
    if not valid:
        console.print(f"[bold red]Error: {msg}[/bold red]")
        return
    
    input_path = Path(input_path)
    
    # Determine encoder type
    encoder_type = EncoderType.AUTO
    if gpu:  # Legacy flag
        encoder_type = EncoderType.NVENC
    elif encoder.lower() != 'auto':
        encoder_type = EncoderType(encoder.lower())
    
    # Select encoder
    if encoder_type == EncoderType.AUTO:
        selected_encoder_type = select_best_encoder(encoders)
    else:
        selected_encoder_type = encoder_type
        if not encoders[selected_encoder_type].available:
            console.print(f"[bold red]Error: Requested encoder '{encoder}' is not available.[/bold red]")
            print_encoder_table(encoders)
            return
    
    selected_encoder = encoders[selected_encoder_type]
    
    # Validate CRF for selected encoder
    valid, msg = validate_crf(crf, selected_encoder_type)
    if not valid:
        console.print(f"[bold red]Error: {msg}[/bold red]")
        return
    
    # Show selected encoder
    if not quiet:
        if selected_encoder_type == EncoderType.CPU:
            console.print(f"[yellow]Using CPU encoder (libx265) - no hardware acceleration available[/yellow]")
        else:
            console.print(f"[bold green]Using encoder: {selected_encoder.display_name}[/bold green]")
    
    # Auto-config
    if auto_config:
        cpu_count = os.cpu_count() or 4
        calc_cpu = max(1, cpu_count // 8)
        
        if selected_encoder_type == EncoderType.NVENC:
            free_vram = get_gpu_free_memory()
            if free_vram > 4500:
                calc_gpu = 3
            elif free_vram > 1800:
                calc_gpu = 2
            else:
                calc_gpu = 1
            vram_msg = f"(Free VRAM: {free_vram} MiB)" if free_vram else "(VRAM detection failed)"
        else:
            calc_gpu = 1  # Default to 1 for other hardware encoders
            vram_msg = ""
        
        if selected_encoder_type == EncoderType.CPU:
            if cpu_workers == 0:
                cpu_workers = calc_cpu
            gpu_workers = 0
        else:
            if gpu_workers == 0:
                gpu_workers = calc_gpu
            if cpu_workers == 0:
                cpu_workers = 0  # Don't mix CPU and GPU by default
        
        if not quiet:
            console.print(f"[bold cyan]Auto-Config:[/bold cyan] CPU Workers={cpu_workers}, Hardware Workers={gpu_workers} {vram_msg}")
    
    # Legacy flag handling
    if cpu_workers == 0 and gpu_workers == 0:
        if workers == 0:
            workers = 1
        
        if selected_encoder_type == EncoderType.CPU:
            cpu_workers = workers
        else:
            gpu_workers = workers
    
    total_workers = cpu_workers + gpu_workers
    
    # File discovery
    files_to_process = []
    
    if input_path.is_file():
        files_to_process.append(input_path)
        # Determine appropriate extension
        target_ext = '.mp4'
        if input_path.suffix.lower() in IMAGE_EXTENSIONS:
            if keep_format:
                target_ext = input_path.suffix.lower()
            else:
                target_ext = '.webp'
                
        if not output_path:
            output_path = input_path.parent / f"{input_path.stem}_processed{target_ext}"
        else:
            output_path = Path(output_path)
            if output_path.is_dir() or (str(output_path).endswith(os.sep)):
                output_path.mkdir(parents=True, exist_ok=True)
                output_path = output_path / f"{input_path.stem}{target_ext}"
    else:
        if not output_path:
            output_path = input_path.parent / f"{input_path.name}_compressed"
        else:
            output_path = Path(output_path)
        
        # Scan directory
        for root, dirs, files in os.walk(input_path, followlinks=follow_symlinks):
            # Skip symlinks if not following
            if not follow_symlinks:
                dirs[:] = [d for d in dirs if not Path(root, d).is_symlink()]
            
            for file in files:
                file_path = Path(root) / file
                
                # Skip symlinks if not following
                if not follow_symlinks and file_path.is_symlink():
                    continue
                
                ext = file_path.suffix.lower()
                if ext in VIDEO_EXTENSIONS or ext in IMAGE_EXTENSIONS:
                    files_to_process.append(file_path)
    
    if not files_to_process:
        console.print("[bold red]No media files found![/bold red]")
        return
    
    # Check output directory writability
    output_path = Path(output_path)
    if not check_directory_writable(output_path):
        console.print(f"[bold red]Error: Cannot write to output directory: {output_path}[/bold red]")
        return
    
    # Clean up partial files from previous runs
    if output_path.exists():
        clean_partial_files(output_path)
    
    if not quiet:
        console.print(f"[bold green]Found {len(files_to_process)} files to process.[/bold green]")
        console.print(f"[dim]Output: {output_path}[/dim]")
        console.print(f"[bold]Strategy: {cpu_workers} CPU Workers | {gpu_workers} Hardware Workers ({selected_encoder.display_name})[/bold]")
    
    if delete_source:
        console.print("[bold red blink]WARNING: Source files will be DELETED after successful conversion![/bold red blink]")

    # State Manager setup
    state_manager = None
    if resume:
        state_manager = StateManager(output_path)
        if not quiet:
            console.print(f"[bold cyan]Resume:[/bold cyan] Tracking progress in {state_manager.state_file}")

    # Calculate total duration and scan media info
    global_mode = "time"
    total_duration = 0.0
    media_cache = {}
    
    should_scan = True
    if no_scan_duration:
        should_scan = False
    elif len(files_to_process) > 100:
        if not Confirm.ask(f"[yellow]Found {len(files_to_process)} files. Scan metadata? (Required for fast skipping)[/yellow]", default=True):
            should_scan = False
    
    if should_scan:
        video_files = []
        for f in files_to_process:
            # Skip duration check if already processed (optimization)
            if state_manager:
                rel_p = str(f.relative_to(input_path)) if input_path.is_dir() else f.name
                if state_manager.is_processed(rel_p):
                    continue
            
            if f.suffix.lower() in VIDEO_EXTENSIONS:
                video_files.append(f)

        if video_files:
            total_duration, media_cache = scan_media_info_parallel(video_files, console)
            if not quiet:
                console.print(f"[bold]Total Video Duration to Process: {total_duration/3600:.2f} hours[/bold]")
    else:
        global_mode = "count"
        if not quiet:
            console.print("[yellow]Skipping duration scan. Progress will show file count.[/yellow]")

    # Fill queue
    file_queue = queue.Queue()
    for f in files_to_process:
        # Don't filter here so that progress bar total count is accurate/consistent (we skip inside worker)
        # OR we could filter here? Filtering here is cleaner for count, but "resume" usually implies seeing progress go to 100%.
        # Let's filter here if global_mode is count to avoid confusion, but wait, we want to see "Total Progress" potentially.
        # Actually, adding all files and skipping them instantly in worker is fast enough, and keeps "Total" consistent with "Files Found".
        
        # DEBUG: Print what we are putting in queue relative to input path
        # rel_p = str(f.relative_to(input_path)) if input_path.is_dir() else f.name
        # console.print(f"[dim]Queue: {rel_p}[/dim]")
        
        file_queue.put(f)
    
    timeout_val = timeout if timeout > 0 else None
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeRemainingColumn(),
        console=console
    ) as progress:
        
        if global_mode == "time":
            total_task_id = progress.add_task("[bold white]Total Progress", total=total_duration)
        else:
            total_task_id = progress.add_task("[bold white]Files Processed", total=len(files_to_process))

        executor = concurrent.futures.ThreadPoolExecutor(max_workers=total_workers)
        futures = []
        
        image_executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=os.cpu_count() or 4,
            thread_name_prefix="ImageWorker"
        )

        try:
            # Start CPU workers
            cpu_encoder = encoders[EncoderType.CPU]
            for _ in range(cpu_workers):
                futures.append(executor.submit(
                    worker_loop, file_queue, input_path, output_path, crf, preset, 
                    image_quality, keep_format, delete_source, cpu_encoder, 
                    progress, total_task_id, global_mode, timeout_val, state_manager, media_cache, image_executor
                ))
            
            # Start hardware workers
            for _ in range(gpu_workers):
                futures.append(executor.submit(
                    worker_loop, file_queue, input_path, output_path, crf, preset,
                    image_quality, keep_format, delete_source, selected_encoder,
                    progress, total_task_id, global_mode, timeout_val, state_manager, media_cache, image_executor
                ))
            
            # Monitoring loop
            def handle_interrupt():
                global interrupt_count, last_interrupt_time
                current_time = time.time()
                
                if current_time - last_interrupt_time < 1.5:
                    interrupt_count += 1
                else:
                    interrupt_count = 1
                
                last_interrupt_time = current_time
                
                if interrupt_count >= 2:
                    console.print("\n[bold red]Force Exit! (Double Ctrl+C detected)[/bold red]")
                    shutdown_event.set()
                    if state_manager:
                        state_manager.save()
                    os._exit(1)
                
                return True
            
            # Register signal handlers for robust exit
            def signal_handler(signum, frame):
                shutdown_event.set()
                if state_manager:
                    state_manager.save()
                # We can't exit immediately safely locally, just set event
                # But if it's SIGTERM we should probably force exit soon if not handled
                console.print(f"\n[yellow]Received signal {signum}, stopping...[/yellow]")
                
            try:
                signal.signal(signal.SIGTERM, signal_handler)
            except Exception:
                pass
            
            while True:
                try:
                    all_done = all(f.done() for f in futures)
                    if all_done:
                        break
                    time.sleep(0.5)
                except KeyboardInterrupt:
                    if not handle_interrupt():
                        continue
                    
                    pause_event.set()
                    progress.stop()
                    
                    console.print("\n[bold yellow]Interrupted! Press Ctrl+C again quickly to force exit.[/bold yellow]")
                    
                    try:
                        response = Confirm.ask("Stop processing?", default=False)
                        if response:
                            progress.console.print("[bold red]Stopping... (cleaning up)[/bold red]")
                            shutdown_event.set()
                            pause_event.clear()
                            executor.shutdown(wait=False, cancel_futures=True)
                            image_executor.shutdown(wait=False, cancel_futures=True)
                            # Final save
                            if state_manager:
                                state_manager.save()
                            time.sleep(0.5)
                            print_summary_report()
                            os._exit(0)
                        else:
                            interrupt_count = 0
                            pause_event.clear()
                            progress.start()
                            progress.console.print("[bold green]Resuming...[/bold green]")
                            continue
                    except (KeyboardInterrupt, EOFError):
                        console.print("\n[bold red]Force Exit![/bold red]")
                        shutdown_event.set()
                        os._exit(1)
        
        finally:
            if not shutdown_event.is_set():
                executor.shutdown(wait=True)
                # Final save
                if state_manager:
                    state_manager.save()
    
    # Print summary
    if not quiet:
        print_summary_report()


if __name__ == "__main__":
    main()