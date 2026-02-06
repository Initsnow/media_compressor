import os
import subprocess
import sys
import re
import shutil
import signal
import tempfile
from pathlib import Path
import click
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn, TimeRemainingColumn
from rich.prompt import Confirm
import concurrent.futures
import queue
import threading
import time
from PIL import Image

console = Console()
shutdown_event = threading.Event()
pause_event = threading.Event()
interrupt_count = 0
last_interrupt_time = 0

def get_gpu_free_memory():
    """Get free VRAM in MiB using nvidia-smi. Returns 0 if failed."""
    try:
        cmd = ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            # Return memory of the first GPU (assuming single GPU usage for now)
            return int(result.stdout.strip().split('\n')[0])
    except:
        pass
    return 0

def get_duration(input_file):
    """Get video duration in seconds using ffprobe."""
    cmd = [
        "ffprobe", "-v", "error", "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1", str(input_file)
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return float(result.stdout.strip())
    except Exception:
        return None

def get_video_codec(input_file):
    """Get video codec name using ffprobe."""
    cmd = [
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=codec_name",
        "-of", "default=noprint_wrappers=1:nokey=1", str(input_file)
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return result.stdout.strip().lower()
    except Exception:
        return None

def convert_time_to_seconds(time_str):
    """Convert HH:MM:SS.ms to seconds."""
    try:
        h, m, s = time_str.split(':')
        return int(h) * 3600 + int(m) * 60 + float(s)
    except:
        return 0.0

def format_bytes(size):
    """Return human readable file size string."""
    power = 2**10
    n = 0
    power_labels = {0 : '', 1: 'K', 2: 'M', 3: 'G', 4: 'T'}
    while size > power:
        size /= power
        n += 1
    return f"{size:.2f} {power_labels[n]}B"

def calculate_saving(old_size, new_size):
    """Return saving string and percentage."""
    if old_size == 0: return "0 B (0%)"
    saved = old_size - new_size
    percent = (saved / old_size) * 100
    return f"{format_bytes(saved)} ({percent:.1f}%)"

def estimate_compression_ratio(input_file, crf, preset, gpu, sample_seconds=10):
    """
    Encode short samples from multiple positions to estimate compression ratio.
    Samples from 25%, 50%, 75% of the video for better accuracy.
    Returns estimated ratio (new_size / old_size). Values < 1.0 mean compression saves space.
    Returns None if estimation fails.
    """

    
    duration = get_duration(input_file)
    if not duration or duration < sample_seconds * 2:
        return None  # Too short to sample, skip estimation
    
    # Get original file's bitrate for the sample duration
    original_size = input_file.stat().st_size
    original_bitrate = original_size / duration  # bytes per second
    
    # Sample positions: 25%, 50%, 75% of the video
    sample_positions = [
        duration * 0.25,
        duration * 0.50,
        duration * 0.75
    ]
    
    ratios = []
    
    for start_pos in sample_positions:
        # Make sure we don't go past the end
        if start_pos + sample_seconds > duration:
            start_pos = max(0, duration - sample_seconds)
        
        expected_sample_size = original_bitrate * sample_seconds
        
        # Create temp file for sample encode
        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as tmp:
            temp_path = tmp.name
        
        try:
            cmd = [
                "ffmpeg", "-y",
                "-hide_banner",
                "-loglevel", "error",
                "-ss", str(start_pos),
                "-t", str(sample_seconds),
                "-i", str(input_file)
            ]
            
            if gpu:
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
            else:
                cmd.extend([
                    "-c:v", "libx265",
                    "-crf", str(crf),
                    "-preset", preset
                ])
            
            cmd.extend(["-an", "-f", "mp4", temp_path])  # No audio for faster test
            
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
            except:
                pass
    
    if ratios:
        # Return average ratio from all samples
        return sum(ratios) / len(ratios)
    
    return None

def calculate_total_duration(files, console):
    """Sum up duration of all files concurrently."""
    total_seconds = 0.0
    
    # Use threads for IO-bound ffprobe calls
    with console.status("[bold green]Scanning files for duration (Parallel)...[/bold green]") as status:
        with concurrent.futures.ThreadPoolExecutor(max_workers=os.cpu_count() or 4) as executor:
            # Map files to futures
            future_to_file = {executor.submit(get_duration, f): f for f in files}
            
            completed_count = 0
            total_files = len(files)
            
            for future in concurrent.futures.as_completed(future_to_file):
                if shutdown_event.is_set(): 
                    executor.shutdown(wait=False, cancel_futures=True)
                    break
                
                completed_count += 1
                status.update(f"[bold green]Scanning {completed_count}/{total_files}...[/bold green]")
                
                try:
                    d = future.result()
                    if d:
                        total_seconds += d
                except Exception:
                    pass
                    
    return total_seconds

def process_video(in_file, input_base_path, output_base_path, crf, preset, delete_source, gpu, progress, total_task_id, global_mode="time"):
    """
    Process a single video file.
    """
    if shutdown_event.is_set(): return

    worker_type = "GPU" if gpu else "CPU"
    
    # Calculate target output file path
    if input_base_path.is_dir():
        rel_path = in_file.relative_to(input_base_path)
        out_file = output_base_path / rel_path.with_suffix('.mp4')
    else:
        out_file = output_base_path
        
    out_file.parent.mkdir(parents=True, exist_ok=True)

    if out_file.exists():
        # console.print(f"[yellow]Skipping {in_file.name} (Output exists)[/yellow]")
        return

    # Check if video is already using an efficient codec (AV1 or H.265/HEVC)
    codec = get_video_codec(in_file)
    if codec in ['hevc', 'h265', 'av1']:
        progress.console.print(f"[bold blue]⊘ Skipping {in_file.name} (Already {codec.upper()} encoded)[/bold blue]")
        # Move or copy original to output directory so it won't be re-processed
        if delete_source:
            shutil.move(in_file, out_file)
        else:
            shutil.copy2(in_file, out_file)
        # Update global progress if applicable
        if global_mode == "count":
            progress.advance(total_task_id, advance=1)
        elif global_mode == "time":
            duration = get_duration(in_file)
            if duration:
                progress.advance(total_task_id, advance=duration)
        return

    # Get duration for later use
    duration = get_duration(in_file)
    
    # Pre-check: estimate compression ratio with a short sample
    # Only for videos longer than 60 seconds (short videos: overhead not worth it)
    if duration and duration > 60:
        estimated_ratio = estimate_compression_ratio(in_file, crf, preset, gpu)
        if estimated_ratio is not None and estimated_ratio >= 0.95:
            # Compression would save less than 5% or make file larger - skip it
            progress.console.print(f"[bold yellow]⊘ Skipping {in_file.name} (Estimated ratio: {estimated_ratio:.2f}x - not worth compressing)[/bold yellow]")
            # Move or copy original to output directory so it won't be re-processed
            if delete_source:
                shutil.move(in_file, out_file)
            else:
                shutil.copy2(in_file, out_file)
            if global_mode == "count":
                progress.advance(total_task_id, advance=1)
            elif global_mode == "time":
                progress.advance(total_task_id, advance=duration)
            return

    temp_out_file = out_file.with_suffix('.mp4.part')
    duration = get_duration(in_file)
    
    cmd = [
        "ffmpeg", "-y",
        "-hide_banner",
        "-loglevel", "error",
        "-stats",
        "-i", str(in_file)
    ]

    if gpu:
        # NVENC settings
        nv_preset = "p4" # medium
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
    else:
        # x265 settings
        cmd.extend([
            "-c:v", "libx265",
            "-crf", str(crf),
            "-preset", preset
        ])

    cmd.extend([
        "-c:a", "copy",
        "-f", "mp4",
        str(temp_out_file)
    ])

    task_id = progress.add_task(f"[{'magenta' if gpu else 'cyan'}]{worker_type}: {in_file.name}", total=duration if duration else None)
    
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
        
        while True:
            # Check for global shutdown signal
            if shutdown_event.is_set():
                process.terminate()
                break
                
            # Check for pause signal
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
            # Get file sizes before rename to compare
            old_size = in_file.stat().st_size
            new_size = temp_out_file.stat().st_size
            
            # Safety check: if compressed file is larger, discard it and keep original
            if new_size >= old_size:
                temp_out_file.unlink()  # Delete the larger compressed file
                # Move or copy original to output directory so it won't be re-processed
                if delete_source:
                    shutil.move(in_file, out_file)
                else:
                    shutil.copy2(in_file, out_file)
                progress.console.print(f"[bold yellow]⚠ Kept original {in_file.name} (Compressed size larger: {format_bytes(old_size)} → {format_bytes(new_size)})[/bold yellow]")
                
                # Update progress
                if global_mode == "time":
                    final_delta = (duration if duration else 0) - last_reported_seconds
                    if final_delta > 0:
                        progress.advance(total_task_id, advance=final_delta)
                else:
                    progress.advance(total_task_id, advance=1)
                    
                progress.update(task_id, completed=duration if duration else 100)
                progress.stop_task(task_id)
                return
            
            # Compressed file is smaller, proceed with rename
            temp_out_file.rename(out_file)
            
            if global_mode == "time":
                final_delta = (duration if duration else 0) - last_reported_seconds
                if final_delta > 0:
                    progress.advance(total_task_id, advance=final_delta)
            else:
                progress.advance(total_task_id, advance=1)
                
            progress.update(task_id, completed=duration if duration else 100)
            progress.stop_task(task_id)
            
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
            
            if temp_out_file.exists():
                temp_out_file.unlink()
            progress.stop_task(task_id)

    except Exception as e:
        progress.console.print(f"[bold red]Exception {in_file.name}: {e}[/bold red]")
        if temp_out_file.exists():
            temp_out_file.unlink()
        progress.stop_task(task_id)

def process_image(in_file, input_base_path, output_base_path, crf, image_quality, keep_format, delete_source, progress, total_task_id, global_mode="time"):
    """
    Process a single image file.
    """
    if shutdown_event.is_set(): return

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

    if out_file.exists():
        return

    # Use a stronger task ID color for images
    task_id = progress.add_task(f"[green]IMG: {in_file.name}", total=None) 
    
    try:
        # Load image
        with Image.open(in_file) as img:
            # Convert to RGB if saving as JPEG (though we default to WebP which handles RGBA)
            # For WebP, RGBA is fine.
            
            save_kwargs = {}
            output_format = out_file.suffix.lower().lstrip('.')
            source_format = in_file.suffix.lower().lstrip('.')
            
            if output_format == 'webp':
                # Use lossless for PNG/BMP/TIFF sources (they were lossless), lossy for JPEG sources
                if source_format in ['png', 'bmp', 'tiff']:
                    save_kwargs = {'lossless': True, 'method': 6}
                else:
                    save_kwargs = {'quality': image_quality, 'method': 6}
            elif output_format in ['jpg', 'jpeg']:
                 if img.mode == 'RGBA':
                     img = img.convert('RGB')
                 save_kwargs = {'quality': image_quality, 'optimize': True}
            elif output_format == 'png':
                 save_kwargs = {'optimize': True}

            # Save
            img.save(out_file, **save_kwargs)
            
        # Finish
        progress.stop_task(task_id)

        old_size = in_file.stat().st_size
        new_size = out_file.stat().st_size
        
        # Safety check: if compressed file is larger, discard it and keep original
        if new_size >= old_size:
            out_file.unlink()  # Delete the larger compressed file
            # Move or copy original to output directory so it won't be re-processed
            if delete_source:
                shutil.move(in_file, out_file)
            else:
                shutil.copy2(in_file, out_file)
            progress.console.print(f"[bold yellow]⚠ Kept original {in_file.name} (Compressed size larger: {format_bytes(old_size)} → {format_bytes(new_size)})[/bold yellow]")
            
            if global_mode != "time":
                progress.advance(total_task_id, advance=1)
            return
        
        if global_mode != "time":
            progress.advance(total_task_id, advance=1)
        
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
        if out_file.exists():
            out_file.unlink()
        progress.stop_task(task_id)

def worker_loop(file_queue, input_path, output_path, crf, preset, image_quality, keep_format, delete_source, gpu, progress, total_task_id, global_mode):
    """
    Continuous worker loop that pulls from queue.
    """
    while not shutdown_event.is_set():
        # Check pause before picking new task
        while pause_event.is_set():
            time.sleep(0.1)
            
        try:
            in_file = file_queue.get(timeout=1) # timeout allows checking shutdown_event
        except queue.Empty:
            if file_queue.empty(): # Double check if truly empty or just timed out
                return
            continue

        if in_file.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp']:
            process_image(
                in_file, input_path, output_path, crf, image_quality, keep_format, delete_source, progress, total_task_id, global_mode
            )
        else:
            process_video(
                in_file, input_path, output_path, crf, preset, delete_source, gpu, progress, total_task_id, global_mode
            )
        file_queue.task_done()

@click.command()
@click.argument("input_path", type=click.Path(exists=True))
@click.option("--output-path", "-o", type=click.Path(), help="Output directory (for folders) or file path.")
@click.option("--crf", default=23, help="CRF/CQ value (default: 23). Lower is better quality.")
@click.option("--preset", default="medium", help="Preset (default: medium). Slower = better compression.")
@click.option("--delete-source", is_flag=True, help="Delete source file after successful compression.")
@click.option("--cpu-workers", default=0, help="Number of CPU workers.")
@click.option("--gpu-workers", default=0, help="Number of GPU workers.")
@click.option("--auto-config", is_flag=True, help="Automatically determine best worker count based on hardware.")
@click.option("--workers", "-w", default=0, help="Legacy: Total workers (defaults to CPU if GPU not specified).")
@click.option("--gpu", is_flag=True, help="Legacy: Use GPU for all workers if specified.")
@click.option("--image-quality", default=90, help="Image quality (1-100). Default 90 (High).")
@click.option("--keep-format", is_flag=True, help="Keep original image format instead of converting to WebP.")
@click.option("--no-scan-duration", is_flag=True, help="Skip pre-calculating total video duration (Faster startup).")
def main(input_path, output_path, crf, preset, delete_source, cpu_workers, gpu_workers, auto_config, workers, gpu, image_quality, keep_format, no_scan_duration):
    """
    Batch compress videos using ffmpeg. Supports Hybrid CPU + GPU processing.
    """
    input_path = Path(input_path)
    
    # --- Configuration Logic ---
    
    # 1. Handle Auto Config
    if auto_config:
        cpu_count = os.cpu_count() or 4
        # Heuristic: x265 scales well up to 8-12 threads per instance. 
        # Divide total threads by 8 to get safe CPU worker count.
        calc_cpu = max(1, cpu_count // 8)
        
        # Smart GPU Config based on VRAM
        free_vram = get_gpu_free_memory()
        if free_vram > 4500:
            calc_gpu = 3 # High-end cards
        elif free_vram > 1800:
            calc_gpu = 2 # Standard 4GB+ cards with some usage
        else:
            calc_gpu = 1 # Low VRAM or P620 (2GB total, ~1.4GB free)
        
        if cpu_workers == 0: cpu_workers = calc_cpu
        if gpu_workers == 0: gpu_workers = calc_gpu
        
        vram_msg = f"(Free VRAM: {free_vram} MiB)" if free_vram else "(VRAM detection failed)"
        console.print(f"[bold cyan]Auto-Config:[/bold cyan] Setting CPU Workers={cpu_workers}, GPU Workers={gpu_workers} {vram_msg}")

    # 2. Handle Legacy Flags (if no explicit cpu/gpu workers set)
    if cpu_workers == 0 and gpu_workers == 0:
        if workers == 0: workers = 1 # Default default
        
        if gpu:
            gpu_workers = workers
        else:
            cpu_workers = workers
            
    total_workers = cpu_workers + gpu_workers
    
    # --- File Discovery ---
    files_to_process = []
    if input_path.is_file():
        files_to_process.append(input_path)
        if not output_path:
            output_path = input_path.parent / f"{input_path.stem}_x265.mp4"
        else:
            output_path = Path(output_path)
            if output_path.is_dir() or (str(output_path).endswith(os.sep)):
                 output_path.mkdir(parents=True, exist_ok=True)
                 output_path = output_path / f"{input_path.stem}.mp4"
    else:
        if not output_path:
            output_path = input_path.parent / f"{input_path.name}_compressed"
        else:
            output_path = Path(output_path)
            
    if not files_to_process: # Only check discovery if manually empty? No, logic here handles both.
        # Check input_path explicitly if it is a directory
        if input_path.is_dir():
             for root, _, files in os.walk(input_path):
                for file in files:
                    ext = file.lower()
                    if ext.endswith(('.mp4', '.mkv', '.mov', '.avi', '.flv', '.wmv', '.ts', '.webm', '.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp')):
                        files_to_process.append(Path(root) / file)
                        
    # Handle single file case already covered above in lines 305-313, 
    # but we need to ensure single file allows images too if passed directly
    if input_path.is_file() and not files_to_process: # if 306 happened, files_to_process has 1 item.
         pass # already added


    if not files_to_process:
        console.print("[bold red]No video files found![/bold red]")
        return

    console.print(f"[bold green]Found {len(files_to_process)} files to process.[/bold green]")
    console.print(f"[dim]Output: {output_path}[/dim]")
    console.print(f"[bold]Strategy: {cpu_workers} CPU Workers | {gpu_workers} GPU Workers[/bold]")
    if delete_source:
        console.print("[bold red blink]WARNING: Source files will be DELETED after successful conversion![/bold red blink]")

    # --- Pre-calculate Total Duration ---
    global_mode = "time"
    total_duration = 0.0
    
    should_scan = True
    if no_scan_duration:
        should_scan = False
    elif len(files_to_process) > 100:
        if not Confirm.ask(f"[yellow]Found {len(files_to_process)} files. Scan total duration? (May take time)[/yellow]", default=False):
            should_scan = False
            
    if should_scan:
        total_duration = calculate_total_duration(files_to_process, console)
        console.print(f"[bold]Total Video Duration to Process: {total_duration/3600:.2f} hours[/bold]")
    else:
        global_mode = "count"
        console.print("[yellow]Skipping duration scan. Progress will show file count.[/yellow]")

    # --- Processing ---
    
    # Fill Queue
    file_queue = queue.Queue()
    for f in files_to_process:
        file_queue.put(f)
        
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeRemainingColumn(),
        console=console
    ) as progress:
        
        
        # Add Global Progress Bar
        if global_mode == "time":
            total_task_id = progress.add_task("[bold white]Total Progress", total=total_duration)
        else:
            total_task_id = progress.add_task("[bold white]Files Processed", total=len(files_to_process))

        # Manually manage executor to avoid context manager's blocking exit
        executor = concurrent.futures.ThreadPoolExecutor(max_workers=total_workers)
        futures = []
        
        try:
            # Start CPU Workers
            for _ in range(cpu_workers):
                futures.append(executor.submit(
                    worker_loop, file_queue, input_path, output_path, crf, preset, image_quality, keep_format, delete_source, False, progress, total_task_id, global_mode
                ))
                
            # Start GPU Workers
            for _ in range(gpu_workers):
                futures.append(executor.submit(
                    worker_loop, file_queue, input_path, output_path, crf, preset, image_quality, keep_format, delete_source, True, progress, total_task_id, global_mode
                ))
            
            # Monitoring Loop with improved interrupt handling
            def handle_interrupt():
                """Handle interrupt with improved logic."""
                global interrupt_count, last_interrupt_time
                current_time = time.time()
                
                # If second interrupt within 1.5 seconds, force exit immediately
                if current_time - last_interrupt_time < 1.5:
                    interrupt_count += 1
                else:
                    interrupt_count = 1
                
                last_interrupt_time = current_time
                
                if interrupt_count >= 2:
                    console.print("\n[bold red]Force Exit! (Double Ctrl+C detected)[/bold red]")
                    shutdown_event.set()
                    os._exit(1)
                
                return True  # First interrupt
            
            while True:
                try:
                    all_done = True
                    for f in futures:
                        if not f.done():
                            all_done = False
                            break
                    
                    if all_done:
                        break
                        
                    time.sleep(0.5)
                except KeyboardInterrupt:
                    # Check for rapid double-press
                    if not handle_interrupt():
                        continue
                    
                    # Set pause signal to silence workers
                    pause_event.set()
                    progress.stop()
                    
                    console.print("\n[bold yellow]Interrupted! Press Ctrl+C again quickly to force exit.[/bold yellow]")
                    
                    try:
                        # Use Confirm in a clean environment (progress stopped, workers paused)
                        response = Confirm.ask("Stop processing?", default=False)
                        if response:
                            progress.console.print("[bold red]Stopping... (cleaning up)[/bold red]")
                            shutdown_event.set()
                            pause_event.clear() # Unpause workers so they can see shutdown_event and exit
                            executor.shutdown(wait=False, cancel_futures=True)
                            time.sleep(0.5)
                            os._exit(0)
                        else:
                            # Reset interrupt counter on resume
                            interrupt_count = 0
                            pause_event.clear() # Resume workers
                            progress.start()
                            progress.console.print("[bold green]Resuming...[/bold green]")
                            continue
                    except (KeyboardInterrupt, EOFError):
                        # Any interrupt during prompt = force exit
                        console.print("\n[bold red]Force Exit![/bold red]")
                        shutdown_event.set()
                        os._exit(1)
        
        finally:
            # Clean shutdown if we finished normally
            if not shutdown_event.is_set():
                executor.shutdown(wait=True)


if __name__ == "__main__":
    main()