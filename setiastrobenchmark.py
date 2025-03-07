import os
import time
import json
import torch
import requests
import numpy as np
import onnxruntime as ort
import tkinter as tk
from tkinter import ttk
from astropy.io import fits
from numba import njit, jit, prange
import psutil
import platform
import base64
import cpuinfo
import multiprocessing

# Check for CUDA, DirectML, or MPS (Metal Performance Shaders)
USE_CUDA = torch.cuda.is_available()
USE_MPS = torch.backends.mps.is_available() if platform.system() == "Darwin" else False

# Load Benchmark Image
BENCHMARK_IMAGE_PATH = "benchmarkimage.fit"
with fits.open(BENCHMARK_IMAGE_PATH) as hdul:
    image = hdul[0].data.astype(np.float32)
    if image.ndim == 2:  # Convert grayscale to 3-channel for CNN
        image = np.stack([image] * 3, axis=0)

H, W = image.shape[1:]  # Image height and width
PATCH_SIZE = 256  # CNN processes 256x256 patches

# ✅ Function to Tile Image into 256x256 Chunks
def tile_image(image_array, patch_size=256):
    """Splits an image into 256x256 patches for batch processing."""
    c, h, w = image_array.shape  # (Channels, Height, Width)
    patches = []
    for i in range(0, h, patch_size):
        for j in range(0, w, patch_size):
            patch = image_array[:, i:i+patch_size, j:j+patch_size]  # Extract patch
            if patch.shape[1] == patch_size and patch.shape[2] == patch_size:
                patches.append(patch)
    return np.array(patches)  # Shape: (num_patches, 3, 256, 256)

image_patches = tile_image(image)  # Tile into patches
image_tensor = torch.tensor(image_patches)  # Shape (N, 3, 256, 256)

# GitHub Repo Details
GITHUB_USERNAME = "setiastro"
GITHUB_REPO = "setiastrosuite"
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")  # ⚠️ Store securely as an environment variable!
JSON_FILE_PATH = "benchmark_results.json"

# ✅ Fetch System Information
def get_system_info():
    """Retrieve OS, CPU, GPU, RAM, and available acceleration backends."""
    system_info = {
        "OS": platform.system() + " " + platform.release(),
        "CPU": cpuinfo.get_cpu_info()["brand_raw"],
        "RAM": f"{round(psutil.virtual_memory().total / (1024 ** 3), 1)} GB",
        "CUDA Available": torch.cuda.is_available(),
        "MPS Available": torch.backends.mps.is_available() if platform.system() == "Darwin" else False,
        "ONNX Providers": ort.get_available_providers()
    }

    # Get GPU details if CUDA is available
    if torch.cuda.is_available():
        system_info["GPU"] = torch.cuda.get_device_name(0)

    return system_info

def open_submission_page():
    """Open the Seti Astro benchmark submission page."""
    import webbrowser
    webbrowser.open("https://setiastro.com/benchmark-submit")

def save_results_locally(results):
    """Save benchmark results locally."""
    filename = "benchmark_results.json"

    # Load existing data if available
    if os.path.exists(filename):
        with open(filename, "r") as file:
            try:
                all_results = json.load(file)
            except json.JSONDecodeError:
                all_results = []
    else:
        all_results = []

    # Append new results
    all_results.append(results)

    # Save updated JSON
    with open(filename, "w") as file:
        json.dump(all_results, file, indent=4)

    print("✅ Results saved to", filename)


# Define ResidualBlock for CNN
class ResidualBlock(torch.nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = torch.nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.relu = torch.nn.ReLU()
        self.conv2 = torch.nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)
        out += residual
        out = self.relu(out)
        return out

# Define SharpeningCNN for Stellar Sharpening
class SharpeningCNN(torch.nn.Module):
    def __init__(self):
        super(SharpeningCNN, self).__init__()

        self.encoder1 = torch.nn.Sequential(
            torch.nn.Conv2d(3, 16, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            ResidualBlock(16)
        )
        self.encoder2 = torch.nn.Sequential(
            torch.nn.Conv2d(16, 32, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            ResidualBlock(32)
        )
        self.encoder3 = torch.nn.Sequential(
            torch.nn.Conv2d(32, 64, kernel_size=3, padding=2, dilation=2),
            torch.nn.ReLU(),
            ResidualBlock(64)
        )
        self.encoder4 = torch.nn.Sequential(
            torch.nn.Conv2d(64, 128, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            ResidualBlock(128)
        )
        self.encoder5 = torch.nn.Sequential(
            torch.nn.Conv2d(128, 256, kernel_size=3, padding=2, dilation=2),
            torch.nn.ReLU(),
            ResidualBlock(256)
        )

        self.decoder5 = torch.nn.Sequential(
            torch.nn.Conv2d(256 + 128, 128, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            ResidualBlock(128)
        )
        self.decoder4 = torch.nn.Sequential(
            torch.nn.Conv2d(128 + 64, 64, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            ResidualBlock(64)
        )
        self.decoder3 = torch.nn.Sequential(
            torch.nn.Conv2d(64 + 32, 32, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            ResidualBlock(32)
        )
        self.decoder2 = torch.nn.Sequential(
            torch.nn.Conv2d(32 + 16, 16, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            ResidualBlock(16)
        )
        self.decoder1 = torch.nn.Sequential(
            torch.nn.Conv2d(16, 3, kernel_size=3, padding=1),
            torch.nn.Sigmoid()
        )

    def forward(self, x):
        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)
        e5 = self.encoder5(e4)

        d5 = self.decoder5(torch.cat([e5, e4], dim=1))
        d4 = self.decoder4(torch.cat([d5, e3], dim=1))
        d3 = self.decoder3(torch.cat([d4, e2], dim=1))
        d2 = self.decoder2(torch.cat([d3, e1], dim=1))
        d1 = self.decoder1(d2)

        return d1

# ✅ Load Model
def load_model(exe_dir, use_gpu=True):
    """Load Stellar SharpeningCNN Model with CUDA, MPS, or CPU"""
    if USE_CUDA and use_gpu:
        device = torch.device("cuda")
        print("Using CUDA for GPU acceleration.")
    elif USE_MPS and use_gpu:
        device = torch.device("mps")
        print("Using MPS (Metal Performance Shaders) for GPU acceleration.")
    else:
        device = torch.device("cpu")
        print("Using CPU.")

    model = SharpeningCNN()
    model_path = os.path.join(exe_dir, 'deep_sharp_stellar_cnn_AI3_5.pth')
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval().to(device)

    return model, device



# ✅ GPU Benchmark (CUDA) with 256x256 Chunks & AMP
def gpu_benchmark(model, device, image_patches, progress_callback):
    """Run GPU Benchmark using CUDA or MPS (Metal Performance Shaders) with AMP"""
    image_patches = image_patches.to(device)

    total_time = 0
    with torch.no_grad():
        with torch.cuda.amp.autocast():  # Enable Mixed Precision
            for i in range(len(image_patches)):
                patch = image_patches[i:i+1]
                start_time = time.time()
                _ = model(patch)

                if USE_CUDA:
                    torch.cuda.synchronize()
                elif USE_MPS:
                    torch.mps.synchronize()

                total_time += (time.time() - start_time) * 1000
                progress_callback(f"GPU Benchmark: {i+1}/{len(image_patches)} patches...", clear=True)

    avg_time = total_time / len(image_patches)
    return avg_time, total_time




# ✅ ONNX Benchmark (DirectML, CUDA, or CPU)
def onnx_benchmark(image_patches, progress_callback):
    """Run ONNX Benchmark using DirectML, CUDA, or CPU with proper model loading."""
    model_path = "deep_sharp_stellar_cnn_AI3_5.onnx"
    if not os.path.exists(model_path):
        return "ONNX Model Not Found"

    available_providers = ort.get_available_providers()

    # Prioritize DirectML, then CUDA, then CPU
    if "DmlExecutionProvider" in available_providers:
        provider = "DmlExecutionProvider"
    elif "CUDAExecutionProvider" in available_providers:
        provider = "CUDAExecutionProvider"
    else:
        provider = "CPUExecutionProvider"

    print(f"Using ONNX Provider: {provider}")

    # ✅ Load ONNX Model
    ort_session = ort.InferenceSession(model_path, providers=[provider])
    input_name = ort_session.get_inputs()[0].name

    total_time = 0
    for i in range(len(image_patches)):
        patch = image_patches[i:i+1].numpy().astype(np.float32)  # Convert patch to numpy
        start_time = time.time()
        ort_session.run(None, {input_name: patch})  # Run inference
        total_time += (time.time() - start_time) * 1000  # Convert to ms

        # ✅ Show Progress
        progress_callback(f"ONNX Benchmark: {i+1}/{len(image_patches)} patches...", clear=True)

    avg_time = total_time / len(image_patches)
    return avg_time, total_time


@njit
def mad_cpu_jit(image_array, median_val):
    """Compute Median Absolute Deviation (MAD)."""
    return np.median(np.abs(image_array - median_val))

def mad_cpu(image_array, runs=3):
    """Run MAD multiple times to separate JIT compilation overhead."""
    times = []
    for _ in range(runs):
        start_time = time.time()
        median_val = np.median(image_array)
        _ = mad_cpu_jit(image_array, median_val)
        times.append((time.time() - start_time) * 1000)  # Convert to ms
    return times  # Return all timings (first includes JIT compile time)

@njit(parallel=True)
def flat_field_correction_jit(image_array, flat_frame, median_flat):
    """Flat field correction using multi-threading."""
    return image_array / (flat_frame / median_flat)

def flat_field_correction(image_array, flat_frame, runs=3):
    """Run Flat-Field Correction multiple times to measure JIT overhead."""
    times = []
    for _ in range(runs):
        start_time = time.time()
        median_flat = np.median(flat_frame)
        _ = flat_field_correction_jit(image_array, flat_frame, median_flat)
        times.append((time.time() - start_time) * 1000)  # Convert to ms
    return times  # Return all timings

# ✅ Benchmark GUI
class BenchmarkGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Seti Astro Suite Benchmark")
        self.root.geometry("500x450")

        self.label = ttk.Label(root, text="Seti Astro Suite Benchmark", font=("Arial", 14))
        self.label.pack(pady=10)

        self.option = tk.StringVar(value="Both")
        self.dropdown = ttk.Combobox(root, textvariable=self.option, values=["CPU", "GPU", "Both"])
        self.dropdown.pack(pady=5)

        self.start_button = ttk.Button(root, text="Run Benchmark", command=self.run_benchmark)
        self.start_button.pack(pady=5)

        self.save_button = ttk.Button(root, text="Save Locally", command=self.save_results_locally, state=tk.DISABLED)
        self.save_button.pack(pady=5)

        self.submit_button = ttk.Button(root, text="Submit Benchmark", command=open_submission_page)
        self.submit_button.pack(pady=5)

        self.progress = ttk.Progressbar(root, orient=tk.HORIZONTAL, length=400, mode='determinate')
        self.progress.pack(pady=10)

        self.result_text = tk.Text(root, height=12, width=65)
        self.result_text.pack(pady=10)

        self.results = {}

    def run_benchmark(self):
        self.progress['value'] = 0
        self.result_text.delete('1.0', tk.END)
        results = {}

        def progress_callback(status, clear=False):
            """Update UI with live progress without flooding new lines."""
            self.result_text.delete('end-2l', 'end') if clear else None  # Remove previous status line
            self.result_text.insert(tk.END, status + "\n")
            self.result_text.see(tk.END)
            self.root.update()

        # ✅ Run CPU Benchmarks
        if self.option.get() in ["CPU", "Both"]:
            progress_callback("Running CPU Benchmarks...")
            start_time = time.time()

            # Actually run MAD and Flat-Field Correction benchmarks
            cpu_times_mad = mad_cpu(image)
            cpu_times_flat = flat_field_correction(image, image)

            progress_callback("Completed CPU Benchmark.")

            results["CPU MAD (Single Core)"] = f"First: {cpu_times_mad[0]:.2f} ms | Avg: {np.mean(cpu_times_mad[1:]):.2f} ms"
            results["CPU Flat-Field (Multi-Core)"] = f"First: {cpu_times_flat[0]:.2f} ms | Avg: {np.mean(cpu_times_flat[1:]):.2f} ms"

        # ✅ Run GPU Benchmarks
        if self.option.get() in ["GPU", "Both"]:
            progress_callback("Running GPU Benchmarks...")

            # Load model properly before running the benchmark
            model, device = load_model(os.getcwd(), use_gpu=True)

            avg_gpu_time, total_gpu_time = gpu_benchmark(model, device, image_tensor, progress_callback)
            progress_callback("Completed GPU Benchmark.")

            avg_onnx_time, total_onnx_time = onnx_benchmark(image_tensor, progress_callback)
            progress_callback("Completed ONNX Benchmark.", clear=True)

            results["GPU Time (CUDA)"] = f"Avg: {avg_gpu_time:.2f} ms | Total: {total_gpu_time:.2f} ms"
            results["ONNX Time"] = f"Avg: {avg_onnx_time:.2f} ms | Total: {total_onnx_time:.2f} ms"

        # ✅ Collect System Info
        results["System Info"] = get_system_info()

        # ✅ Display Final Results
        final_results = "\n".join([f"{k}: {v}" for k, v in results.items()])
        self.result_text.insert(tk.END, final_results)
        self.result_text.see(tk.END)

        # ✅ Store Results for Save/Upload
        self.results = results
        self.save_button["state"] = tk.NORMAL

        self.show_results_popup()

    def show_results_popup(self):
        """Automatically display a pop-up with the benchmark JSON, plus Copy & Submit buttons."""
        # Create a pop-up window
        popup = tk.Toplevel(self.root)
        popup.title("Benchmark Results")

        # Convert results to a JSON string
        json_string = json.dumps([self.results], indent=4)

        # A label (optional)
        label = ttk.Label(popup, text="Your Benchmark JSON Data", font=("Arial", 12))
        label.pack(pady=5)

        # A Text widget to display the JSON
        text_box = tk.Text(popup, wrap="word", width=80, height=20)
        text_box.pack(padx=10, pady=10)
        text_box.insert("1.0", json_string)
        text_box.configure(state="disabled")  # Make it read-only

        # A frame to hold the buttons side by side (optional)
        button_frame = ttk.Frame(popup)
        button_frame.pack(pady=10)

        # Copy to Clipboard button
        copy_button = ttk.Button(button_frame, text="Copy to Clipboard",
                                command=lambda: self.copy_to_clipboard(json_string))
        copy_button.pack(side=tk.LEFT, padx=5)

        # Submit Benchmark button
        submit_button = ttk.Button(button_frame, text="Submit Benchmark",
                                command=open_submission_page)
        submit_button.pack(side=tk.LEFT, padx=5)

    def copy_to_clipboard(self, text):
        self.root.clipboard_clear()
        self.root.clipboard_append(text)
        self.root.update()  # Keep clipboard data available
        from tkinter import messagebox
        messagebox.showinfo("Copied", "Benchmark JSON copied to clipboard!")

    def save_results_locally(self):
        save_results_locally(self.results)
        self.result_text.insert(tk.END, "\n✅ Results saved!\n")


if __name__ == "__main__":
    # Needed on Windows when using multiprocessing in a frozen app
    multiprocessing.freeze_support()

    root = tk.Tk()
    app = BenchmarkGUI(root)
    root.mainloop()