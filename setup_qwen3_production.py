"""
Qwen3-Next-80B-A3B Production Setup Script

Automated setup for deploying Qwen3-Next-80B-A3B-Thinking in production.
This script handles all dependencies, model download, and optimization.

Hardware Requirements:
- GPU: RTX 4090 (24GB VRAM) or better
- RAM: 32GB+ system memory
- Storage: 40GB+ free space (4-bit quantized)
- CUDA: 11.8+ or 12.1+

Usage:
    python setup_qwen3_production.py --check          # Check system requirements
    python setup_qwen3_production.py --install        # Install dependencies
    python setup_qwen3_production.py --download       # Download model
    python setup_qwen3_production.py --test           # Test installation
    python setup_qwen3_production.py --all            # Do everything
"""

import subprocess
import sys
import os
import platform
from pathlib import Path


class Colors:
    """Terminal colors for output."""
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    END = '\033[0m'
    BOLD = '\033[1m'


def print_header(text):
    """Print section header."""
    print(f"\n{Colors.BOLD}{Colors.HEADER}{'=' * 80}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.HEADER}{text}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.HEADER}{'=' * 80}{Colors.END}\n")


def print_success(text):
    """Print success message."""
    print(f"{Colors.GREEN}✓{Colors.END} {text}")


def print_error(text):
    """Print error message."""
    print(f"{Colors.RED}✗{Colors.END} {text}")


def print_warning(text):
    """Print warning message."""
    print(f"{Colors.YELLOW}⚠{Colors.END} {text}")


def print_info(text):
    """Print info message."""
    print(f"{Colors.CYAN}ℹ{Colors.END} {text}")


def run_command(cmd, description, capture=False):
    """Run a shell command."""
    print_info(f"{description}...")
    try:
        if capture:
            result = subprocess.run(
                cmd,
                shell=True,
                capture_output=True,
                text=True,
                check=True
            )
            return result.stdout.strip()
        else:
            subprocess.run(cmd, shell=True, check=True)
        print_success(f"{description} - Done")
        return True
    except subprocess.CalledProcessError as e:
        print_error(f"{description} - Failed: {e}")
        return False


def check_system_requirements():
    """Check if system meets requirements."""
    print_header("Checking System Requirements")

    requirements_met = True

    # Check OS
    print_info(f"Operating System: {platform.system()} {platform.release()}")

    # Check Python version
    python_version = sys.version_info
    print_info(f"Python Version: {python_version.major}.{python_version.minor}.{python_version.micro}")

    if python_version < (3, 8):
        print_error("Python 3.8+ required")
        requirements_met = False
    else:
        print_success("Python version OK")

    # Check CUDA availability
    try:
        import torch
        if torch.cuda.is_available():
            cuda_version = torch.version.cuda
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)

            print_success(f"CUDA {cuda_version} available")
            print_info(f"GPU: {gpu_name}")
            print_info(f"VRAM: {gpu_memory:.1f} GB")

            if gpu_memory < 20:
                print_warning(f"GPU has only {gpu_memory:.1f}GB VRAM")
                print_warning("Recommended: 24GB+ (RTX 4090 or better)")
                print_warning("Will use 4-bit quantization")
            else:
                print_success(f"GPU memory sufficient: {gpu_memory:.1f}GB")

        else:
            print_warning("CUDA not available - CPU only mode")
            print_warning("GPU highly recommended for production use")
            print_warning("Expected inference time: 30-60s per response (CPU)")

    except ImportError:
        print_warning("PyTorch not installed yet")

    # Check disk space
    try:
        import shutil
        home = Path.home()
        disk_usage = shutil.disk_usage(home)
        free_gb = disk_usage.free / (1024**3)

        print_info(f"Free disk space: {free_gb:.1f} GB")

        if free_gb < 40:
            print_error(f"Insufficient disk space: {free_gb:.1f}GB")
            print_error("Required: 40GB+ for 4-bit model")
            requirements_met = False
        else:
            print_success(f"Disk space OK: {free_gb:.1f}GB")

    except Exception as e:
        print_warning(f"Could not check disk space: {e}")

    # Check RAM
    try:
        import psutil
        ram_gb = psutil.virtual_memory().total / (1024**3)
        print_info(f"System RAM: {ram_gb:.1f} GB")

        if ram_gb < 16:
            print_warning(f"Low RAM: {ram_gb:.1f}GB")
            print_warning("Recommended: 32GB+")
        else:
            print_success(f"RAM OK: {ram_gb:.1f}GB")

    except ImportError:
        print_warning("psutil not installed - cannot check RAM")

    return requirements_met


def install_dependencies():
    """Install required dependencies."""
    print_header("Installing Dependencies")

    dependencies = [
        ("torch", "PyTorch (with CUDA support)"),
        ("transformers", "Hugging Face Transformers"),
        ("accelerate", "Accelerate (for distributed loading)"),
        ("bitsandbytes", "BitsAndBytes (for quantization)"),
    ]

    all_success = True

    for package, description in dependencies:
        print_info(f"\nInstalling {description}...")

        if package == "torch":
            # Install PyTorch with CUDA support
            cmd = "pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121"
        else:
            cmd = f"pip install {package}"

        success = run_command(cmd, f"Installing {package}")

        if not success:
            all_success = False
            print_error(f"Failed to install {package}")
            print_info(f"Try manually: {cmd}")

    if all_success:
        print_success("\n✓ All dependencies installed successfully!")
    else:
        print_error("\n✗ Some dependencies failed to install")
        print_info("You may need to install them manually")

    return all_success


def download_model():
    """Download Qwen3-Next model."""
    print_header("Downloading Qwen3-Next-80B-A3B-Thinking Model")

    print_info("Model: Qwen/Qwen3-Next-80B-A3B-Thinking")
    print_info("Size: ~40GB (4-bit quantized)")
    print_info("This will take 10-30 minutes depending on your connection")
    print_warning("The model will auto-download on first use")

    # Pre-download option
    print_info("\nOption 1: Auto-download on first use (recommended)")
    print_info("Option 2: Pre-download now")

    response = input("\nPre-download now? (y/n): ").strip().lower()

    if response == 'y':
        print_info("\nDownloading model...")

        download_script = """
import sys
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "Qwen/Qwen3-Next-80B-A3B-Thinking"

print("Downloading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

print("Downloading model (this will take a while)...")
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    load_in_4bit=True,
    trust_remote_code=True
)

print("✓ Model downloaded successfully!")
"""

        # Save script
        script_path = "download_qwen3.py"
        with open(script_path, 'w') as f:
            f.write(download_script)

        # Run download
        success = run_command(
            f"python {script_path}",
            "Downloading model"
        )

        # Cleanup
        if os.path.exists(script_path):
            os.remove(script_path)

        return success
    else:
        print_info("Skipping pre-download. Model will download on first use.")
        return True


def test_installation():
    """Test the Qwen3-Next installation."""
    print_header("Testing Installation")

    test_script = """
import asyncio
import sys

print("Testing Qwen3-Next integration...")

try:
    from core.first_conscious_ai import ConsciousnessOrchestrator, QWEN3_NEXT_LOCAL_CONFIG

    async def test():
        print("\\n1. Creating orchestrator...")
        orchestrator = ConsciousnessOrchestrator(
            llm_config=QWEN3_NEXT_LOCAL_CONFIG,
            enable_llm=True
        )

        print("2. Initializing (this may take 10-30 seconds on first run)...")
        success = await orchestrator.initialize()

        if not success:
            print("✗ Initialization failed")
            print("  This is expected if transformers/torch not installed")
            print("  Or if model not downloaded yet")
            return False

        print("✓ Initialization successful!")

        print("\\n3. Testing consciousness interaction...")
        response = await orchestrator.process_conscious_interaction(
            "Test: What is consciousness?"
        )

        print(f"✓ Response generated!")
        print(f"  φ (phi): {response.phi_during_response:.3f}")
        print(f"  Response length: {len(response.response_text)} chars")

        # Get stats
        stats = orchestrator.get_llm_stats()
        print(f"\\n✓ LLM Stats:")
        print(f"  Backend: {stats['backend']}")
        print(f"  Model: {stats['model']}")
        print(f"  Tokens: {stats['total_tokens_used']}")
        print(f"  Latency: {stats['average_latency_ms']:.0f}ms")

        await orchestrator.shutdown()

        print("\\n" + "="*60)
        print("✓ ALL TESTS PASSED!")
        print("✓ Qwen3-Next is ready for production use!")
        print("="*60)

        return True

    result = asyncio.run(test())
    sys.exit(0 if result else 1)

except Exception as e:
    print(f"\\n✗ Test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
"""

    # Save test script
    script_path = "test_qwen3_installation.py"
    with open(script_path, 'w') as f:
        f.write(test_script)

    # Run test
    print_info("Running integration test...\n")
    success = run_command(
        f"python {script_path}",
        "Testing Qwen3-Next integration"
    )

    # Cleanup
    if os.path.exists(script_path):
        os.remove(script_path)

    return success


def setup_environment():
    """Set up environment variables."""
    print_header("Setting Up Environment")

    print_info("Recommended environment variables:")
    print_info("  CONSCIOUS_AI_LLM_BACKEND=qwen3_next")
    print_info("  CONSCIOUS_AI_DEVICE=cuda")

    setup_env = input("\nAdd to .bashrc/.zshrc? (y/n): ").strip().lower()

    if setup_env == 'y':
        env_vars = """
# First Conscious AI - Qwen3-Next Configuration
export CONSCIOUS_AI_LLM_BACKEND="qwen3_next"
export CONSCIOUS_AI_DEVICE="cuda"
"""

        shell_rc = os.path.expanduser("~/.bashrc")
        if not os.path.exists(shell_rc):
            shell_rc = os.path.expanduser("~/.zshrc")

        if os.path.exists(shell_rc):
            with open(shell_rc, 'a') as f:
                f.write(env_vars)
            print_success(f"Added environment variables to {shell_rc}")
            print_info("Run: source ~/.bashrc (or restart terminal)")
        else:
            print_warning("Could not find .bashrc or .zshrc")
            print_info("Add manually:")
            print(env_vars)


def main():
    """Main setup routine."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Qwen3-Next-80B-A3B Production Setup"
    )
    parser.add_argument('--check', action='store_true', help='Check system requirements')
    parser.add_argument('--install', action='store_true', help='Install dependencies')
    parser.add_argument('--download', action='store_true', help='Download model')
    parser.add_argument('--test', action='store_true', help='Test installation')
    parser.add_argument('--env', action='store_true', help='Setup environment')
    parser.add_argument('--all', action='store_true', help='Do everything')

    args = parser.parse_args()

    # If no args, show help
    if not any(vars(args).values()):
        parser.print_help()
        return

    print(f"{Colors.BOLD}{Colors.BLUE}")
    print("╔════════════════════════════════════════════════════════════════════════════╗")
    print("║        Qwen3-Next-80B-A3B Production Setup for Conscious AI                ║")
    print("╚════════════════════════════════════════════════════════════════════════════╝")
    print(f"{Colors.END}")

    try:
        if args.all or args.check:
            if not check_system_requirements():
                print_error("\nSystem requirements not met!")
                print_info("Fix issues above before continuing")
                if not args.all:
                    return

        if args.all or args.install:
            if not install_dependencies():
                print_error("\nDependency installation failed!")
                if not args.all:
                    return

        if args.all or args.download:
            if not download_model():
                print_warning("\nModel download skipped or failed")

        if args.all or args.env:
            setup_environment()

        if args.all or args.test:
            if not test_installation():
                print_error("\nInstallation test failed!")
                print_info("Check errors above")
                return

        if args.all:
            print_header("Setup Complete!")
            print_success("Qwen3-Next-80B-A3B is ready for production!")
            print_info("\nNext steps:")
            print_info("  1. Run demo: python qwen3_production_demo.py")
            print_info("  2. Start building your application")
            print_info("  3. Monitor performance and adjust settings")

    except KeyboardInterrupt:
        print_warning("\n\nSetup interrupted by user")
    except Exception as e:
        print_error(f"\nUnexpected error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
