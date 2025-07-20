#!/usr/bin/env python
"""
Setup script for Audio Pipeline
Installs dependencies and sets up the environment
"""

import subprocess
import sys
import os
import platform
from pathlib import Path

def run_command(command, description=""):
    """Run a shell command and handle errors"""
    print(f"📦 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        return result
    except subprocess.CalledProcessError as e:
        print(f"❌ Error during {description}:")
        print(f"Command: {command}")
        print(f"Error: {e.stderr}")
        return None

def check_python_version():
    """Check if Python version is compatible"""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Python 3.8 or higher is required")
        sys.exit(1)
    print(f"✅ Python {version.major}.{version.minor}.{version.micro} detected")

def check_ffmpeg():
    """Check if FFmpeg is installed"""
    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
        print("✅ FFmpeg is installed")
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("⚠️  FFmpeg not found")
        return False

def install_ffmpeg():
    """Install FFmpeg based on the operating system"""
    system = platform.system().lower()

    if system == "darwin":  # macOS
        print("📦 Installing FFmpeg on macOS using Homebrew...")
        if run_command("brew install ffmpeg", "FFmpeg installation"):
            return True
    elif system == "linux":
        print("📦 Installing FFmpeg on Linux...")
        # Try different package managers
        if run_command("sudo apt update && sudo apt install -y ffmpeg", "FFmpeg installation (apt)"):
            return True
        elif run_command("sudo yum install -y ffmpeg", "FFmpeg installation (yum)"):
            return True
        elif run_command("sudo pacman -S ffmpeg", "FFmpeg installation (pacman)"):
            return True
    elif system == "windows":
        print("📦 Installing FFmpeg on Windows...")
        
        # Try winget first (Windows 10/11)
        print("   Trying winget installation...")
        if run_command("winget install ffmpeg", "FFmpeg installation (winget)"):
            return True
        
        # Try Chocolatey
        print("   Trying Chocolatey installation...")
        if run_command("choco install ffmpeg", "FFmpeg installation (Chocolatey)"):
            return True
        
        # Manual installation instructions
        print("   ❌ Automatic installation failed.")
        print("   📝 Please install FFmpeg manually:")
        print("      1. Download from: https://ffmpeg.org/download.html#build-windows")
        print("      2. Extract and add to PATH")
        print("      3. Or use: winget install ffmpeg")

    return False

def install_requirements():
    """Install Python requirements"""
    if os.path.exists("requirements.txt"):
        return run_command(f"{sys.executable} -m pip install -r requirements.txt", 
                          "Python requirements installation")
    else:
        print("❌ requirements.txt not found")
        return False

def create_directories():
    """Create necessary directories"""
    directories = ["input", "output", "temp", "logs"]
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        print(f"📁 Created directory: {directory}")

def create_env_file():
    """Create .env file template"""
    env_lines = [
        "# Environment Variables for Audio Pipeline",
        "# Copy this file to .env and fill in your values",
        "",
        "# LM Studio Configuration", 
        "LM_STUDIO_URL=http://localhost:1234",
        "LM_STUDIO_MODEL=",
        "",
        "# OpenAI API (if using cloud Whisper)",
        "OPENAI_API_KEY=",
        "",
        "# Logging",
        "LOG_LEVEL=INFO",
        "",
        "# Processing", 
        "DEFAULT_WHISPER_MODEL=large-v3",
        "DEFAULT_OUTPUT_DIR=./output",
    ]

    with open('.env.template', 'w') as f:
        f.write('\n'.join(env_lines))
    print("✅ Created .env.template file")

def main():
    """Main setup function"""
    print("🚀 Audio Pipeline Setup")
    print("=" * 40)

    # Check Python version
    check_python_version()

    # Check and install FFmpeg if needed
    if not check_ffmpeg():
        print("⚠️  FFmpeg is required for audio processing")
        response = input("Would you like to install FFmpeg automatically? (y/n): ")
        if response.lower() == 'y':
            if not install_ffmpeg():
                print("❌ Automatic FFmpeg installation failed")
                print("   Please install FFmpeg manually and run setup again")
                sys.exit(1)
        else:
            print("❌ Please install FFmpeg manually before continuing")
            sys.exit(1)

    # Install Python requirements
    if not install_requirements():
        print("❌ Failed to install Python requirements")
        sys.exit(1)

    # Create directories
    create_directories()

    # Create environment template
    create_env_file()

    print("\n🎉 Setup completed successfully!")
    print("\nNext steps:")
    print("1. Copy config_template.yaml to config.yaml and customize")
    print("2. Start your LM Studio server with your preferred model")
    print("3. Run: python audio_pipeline.py --help")
    print("\nExample usage:")
    print("python audio_pipeline.py input/your_audio.mp3 --output-dir output/")

if __name__ == "__main__":
    main()
