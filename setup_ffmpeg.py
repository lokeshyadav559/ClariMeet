#!/usr/bin/env python
"""
FFmpeg Setup Script for Audio Pipeline
Handles FFmpeg installation and configuration on Windows
"""

import os
import sys
import subprocess
import requests
import zipfile
import shutil
from pathlib import Path

def check_ffmpeg():
    """Check if FFmpeg is already installed"""
    try:
        result = subprocess.run(['ffmpeg', '-version'], 
                              capture_output=True, 
                              check=True, 
                              timeout=10)
        print("✅ FFmpeg is already installed and working!")
        return True
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
        return False

def install_ffmpeg_chocolatey():
    """Install FFmpeg using Chocolatey"""
    print("🔧 Attempting to install FFmpeg using Chocolatey...")
    
    try:
        # Check if Chocolatey is available
        result = subprocess.run(['choco', '--version'], 
                              capture_output=True, 
                              check=True)
        print(f"✅ Chocolatey found: {result.stdout.decode().strip()}")
        
        # Install FFmpeg
        print("📦 Installing FFmpeg via Chocolatey...")
        result = subprocess.run(['choco', 'install', 'ffmpeg', '-y'], 
                              capture_output=True, 
                              text=True)
        
        if result.returncode == 0:
            print("✅ FFmpeg installed successfully via Chocolatey!")
            return True
        else:
            print(f"❌ Chocolatey installation failed: {result.stderr}")
            return False
            
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        print(f"❌ Chocolatey not available: {e}")
        return False

def download_ffmpeg_manual():
    """Download and install FFmpeg manually"""
    print("🔧 Downloading FFmpeg manually...")
    
    # Create ffmpeg directory
    ffmpeg_dir = Path("C:/ffmpeg")
    ffmpeg_dir.mkdir(exist_ok=True)
    
    # Download URL for Windows FFmpeg
    download_url = "https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-master-latest-win64-gpl.zip"
    
    try:
        print(f"📥 Downloading FFmpeg from: {download_url}")
        response = requests.get(download_url, stream=True)
        response.raise_for_status()
        
        # Save the file
        zip_path = ffmpeg_dir / "ffmpeg.zip"
        with open(zip_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        print("📦 Extracting FFmpeg...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(ffmpeg_dir)
        
        # Find the extracted directory
        extracted_dir = None
        for item in ffmpeg_dir.iterdir():
            if item.is_dir() and item.name.startswith('ffmpeg'):
                extracted_dir = item
                break
        
        if extracted_dir:
            # Move contents to ffmpeg directory
            for item in extracted_dir.iterdir():
                shutil.move(str(item), str(ffmpeg_dir / item.name))
            shutil.rmtree(extracted_dir)
        
        # Clean up zip file
        zip_path.unlink()
        
        print("✅ FFmpeg downloaded and extracted successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Manual download failed: {e}")
        return False

def add_to_path():
    """Add FFmpeg to system PATH"""
    print("🔧 Adding FFmpeg to PATH...")
    
    ffmpeg_path = "C:/ffmpeg/bin"
    
    if os.path.exists(ffmpeg_path):
        # Get current PATH
        current_path = os.environ.get('PATH', '')
        
        if ffmpeg_path not in current_path:
            # Add to PATH for current session
            os.environ['PATH'] = ffmpeg_path + os.pathsep + current_path
            print(f"✅ Added {ffmpeg_path} to PATH for current session")
        else:
            print("✅ FFmpeg path already in PATH")
        return True
    else:
        print(f"❌ FFmpeg path not found: {ffmpeg_path}")
        return False

def verify_installation():
    """Verify FFmpeg installation"""
    print("🔍 Verifying FFmpeg installation...")
    
    try:
        result = subprocess.run(['ffmpeg', '-version'], 
                              capture_output=True, 
                              check=True, 
                              timeout=10)
        
        version_output = result.stdout.decode()
        print("✅ FFmpeg is working correctly!")
        print(f"📋 Version info: {version_output.split('Copyright')[0].strip()}")
        return True
        
    except Exception as e:
        print(f"❌ FFmpeg verification failed: {e}")
        return False

def main():
    """Main installation function"""
    print("=" * 60)
    print("FFMPEG SETUP FOR AUDIO PIPELINE")
    print("=" * 60)
    
    # Check if already installed
    if check_ffmpeg():
        return True
    
    print("FFmpeg not found. Installing...")
    
    # Try Chocolatey first
    if install_ffmpeg_chocolatey():
        if verify_installation():
            return True
    
    # Try manual download
    print("\n🔄 Trying manual download...")
    if download_ffmpeg_manual():
        if add_to_path():
            if verify_installation():
                return True
    
    print("\n❌ FFmpeg installation failed!")
    print("📋 Manual installation steps:")
    print("1. Download FFmpeg from: https://ffmpeg.org/download.html")
    print("2. Extract to C:/ffmpeg/")
    print("3. Add C:/ffmpeg/bin to your system PATH")
    print("4. Restart your terminal/IDE")
    
    return False

if __name__ == "__main__":
    success = main()
    if success:
        print("\n🎉 FFmpeg setup completed successfully!")
        print("You can now run the audio pipeline.")
    else:
        print("\n⚠️  FFmpeg setup failed. Please install manually.")
        sys.exit(1) 