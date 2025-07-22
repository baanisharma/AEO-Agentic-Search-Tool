#!/usr/bin/env python3
"""
AIO Search Tool - Web Application Startup Script
"""

import os
import sys
import subprocess
import webbrowser
import time
from pathlib import Path

def check_dependencies():
    """Check if required dependencies are installed"""
    required_packages = [
        'streamlit',
        'plotly',
        'pandas',
        'streamlit-option-menu'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"Missing required packages: {', '.join(missing_packages)}")
        print("Installing missing packages...")
        
        for package in missing_packages:
            try:
                subprocess.check_call([sys.executable, '-m', 'pip', 'install', package])
                print(f"✓ Installed {package}")
            except subprocess.CalledProcessError:
                print(f"✗ Failed to install {package}")
                return False
    
    return True

def setup_environment():
    """Setup environment variables and directories"""
    # Create necessary directories
    directories = ['aio_output', 'logs', 'temp']
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
    
    # Set environment variables
    os.environ['AIO_OUTPUT_DIR'] = str(Path.cwd() / 'aio_output')
    os.environ['AIO_LOG_DIR'] = str(Path.cwd() / 'logs')
    os.environ['AIO_TEMP_DIR'] = str(Path.cwd() / 'temp')

def start_web_app():
    """Start the Streamlit web application"""
    print("🚀 Starting AIO Search Tool Web Application...")
    print("=" * 50)
    
    # Check if web_app.py exists
    if not Path('web_app.py').exists():
        print("❌ Error: web_app.py not found!")
        print("Please make sure you're in the correct directory.")
        return False
    
    # Setup environment
    setup_environment()
    
    # Start Streamlit
    try:
        print("📊 Launching Streamlit application...")
        print("🌐 The application will open in your default browser")
        print("⏳ Please wait a moment...")
        
        # Start Streamlit in a subprocess
        process = subprocess.Popen([
            sys.executable, '-m', 'streamlit', 'run', 'web_app.py',
            '--server.port', '8501',
            '--server.address', 'localhost',
            '--browser.gatherUsageStats', 'false'
        ])
        
        # Wait a moment for the server to start
        time.sleep(3)
        
        # Open browser
        try:
            webbrowser.open('http://localhost:8501')
            print("✅ Application started successfully!")
            print("🔗 URL: http://localhost:8501")
            print("🛑 Press Ctrl+C to stop the application")
            
            # Wait for the process to complete
            process.wait()
            
        except Exception as e:
            print(f"⚠️ Could not open browser automatically: {e}")
            print("🔗 Please open http://localhost:8501 in your browser")
            process.wait()
            
    except KeyboardInterrupt:
        print("\n🛑 Stopping application...")
        if 'process' in locals():
            process.terminate()
        print("✅ Application stopped")
    except Exception as e:
        print(f"❌ Error starting application: {e}")
        return False
    
    return True

def main():
    """Main function"""
    print("🔍 AIO Search Tool - Web Application")
    print("=" * 40)
    
    # Check dependencies
    if not check_dependencies():
        print("❌ Failed to install required dependencies")
        return 1
    
    # Start the application
    if start_web_app():
        return 0
    else:
        return 1

if __name__ == "__main__":
    sys.exit(main()) 