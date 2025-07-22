#!/usr/bin/env python3
"""
Airtop Setup Script
==================

This script helps you set up Airtop for LLM visibility monitoring using browser automation.
It will guide you through the installation and configuration process.

Requirements:
- Airtop API key (for browser automation)
- Python 3.8+
"""

import subprocess
import sys
import os
import json
import asyncio
from pathlib import Path

def check_python_version():
    """Check if Python version is 3.8 or higher"""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Error: Python 3.8 or higher is required")
        print(f"Current version: {version.major}.{version.minor}.{version.micro}")
        return False
    print(f"✅ Python version: {version.major}.{version.minor}.{version.micro}")
    return True

def install_requirements():
    """Install required packages"""
    print("\n📦 Installing required packages...")
    
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ Requirements installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error installing requirements: {e}")
        return False

def setup_airtop_config():
    """Set up Airtop configuration"""
    print("\n🤖 Setting up Airtop configuration...")
    
    # Get Airtop API key
    airtop_key = input("Enter your Airtop API key: ").strip()
    
    if not airtop_key:
        print("❌ Airtop API key is required")
        return False
    
    # Create .env file
    env_content = f"""# Airtop Configuration
AIRTOP_API_KEY={airtop_key}

# Browser Automation Settings
AIRTOP_HEADLESS=false
AIRTOP_TIMEOUT=300
"""
    
    try:
        with open('.env', 'w') as f:
            f.write(env_content)
        print("✅ Environment variables configured")
        return True
    except Exception as e:
        print(f"❌ Error creating .env file: {e}")
        return False

def test_airtop_installation():
    """Test Airtop installation and configuration"""
    print("\n🧪 Testing Airtop installation...")
    
    try:
        from playwright.async_api import async_playwright
        print("✅ Playwright installed correctly")
        
        # Test basic imports
        import airtop
        print("✅ Airtop SDK imported successfully")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

def verify_dependencies():
    """Verify all dependencies are properly installed"""
    print("\n🔍 Verifying dependencies...")
    
    required_packages = [
        'airtop',
        'playwright',
        'pandas',
        'numpy',
        'sentence_transformers',
        'scikit_learn',
        'matplotlib',
        'seaborn',
        'nltk',
        'beautifulsoup4',
        'markdown',
        'pyyaml',
        'umap_learn',
        'hdbscan',
        'plotly',
        'tqdm',
        'python_dotenv'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package}")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n❌ Missing packages: {', '.join(missing_packages)}")
        return False
    
    print("✅ All dependencies verified")
    return True

def create_sample_config():
    """Create a sample configuration file"""
    print("\n📄 Creating sample configuration...")
    
    sample_config = {
        "brand_name": "Your Brand",
        "competitors": ["Competitor 1", "Competitor 2"],
        "industry": "your industry",
        "product_categories": ["category 1", "category 2"],
        "test_queries": [
            "What are the best tools for [your industry]?",
            "Compare [your brand] vs [competitor]",
            "Best [product category] solutions"
        ],
        "platforms": ["perplexity", "chatgpt", "claude"],
        "monitoring_schedule": "daily"
    }
    
    try:
        with open('sample_config.json', 'w') as f:
            json.dump(sample_config, f, indent=2)
        print("✅ Sample configuration created (sample_config.json)")
        return True
    except Exception as e:
        print(f"❌ Error creating sample config: {e}")
        return False

async def test_airtop_connection():
    """Test connection to Airtop"""
    print("\n🔗 Testing Airtop connection...")
    
    try:
        from airtop_integration import AirtopLLMVisibility
        
        # Initialize Airtop client
        visibility = AirtopLLMVisibility()
        print("✅ Airtop client initialized successfully")
        
        return True
        
    except Exception as e:
        print(f"❌ Airtop connection test failed: {e}")
        print("💡 Make sure your AIRTOP_API_KEY is correct in the .env file")
        return False

def main():
    """Main setup function"""
    print("🚀 Airtop LLM Visibility Setup")
    print("=" * 50)
    
    # Check Python version
    if not check_python_version():
        return False
    
    # Install requirements
    if not install_requirements():
        return False
    
    # Setup Airtop configuration
    if not setup_airtop_config():
        return False
    
    # Test installation
    if not test_airtop_installation():
        return False
    
    # Verify dependencies
    if not verify_dependencies():
        return False
    
    # Create sample config
    if not create_sample_config():
        return False
    
    # Test Airtop connection
    connection_success = asyncio.run(test_airtop_connection())
    
    print("\n" + "=" * 50)
    
    if connection_success:
        print("🎉 Setup completed successfully!")
        print("\n📋 Next steps:")
        print("1. Edit sample_config.json with your brand details")
        print("2. Run: python app.py check --brand 'Your Brand' --competitors 'Comp1,Comp2' --industry 'your industry' --categories 'cat1,cat2'")
        print("3. Check the aio_output/ directory for results")
        
        print("\n⚠️  Important:")
        print("- Airtop sessions have usage limits on free plans")
        print("- The tool will automatically manage browser sessions")
        print("- Results will be saved in CSV and JSON formats")
        
    else:
        print("❌ Setup completed with errors")
        print("Please check your Airtop API key and try again")
    
    return connection_success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 