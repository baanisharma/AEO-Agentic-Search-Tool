#!/usr/bin/env python3
"""
AIO Search Tool API Server
==========================

Starts the FastAPI backend server for AI optimization analysis
"""

import uvicorn
import os

def main():
    print("🚀 Starting AIO Search Tool API Server")
    print("=" * 50)
    print("📡 API Documentation: http://localhost:8000/docs")
    print("🔍 Health Check: http://localhost:8000/api/health")
    print("🎯 Backend API Server: http://localhost:8000")
    print("=" * 50)
    
    # Check environment
    airtop_key = os.getenv('AIRTOP_API_KEY')
    if airtop_key:
        print("✅ Airtop API key found - Full functionality available")
    else:
        print("⚠️  Airtop API key not found - Visibility checker will be limited")
        print("💡 Add AIRTOP_API_KEY to .env file for full features")
    
    print("\n🔥 Starting server on http://localhost:8000...")
    
    # Use string import format for reload to work
    uvicorn.run(
        "api_server:app",  # Import string instead of app object
        host="0.0.0.0",
        port=8000,
        reload=True,  # Auto-reload on code changes
        log_level="info"
    )

if __name__ == "__main__":
    main() 