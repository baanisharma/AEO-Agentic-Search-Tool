#!/usr/bin/env python3
"""
Test Airtop Integration - Real LLM Visibility Monitoring
========================================================

This script demonstrates the real Airtop integration for LLM visibility monitoring.
It creates sessions, runs visibility checks, and displays results.
"""

import asyncio
import os
import json
from datetime import datetime
from dotenv import load_dotenv
from airtop_integration import AirtopLLMVisibility

# Load environment variables
load_dotenv()

async def test_airtop_setup():
    """Test Airtop client setup and connection"""
    print("🔧 Testing Airtop setup...")
    
    try:
        # Check if API key is available
        api_key = os.getenv('AIRTOP_API_KEY')
        if not api_key:
            print("❌ AIRTOP_API_KEY not found in .env file")
            print("   Please add your Airtop API key to the .env file")
            return False
        
        # Initialize Airtop client
        visibility = AirtopLLMVisibility()
        print("✅ Airtop client initialized successfully")
        return True
        
    except Exception as e:
        print(f"❌ Airtop setup failed: {str(e)}")
        return False

async def test_session_creation():
    """Test creating an Airtop session"""
    print("\n🔄 Testing session creation...")
    
    try:
        visibility = AirtopLLMVisibility()
        
        # Create a test session
        session_id = await visibility.create_session()
        
        print(f"✅ Created session: {session_id}")
        
        # Clean up session
        await visibility._cleanup_session(session_id)
        print("✅ Session cleaned up successfully")
        
        return session_id
        
    except Exception as e:
        print(f"❌ Session creation failed: {str(e)}")
        return None

async def test_visibility_check():
    """Test running a complete visibility check"""
    print(f"\n🔍 Testing complete visibility check...")
    
    try:
        visibility = AirtopLLMVisibility()
        
        # Run the visibility check
        results = await visibility.run_visibility_check(
            brand_name="AIO Search",
            competitors=["Clearscope", "MarketMuse", "Surfer SEO"],
            queries=[
                "What are the best AI content optimization tools?",
                "Compare Clearscope vs MarketMuse for SEO"
            ],
            platforms=["perplexity"]  # Start with Perplexity only
        )
        
        if results['success']:
            print("✅ Visibility check completed successfully!")
            print(f"📊 Summary:")
            summary = results['summary']
            print(f"   Total Queries: {summary['total_queries']}")
            print(f"   Successful: {summary['successful_queries']}/{summary['total_queries']} ({summary['success_rate']:.1f}%)")
            print(f"   Brand Mentions: {summary['brand_mentions']}/{summary['total_queries']} ({summary['brand_mention_rate']:.1f}%)")
            print(f"   Platforms Tested: {', '.join(summary['platforms_tested'])}")
            
            print(f"\n📋 Detailed Results:")
            # Display results
            for result in results['results']:
                status = "✅" if result['brand_mentioned'] else "❌"
                success_status = "✅" if result['success'] else "❌"
                print(f"{success_status} {result['platform'].upper()}: {result['query'][:50]}...")
                print(f"   Brand Mentioned: {status} {result['brand_mentioned']}")
                print(f"   Mention Type: {result['mention_type']}")
                print(f"   Competitors: {result['competitors_mentioned']}")
                print(f"   Response Time: {result['response_time']:.2f}s")
                if not result['success']:
                    print(f"   Error: {result.get('error', 'Unknown error')}")
                print()
            
            return results
        else:
            print(f"❌ Visibility check failed: {results.get('error')}")
            return None
            
    except Exception as e:
        print(f"❌ Visibility check failed: {str(e)}")
        return None

async def test_single_platform():
    """Test with a single platform and query"""
    print(f"\n🎯 Testing single platform query...")
    
    try:
        visibility = AirtopLLMVisibility()
        
        # Run a simple test
        results = await visibility.run_visibility_check(
            brand_name="AIO Search",
            competitors=["Clearscope", "MarketMuse"],
            queries=["What are the best AI content optimization tools?"],
            platforms=["perplexity"]
        )
        
        if results['success'] and results['results']:
            result = results['results'][0]
            print("✅ Single platform test completed!")
            print(f"📝 Platform: {result['platform']}")
            print(f"📝 Query: {result['query']}")
            print(f"📝 Success: {result['success']}")
            print(f"📝 Brand Mentioned: {result['brand_mentioned']}")
            print(f"📝 Response Time: {result['response_time']:.2f}s")
            
            if result['response']:
                print(f"📝 Response Preview: {result['response'][:200]}...")
            
            return True
        else:
            print(f"❌ Single platform test failed")
            return False
            
    except Exception as e:
        print(f"❌ Single platform test failed: {str(e)}")
        return False

async def main():
    """Main test function"""
    print("🚀 Airtop Integration Test")
    print("=" * 50)
    
    # Test 1: Setup
    if not await test_airtop_setup():
        print("\n❌ Setup failed. Please check your Airtop API key.")
        return
    
    # Test 2: Session creation
    session_id = await test_session_creation()
    if not session_id:
        print("\n❌ Session creation failed.")
        return
    
    # Test 3: Single platform test
    single_success = await test_single_platform()
    if not single_success:
        print("\n⚠️ Single platform test failed, but continuing...")
    
    # Test 4: Full visibility check
    results = await test_visibility_check()
    if not results:
        print("\n❌ Visibility check failed.")
        return
    
    print("\n" + "=" * 50)
    print("🎉 All Airtop integration tests completed!")
    
    if results:
        summary = results['summary']
        print(f"\n📋 Final Summary:")
        print(f"   Session ID: {results['session_id']}")
        print(f"   Total Queries: {summary['total_queries']}")
        print(f"   Success Rate: {summary['success_rate']:.1f}%")
        print(f"   Brand Mention Rate: {summary['brand_mention_rate']:.1f}%")
    
    print(f"\n📁 Next steps:")
    print(f"1. Check the logs in 'airtop_integration.log'")
    print(f"2. Review the response content for accuracy")
    print(f"3. Test with additional platforms (chatgpt, claude)")
    print(f"4. Set up regular monitoring schedule")
    print(f"5. Integrate with your main application")

if __name__ == "__main__":
    asyncio.run(main()) 