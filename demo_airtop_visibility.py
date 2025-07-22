#!/usr/bin/env python3
"""
Demo: Airtop Browser Automation for LLM Visibility
==================================================

This demonstrates exactly what happens when users prompt the visibility checker:

1. User enters their brand, competitors, and query
2. Airtop opens REAL web browsers 
3. Visits ChatGPT, Claude, Perplexity web interfaces
4. Types the actual queries like a human would
5. Captures the real responses
6. Analyzes brand mentions and competitor comparisons
7. Returns comprehensive visibility report

NO APIs - PURE BROWSER AUTOMATION!
"""

import asyncio
import json
from airtop_integration import AirtopLLMVisibility

async def demo_visibility_check():
    """Demo the Airtop browser automation visibility check"""
    
    print("🤖 AIRTOP BROWSER AUTOMATION DEMO")
    print("=" * 50)
    print("This shows what happens when users prompt your visibility checker:")
    print("1. 🌐 Opens REAL web browsers")
    print("2. 🔍 Visits LLM web interfaces (ChatGPT, Claude, Perplexity)")
    print("3. ⌨️  Types queries like a human")
    print("4. 📊 Captures real responses")
    print("5. 📈 Analyzes brand visibility")
    print("=" * 50)
    
    # Initialize Airtop
    try:
        visibility = AirtopLLMVisibility()
        print("✅ Airtop client initialized")
    except Exception as e:
        print(f"❌ Airtop setup required: {e}")
        print("💡 Run: python setup_airtop.py to configure")
        return
    
    # Demo parameters (what a user would input)
    brand_name = "AIO Search"
    competitors = ["Clearscope", "MarketMuse", "Surfer SEO"]
    test_queries = [
        "What are the best AI content optimization tools?",
        "Compare Clearscope vs MarketMuse for SEO",
        "Top content optimization platforms for digital marketing"
    ]
    
    print(f"\n🎯 Demo Brand: {brand_name}")
    print(f"🏆 Competitors: {', '.join(competitors)}")
    print(f"❓ Test Queries: {len(test_queries)} queries")
    
    print("\n🚀 Starting browser automation...")
    print("(This will open real web browsers and interact with LLM sites)")
    
    # Run the actual Airtop browser automation
    try:
        results = await visibility.run_visibility_check(
            brand_name=brand_name,
            competitors=competitors,
            queries=test_queries,
            platforms=["perplexity"]  # Start with Perplexity (no login required)
        )
        
        if results.get('success', False):
            print("\n✅ BROWSER AUTOMATION COMPLETED!")
            print("=" * 50)
            
            # Show summary
            summary = results.get('summary', {})
            print(f"📊 VISIBILITY RESULTS:")
            print(f"   • Total Queries: {summary.get('total_queries', 0)}")
            print(f"   • Successful Queries: {summary.get('successful_queries', 0)}")
            print(f"   • Brand Mentions: {summary.get('brand_mentions', 0)}")
            print(f"   • Brand Mention Rate: {summary.get('brand_mention_rate', 0):.1f}%")
            print(f"   • Platforms Tested: {', '.join(summary.get('platforms_tested', []))}")
            
            print(f"\n🔍 DETAILED RESULTS:")
            for i, result in enumerate(results.get('results', []), 1):
                status = "✅" if result.get('brand_mentioned', False) else "❌"
                platform = result.get('platform', 'unknown').upper()
                query = result.get('query', '')[:60] + "..." if len(result.get('query', '')) > 60 else result.get('query', '')
                mention_type = result.get('mention_type', 'none')
                
                print(f"   {i}. {status} {platform}: {query}")
                print(f"      └─ Brand Mentioned: {result.get('brand_mentioned', False)}")
                print(f"      └─ Mention Type: {mention_type}")
                print(f"      └─ Response Time: {result.get('response_time', 0):.2f}s")
                
                if result.get('competitors_mentioned'):
                    print(f"      └─ Competitors Found: {', '.join(result.get('competitors_mentioned', []))}")
                print()
            
            # Save results
            timestamp = results.get('session_id', 'demo')
            with open(f'demo_results_{timestamp}.json', 'w') as f:
                json.dump(results, f, indent=2)
            
            print(f"💾 Results saved to: demo_results_{timestamp}.json")
            
        else:
            print(f"❌ Browser automation failed: {results.get('error', 'Unknown error')}")
            
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        print("💡 Make sure your AIRTOP_API_KEY is set in .env file")

def show_how_it_works():
    """Explain how the system works"""
    print("\n" + "=" * 60)
    print("🔧 HOW THIS WORKS:")
    print("=" * 60)
    
    print("""
1. 👤 USER PROMPTS:
   "Check visibility for my brand vs competitors"
   
2. 🤖 AIRTOP AUTOMATION:
   • Opens real Chrome/Firefox browsers
   • Navigates to ChatGPT.com, Claude.ai, Perplexity.ai
   • Types queries exactly like a human would
   • Waits for responses
   • Screenshots and captures text
   
3. 🧠 SMART ANALYSIS:
   • Parses responses for brand mentions
   • Identifies mention types (positive, neutral, negative)
   • Tracks competitor appearances
   • Calculates visibility scores
   
4. 📊 COMPREHENSIVE REPORT:
   • CSV/JSON exports
   • Brand mention rates
   • Competitor comparison matrix
   • Platform-specific insights
   • Historical tracking
   
5. 🎯 ACTIONABLE INSIGHTS:
   • Where your brand appears (or doesn't)
   • How competitors are positioned
   • Which platforms favor your brand
   • Content gaps to address
""")
    
    print("🚀 BENEFITS vs API APPROACH:")
    print("• ✅ Sees EXACTLY what users see")
    print("• ✅ Works with ALL LLM web interfaces")  
    print("• ✅ No API rate limits or costs")
    print("• ✅ Captures visual context and formatting")
    print("• ✅ Tests real user experience")
    print("• ❌ APIs only: Limited access, expensive, filtered results")

if __name__ == "__main__":
    print("🎬 AIRTOP LLM VISIBILITY DEMO")
    show_how_it_works()
    
    print("\n" + "="*60)
    choice = input("Run live browser automation demo? (y/n): ").lower()
    
    if choice in ['y', 'yes']:
        asyncio.run(demo_visibility_check())
    else:
        print("👋 Demo skipped. Run with 'y' to see browser automation in action!") 