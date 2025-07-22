"""
AI Search Engine Visibility Monitor - Live Browser Automation
===========================================================

This tool monitors how brands and products are represented in the answers given by 
public-facing AI search engines, such as ChatGPT with browsing, Perplexity, Copilot, 
and Google AI Overviews.

🎯 GOAL: See exactly what real users see when they ask AI systems about a brand, 
product, or topic. Track actual, up-to-date answers including sources and citations.

🔧 HOW WE DO THIS:
• Automate real web browsers to interact with public AI search interfaces
• Enter queries as a user would and wait for AI answers to load
• Extract answer text, citations, and links the AI provides
• Use cloud browser automation (Airtop) for scale and anti-bot handling
• Save data for trend analysis, sentiment tracking, and competitor comparison

✅ WHY THIS APPROACH:
• AI search engines update answers in real-time with new sources
• Monitor live web interface = most accurate current picture
• Detect emerging issues, misinformation, or PR crises immediately
• Verify if content/reputation efforts appear in actual user-facing answers
• "See through the eyes" of real users querying AI search engines

🚀 RESULT: Live AI visibility monitoring with real-time citation tracking
"""

import os
import json
import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import pandas as pd
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('ai_search_visibility.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class AirtopLLMVisibility:
    """
    Live AI Search Engine Visibility Monitor
    
    Monitors brand presence across ChatGPT, Perplexity, Copilot, and Google AI Overviews
    using real browser automation to capture exactly what users see.
    """
    
    def __init__(self):
        """Initialize AI Search Engine Visibility Monitor"""
        self.api_key = os.getenv('AIRTOP_API_KEY')
        
        # Always load AI search engine configurations
        self.ai_search_engines = self._get_ai_search_config()
        self.active_sessions = {}
        
        # Initialize Airtop client if API key is available
        if self.api_key:
            try:
                from airtop import AsyncAirtop
                self.airtop_client = AsyncAirtop(api_key=self.api_key)
                logger.info("🤖 Airtop browser automation client initialized")
                self.demo_mode = False
            except ImportError:
                logger.warning("⚠️ Airtop SDK not installed. Running in demo mode.")
                self.airtop_client = None
                self.demo_mode = True
            except Exception as e:
                logger.warning(f"⚠️ Failed to initialize Airtop client: {str(e)}. Running in demo mode.")
                self.airtop_client = None
                self.demo_mode = True
        else:
            logger.warning("⚠️ AIRTOP_API_KEY not found. Running in demo mode.")
            self.airtop_client = None
            self.demo_mode = True
        
    def _get_ai_search_config(self) -> Dict:
        """Get configuration for AI search engines we monitor"""
        return {
            "perplexity": {
                "name": "Perplexity AI",
                "url": "https://www.perplexity.ai/",
                "login_required": False,
                "citation_tracking": True,
                "real_time_search": True,
                "selectors": {
                    "input": "textarea[placeholder*='Ask anything']",
                    "send": "button[aria-label='Send']",
                    "response": ".prose, [data-testid='answer']",
                    "sources": "[data-testid='source'], .citation",
                    "citations": "a[href*='http'], .source-link"
                },
                "wait_time": 15
            },
            "chatgpt_browse": {
                "name": "ChatGPT with Browsing",
                "url": "https://chat.openai.com/",
                "login_required": True,
                "citation_tracking": True,
                "real_time_search": True,
                "selectors": {
                    "input": "textarea[data-id='root']",
                    "send": "button[data-testid='send-button']",
                    "response": "[data-testid*='conversation-turn']",
                    "sources": "[data-testid='source'], .citation-link",
                    "citations": "a[href*='http']:not([href*='chat.openai.com'])"
                },
                "wait_time": 20
            },
            "copilot": {
                "name": "Microsoft Copilot",
                "url": "https://copilot.microsoft.com/",
                "login_required": False,
                "citation_tracking": True,
                "real_time_search": True,
                "selectors": {
                    "input": "textarea[aria-label*='Ask me anything']",
                    "send": "button[aria-label='Send']",
                    "response": ".response-content, [data-testid='response']",
                    "sources": ".source-attribution, .citation",
                    "citations": "a[href*='http']"
                },
                "wait_time": 18
            },
            "google_ai": {
                "name": "Google AI Overview",
                "url": "https://www.google.com/",
                "login_required": False,
                "citation_tracking": True,
                "real_time_search": True,
                "selectors": {
                    "input": "textarea[name='q'], input[name='q']",
                    "send": "button[type='submit'], input[type='submit']",
                    "response": "[data-attrid='AIOverview'], .ai-overview",
                    "sources": ".source-link, .citation",
                    "citations": "a[href*='http']:not([href*='google.com/search'])"
                },
                "wait_time": 10
            }
        }
    
    async def cleanup_all_sessions(self):
        """Clean up all existing browser sessions"""
        try:
            logger.info("🧹 Cleaning up existing browser sessions...")
            
            sessions_response = await self.airtop_client.sessions.list()
            
            if hasattr(sessions_response, 'data') and sessions_response.data and hasattr(sessions_response.data, 'sessions'):
                for session in sessions_response.data.sessions:
                    try:
                        if session.status == 'running':
                            logger.info(f"🔄 Terminating session: {session.id}")
                            await self.airtop_client.sessions.terminate(id=session.id)
                    except Exception as e:
                        logger.warning(f"⚠️ Failed to terminate session {session.id}: {str(e)}")
                        
            logger.info("✅ Session cleanup completed")
            
        except Exception as e:
            logger.warning(f"⚠️ Session cleanup failed: {str(e)}")
    
    async def create_browser_session(self) -> str:
        """Create a new browser automation session"""
        try:
            await self.cleanup_all_sessions()
            
            logger.info("🌐 Creating new browser session...")
            
            session_response = await self.airtop_client.sessions.create()
            session_id = session_response.data.id
            
            logger.info(f"✅ Browser session created: {session_id}")
            self.active_sessions[session_id] = {
                'created_at': datetime.now(),
                'windows': []
            }
            
            return session_id
            
        except Exception as e:
            logger.error(f"❌ Failed to create browser session: {str(e)}")
            raise
    
    async def create_browser_window(self, session_id: str) -> str:
        """Create a new browser window in the session"""
        try:
            logger.info(f"🪟 Creating browser window in session: {session_id}")
            
            window_response = await self.airtop_client.windows.create(
                session_id=session_id,
                wait_until="load"
            )
            window_id = window_response.data.window_id
            
            logger.info(f"✅ Browser window created: {window_id}")
            
            if session_id in self.active_sessions:
                self.active_sessions[session_id]['windows'].append(window_id)
            
            return window_id
            
        except Exception as e:
            logger.error(f"❌ Failed to create browser window: {str(e)}")
            raise
    
    async def run_visibility_check(self, brand_name: str, competitors: List[str], 
                                 queries: List[str] = None, platforms: List[str] = None) -> Dict:
        """
        🎯 MAIN FUNCTION: Monitor brand visibility across AI search engines
        
        This simulates real users asking AI search engines about your brand and captures:
        • Actual AI responses users see
        • Citations and sources cited
        • Real-time search results
        • Competitor mentions
        • Sentiment and positioning
        
        Args:
            brand_name: Your brand to monitor
            competitors: Competitor brands to track
            queries: Questions to ask AI search engines
            platforms: Which AI search engines to test
            
        Returns:
            Dict: Complete visibility analysis with real-time data
        """
        try:
            if queries is None:
                queries = self._generate_ai_search_queries(brand_name, competitors)
            
            if platforms is None:
                platforms = ["perplexity"]  # Start with Perplexity (no login required)
            
            # If in demo mode, return demo data
            if self.demo_mode or not self.airtop_client:
                logger.info(f"🎭 Running in demo mode for: {brand_name}")
                return self._generate_demo_results(brand_name, competitors, queries, platforms)
            
            logger.info(f"🚀 Starting AI Search Visibility Check for: {brand_name}")
            logger.info(f"🔍 Testing {len(queries)} queries across {len(platforms)} AI search engines")
            
            # Create browser session
            session_id = await self.create_browser_session()
            
            results = []
            
            for platform in platforms:
                if platform not in self.ai_search_engines:
                    logger.warning(f"⚠️ AI search engine '{platform}' not configured, skipping")
                    continue
                
                engine_config = self.ai_search_engines[platform]
                logger.info(f"🤖 Testing {engine_config['name']}...")
                
                for query in queries:
                    try:
                        # Create window for this query
                        window_id = await self.create_browser_window(session_id)
                        
                        # Run the query and capture live AI response
                        result = await self._capture_live_ai_response(
                            session_id, window_id, platform, engine_config, 
                            query, brand_name, competitors
                        )
                        
                        results.append(result)
                        
                        # Brief pause between queries
                        await asyncio.sleep(3)
                        
                    except Exception as e:
                        logger.error(f"❌ Failed to test query '{query}' on {platform}: {str(e)}")
                        results.append({
                            'platform': platform,
                            'engine_name': engine_config['name'],
                            'query': query,
                            'success': False,
                            'error': str(e),
                            'timestamp': datetime.now().isoformat()
                        })
            
            # Cleanup browser session
            await self._cleanup_session(session_id)
            
            logger.info(f"✅ AI Search Visibility Check completed: {len(results)} queries processed")
            
            return {
                'success': True,
                'session_id': session_id,
                'brand_monitored': brand_name,
                'competitors_tracked': competitors,
                'ai_engines_tested': platforms,
                'results': results,
                'summary': self._generate_visibility_summary(results, brand_name),
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"❌ AI Search Visibility Check failed: {str(e)}")
            return {'success': False, 'error': str(e)}
    
    def _generate_ai_search_queries(self, brand_name: str, competitors: List[str]) -> List[str]:
        """Generate realistic queries users ask AI search engines"""
        queries = [
            f"What is {brand_name}?",
            f"Best alternatives to {brand_name}",
            f"Is {brand_name} worth it?",
            f"{brand_name} vs {competitors[0] if competitors else 'competitors'}",
            f"Problems with {brand_name}",
            f"Latest news about {brand_name}",
            f"How much does {brand_name} cost?",
            f"{brand_name} reviews and ratings",
            f"Who owns {brand_name}?",
            f"Is {brand_name} better than {competitors[0] if competitors else 'other options'}?"
        ]
        return queries[:6]  # Limit for demo
    
    async def _capture_live_ai_response(self, session_id: str, window_id: str, platform: str, 
                                      engine_config: Dict, query: str, brand_name: str, 
                                      competitors: List[str]) -> Dict:
        """
        🎯 CORE FUNCTION: Capture live AI search engine response
        
        This simulates a real user:
        1. Opens the AI search engine
        2. Types the query
        3. Waits for AI to generate answer
        4. Captures the full response + citations
        5. Analyzes brand mentions and sentiment
        """
        start_time = datetime.now()
        
        try:
            logger.info(f"🔍 Querying {engine_config['name']}: {query[:60]}...")
            
            # Navigate to AI search engine
            await self.airtop_client.windows.load_url(
                session_id=session_id,
                window_id=window_id,
                url=engine_config["url"]
            )
            
            # Wait for page to load
            await asyncio.sleep(5)
            
            # Type the query (simulating real user)
            await self.airtop_client.windows.type(
                session_id=session_id,
                window_id=window_id,
                text=query,
                element_description="search input field"
            )
            
            # Submit query (simulating real user clicking)
            await self.airtop_client.windows.click(
                session_id=session_id,
                window_id=window_id,
                element_description="search button"
            )
            
            # Wait for AI to generate response
            logger.info(f"⏳ Waiting for {engine_config['name']} to generate AI response...")
            await asyncio.sleep(engine_config["wait_time"])
            
            # Capture the complete page content
            response_content = await self.airtop_client.windows.scrape_content(
                session_id=session_id,
                window_id=window_id
            )
            
            response_text = response_content.content if hasattr(response_content, 'content') else str(response_content)
            
            # Extract AI response and citations
            ai_analysis = self._analyze_ai_response(response_text, brand_name, competitors, engine_config)
            
            response_time = (datetime.now() - start_time).total_seconds()
            
            result = {
                'platform': platform,
                'engine_name': engine_config['name'],
                'query': query,
                'ai_response': ai_analysis['ai_response'],
                'brand_mentioned': ai_analysis['brand_mentioned'],
                'mention_type': ai_analysis['mention_type'],
                'mention_context': ai_analysis['mention_context'],
                'competitors_mentioned': ai_analysis['competitors_mentioned'],
                'sources_cited': ai_analysis['sources_cited'],
                'citation_urls': ai_analysis['citation_urls'],
                'sentiment': ai_analysis['sentiment'],
                'response_time': response_time,
                'timestamp': datetime.now().isoformat(),
                'success': True,
                'raw_content': response_text[:1000] + "..." if len(response_text) > 1000 else response_text
            }
            
            logger.info(f"✅ Captured AI response from {engine_config['name']}")
            return result
            
        except Exception as e:
            logger.error(f"❌ Error capturing AI response from {platform}: {str(e)}")
            response_time = (datetime.now() - start_time).total_seconds()
            
            return {
                'platform': platform,
                'engine_name': engine_config['name'],
                'query': query,
                'ai_response': '',
                'brand_mentioned': False,
                'mention_type': 'error',
                'mention_context': '',
                'competitors_mentioned': [],
                'sources_cited': [],
                'citation_urls': [],
                'sentiment': 'unknown',
                'response_time': response_time,
                'timestamp': datetime.now().isoformat(),
                'success': False,
                'error': str(e)
            }
    
    def _analyze_ai_response(self, response_text: str, brand_name: str, 
                           competitors: List[str], engine_config: Dict) -> Dict:
        """Analyze AI search engine response for brand mentions, citations, and sentiment"""
        import re
        
        result = {
            'ai_response': '',
            'brand_mentioned': False,
            'mention_type': 'none',
            'mention_context': '',
            'competitors_mentioned': [],
            'sources_cited': [],
            'citation_urls': [],
            'sentiment': 'neutral'
        }
        
        if not response_text:
            return result
        
        # Extract main AI response (clean up HTML/formatting)
        clean_response = re.sub(r'<[^>]+>', ' ', response_text)
        clean_response = re.sub(r'\s+', ' ', clean_response).strip()
        result['ai_response'] = clean_response[:500] + "..." if len(clean_response) > 500 else clean_response
        
        # Check for brand mentions with context
        brand_pattern = re.compile(r'\b' + re.escape(brand_name) + r'\b', re.IGNORECASE)
        brand_matches = list(brand_pattern.finditer(clean_response))
        
        if brand_matches:
            result['brand_mentioned'] = True
            
            # Extract context around brand mention
            for match in brand_matches:
                start = max(0, match.start() - 100)
                end = min(len(clean_response), match.end() + 100)
                context = clean_response[start:end]
                result['mention_context'] = context
                break  # Use first mention for context
            
            # Determine mention type and sentiment
            context_lower = result['mention_context'].lower()
            
            if any(word in context_lower for word in ['best', 'top', 'leading', 'excellent', 'recommended', 'premier']):
                result['mention_type'] = 'positive'
                result['sentiment'] = 'positive'
            elif any(word in context_lower for word in ['worst', 'bad', 'poor', 'avoid', 'problem', 'issue']):
                result['mention_type'] = 'negative'
                result['sentiment'] = 'negative'
            elif any(word in context_lower for word in ['alternative', 'instead of', 'rather than']):
                result['mention_type'] = 'alternative'
            else:
                result['mention_type'] = 'mentioned'
        
        # Check for competitor mentions
        for competitor in competitors:
            if re.search(r'\b' + re.escape(competitor) + r'\b', clean_response, re.IGNORECASE):
                result['competitors_mentioned'].append(competitor)
        
        # Extract citations and sources (basic extraction)
        url_pattern = re.compile(r'https?://[^\s<>"]+')
        urls = url_pattern.findall(response_text)
        
        # Filter out common platform URLs
        filtered_urls = []
        for url in urls:
            if not any(platform in url.lower() for platform in ['google.com/search', 'chat.openai.com', 'perplexity.ai']):
                filtered_urls.append(url)
        
        result['citation_urls'] = list(set(filtered_urls))[:5]  # Limit to 5 unique URLs
        result['sources_cited'] = [self._extract_domain(url) for url in result['citation_urls']]
        
        return result
    
    def _extract_domain(self, url: str) -> str:
        """Extract domain from URL for source attribution"""
        try:
            from urllib.parse import urlparse
            domain = urlparse(url).netloc
            return domain.replace('www.', '') if domain else url
        except:
            return url
    
    def _generate_visibility_summary(self, results: List[Dict], brand_name: str) -> Dict:
        """Generate comprehensive visibility summary"""
        if not results:
            return {}
        
        successful_results = [r for r in results if r.get('success', False)]
        total_queries = len(results)
        successful_queries = len(successful_results)
        brand_mentions = len([r for r in successful_results if r.get('brand_mentioned', False)])
        
        # Sentiment analysis
        sentiments = [r.get('sentiment', 'neutral') for r in successful_results if r.get('brand_mentioned', False)]
        sentiment_counts = {s: sentiments.count(s) for s in ['positive', 'negative', 'neutral']}
        
        # Source analysis
        all_sources = []
        for r in successful_results:
            all_sources.extend(r.get('sources_cited', []))
        
        source_frequency = {}
        for source in all_sources:
            source_frequency[source] = source_frequency.get(source, 0) + 1
        
        top_sources = sorted(source_frequency.items(), key=lambda x: x[1], reverse=True)[:5]
        
        # AI engines tested
        engines_tested = list(set([r.get('engine_name', r.get('platform', 'unknown')) for r in results]))
        
        return {
            'brand_name': brand_name,
            'total_queries': total_queries,
            'successful_queries': successful_queries,
            'success_rate': round((successful_queries / total_queries * 100), 1) if total_queries > 0 else 0,
            'brand_mentions': brand_mentions,
            'brand_mention_rate': round((brand_mentions / total_queries * 100), 1) if total_queries > 0 else 0,
            'sentiment_breakdown': sentiment_counts,
            'top_citing_sources': dict(top_sources),
            'ai_engines_tested': engines_tested,
            'monitoring_timestamp': datetime.now().isoformat()
        }
    
    async def _cleanup_session(self, session_id: str):
        """Clean up browser session resources"""
        try:
            logger.info(f"🧹 Cleaning up browser session: {session_id}")
            await self.airtop_client.sessions.terminate(id=session_id)
            
            if session_id in self.active_sessions:
                del self.active_sessions[session_id]
                
            logger.info("✅ Browser session cleanup completed")
            
        except Exception as e:
            logger.error(f"❌ Error during session cleanup: {str(e)}")
    
    def _generate_demo_results(self, brand_name: str, competitors: List[str], 
                             queries: List[str], platforms: List[str]) -> Dict:
        """Generate demo results for when API is not available"""
        import random
        
        results = []
        
        for platform in platforms:
            if platform not in self.ai_search_engines:
                continue
                
            engine_config = self.ai_search_engines[platform]
            
            for query in queries:
                # Simulate realistic results
                brand_mentioned = random.choice([True, True, False])  # 67% chance
                sentiment = random.choice(["positive", "neutral", "negative"]) if brand_mentioned else "neutral"
                
                results.append({
                    'platform': platform,
                    'engine_name': engine_config['name'],
                    'query': query,
                    'ai_response': f"Demo response for '{query}' from {engine_config['name']}. {brand_name} {'is mentioned' if brand_mentioned else 'is not mentioned'} in this simulated response.",
                    'brand_mentioned': brand_mentioned,
                    'mention_type': 'mentioned' if brand_mentioned else 'none',
                    'mention_context': f"...relevant context about {brand_name}..." if brand_mentioned else '',
                    'competitors_mentioned': random.sample(competitors, min(len(competitors), random.randint(0, 2))),
                    'sources_cited': ['example.com', 'techcrunch.com', 'forbes.com'][:random.randint(1, 3)] if brand_mentioned else [],
                    'citation_urls': [f'https://example{i}.com' for i in range(random.randint(1, 3))] if brand_mentioned else [],
                    'sentiment': sentiment,
                    'response_time': random.uniform(5.0, 20.0),
                    'timestamp': datetime.now().isoformat(),
                    'success': True,
                    'raw_content': f"Demo content from {engine_config['name']}..."
                })
        
        return {
            'success': True,
            'demo_mode': True,
            'brand_monitored': brand_name,
            'competitors_tracked': competitors,
            'ai_engines_tested': platforms,
            'results': results,
            'summary': self._generate_visibility_summary(results, brand_name),
            'timestamp': datetime.now().isoformat()
        }

# Example usage for testing
async def main():
    """Example of monitoring brand visibility across AI search engines"""
    
    # Initialize the monitor
    monitor = AirtopLLMVisibility()
    
    # Run visibility check
    results = await monitor.run_visibility_check(
        brand_name="AIO Search",
        competitors=["Clearscope", "MarketMuse", "Surfer SEO"],
        queries=[
            "What are the best AI content optimization tools?",
            "AIO Search vs competitors comparison"
        ],
        platforms=["perplexity"]  # Start with Perplexity
    )
    
    if results['success']:
        print("🎉 AI Search Visibility Monitoring Complete!")
        print(f"📊 Summary: {results['summary']}")
        
        for result in results['results']:
            if result['success']:
                status = "✅" if result['brand_mentioned'] else "❌"
                print(f"{status} {result['engine_name']}: {result['query'][:50]}...")
                print(f"   📝 Brand Mentioned: {result['brand_mentioned']}")
                print(f"   🎭 Sentiment: {result['sentiment']}")
                print(f"   📚 Sources: {', '.join(result['sources_cited'][:3])}")
                print("---")
    else:
        print(f"❌ Monitoring failed: {results.get('error')}")

if __name__ == "__main__":
    asyncio.run(main()) 