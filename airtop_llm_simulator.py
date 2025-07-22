"""
Airtop LLM Simulator - Production System
========================================

Production-ready system for simulating real user interactions with LLM platforms
that have web search capabilities. Uses Airtop for authentic browser automation.

This system:
- Interacts with real LLM platforms (ChatGPT, Claude, Gemini, etc.)
- Performs actual web searches through these platforms
- Captures authentic responses including web search results
- Handles authentication and session management
- Scales for production use
"""

import asyncio
import json
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import aiohttp
import pandas as pd
from playwright.async_api import async_playwright, Browser, Page
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('airtop_simulator.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class LLMPlatform:
    """Configuration for an LLM platform"""
    name: str
    url: str
    login_required: bool
    web_search_enabled: bool
    selectors: Dict[str, str]
    rate_limit: int  # requests per minute
    session_timeout: int  # minutes

@dataclass
class SimulationResult:
    """Result from LLM platform simulation"""
    platform: str
    prompt: str
    response: str
    web_search_results: List[str]
    response_time: float
    timestamp: datetime
    success: bool
    error_message: Optional[str] = None

class AirtopLLMSimulator:
    """
    Production-ready LLM simulator using Airtop for authentic web interactions
    """
    
    def __init__(self):
        """Initialize the simulator with platform configurations"""
        self.platforms = self._configure_platforms()
        self.browser: Optional[Browser] = None
        self.pages: Dict[str, Page] = {}
        self.sessions: Dict[str, Dict] = {}
        self.rate_limiters: Dict[str, List[float]] = {}
        
    def _configure_platforms(self) -> Dict[str, LLMPlatform]:
        """Configure supported LLM platforms"""
        return {
            "chatgpt": LLMPlatform(
                name="ChatGPT",
                url="https://chat.openai.com/",
                login_required=True,
                web_search_enabled=True,
                selectors={
                    "input_box": "textarea[data-id='root']",
                    "send_button": "button[data-testid='send-button']",
                    "response_container": "[data-testid='conversation-turn-2']",
                    "web_search_toggle": "[data-testid='web-search-toggle']",
                    "login_button": "button[data-testid='login-button']",
                    "email_input": "input[name='username']",
                    "password_input": "input[name='password']",
                    "continue_button": "button[type='submit']"
                },
                rate_limit=20,  # 20 requests per minute
                session_timeout=60  # 60 minutes
            ),
            "claude": LLMPlatform(
                name="Claude",
                url="https://claude.ai/",
                login_required=True,
                web_search_enabled=True,
                selectors={
                    "input_box": "div[contenteditable='true']",
                    "send_button": "button[aria-label='Send message']",
                    "response_container": ".claude-response",
                    "web_search_toggle": "[data-testid='web-search']",
                    "login_button": "button[data-testid='login']",
                    "email_input": "input[type='email']",
                    "password_input": "input[type='password']",
                    "continue_button": "button[type='submit']"
                },
                rate_limit=15,  # 15 requests per minute
                session_timeout=60
            ),
            "gemini": LLMPlatform(
                name="Gemini",
                url="https://gemini.google.com/",
                login_required=True,
                web_search_enabled=True,
                selectors={
                    "input_box": "textarea[aria-label='Chat input']",
                    "send_button": "button[aria-label='Send message']",
                    "response_container": ".response-content",
                    "web_search_toggle": "[data-testid='web-search']",
                    "login_button": "button[data-testid='login']",
                    "email_input": "input[type='email']",
                    "password_input": "input[type='password']",
                    "continue_button": "button[type='submit']"
                },
                rate_limit=25,  # 25 requests per minute
                session_timeout=60
            ),
            "perplexity": LLMPlatform(
                name="Perplexity",
                url="https://www.perplexity.ai/",
                login_required=False,
                web_search_enabled=True,
                selectors={
                    "input_box": "textarea[placeholder*='Ask anything']",
                    "send_button": "button[aria-label='Send']",
                    "response_container": ".response-text",
                    "web_search_results": ".search-results",
                    "focus_button": "button[aria-label='Focus']"
                },
                rate_limit=30,  # 30 requests per minute
                session_timeout=30
            )
        }
    
    async def initialize(self):
        """Initialize browser and establish connections"""
        try:
            self.playwright = await async_playwright().start()
            
            # Launch browser with realistic settings
            self.browser = await self.playwright.chromium.launch(
                headless=False,  # Set to True for production
                args=[
                    '--no-sandbox',
                    '--disable-blink-features=AutomationControlled',
                    '--disable-dev-shm-usage',
                    '--disable-web-security',
                    '--disable-features=VizDisplayCompositor'
                ]
            )
            
            # Set up user agent and viewport
            context = await self.browser.new_context(
                viewport={'width': 1920, 'height': 1080},
                user_agent='Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
            )
            
            logger.info("Browser initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize browser: {str(e)}")
            raise
    
    async def authenticate_platform(self, platform_name: str, credentials: Dict[str, str]) -> bool:
        """
        Authenticate with an LLM platform
        
        Args:
            platform_name: Name of the platform (chatgpt, claude, etc.)
            credentials: Dict with 'email' and 'password'
            
        Returns:
            bool: True if authentication successful
        """
        try:
            platform = self.platforms[platform_name]
            if not platform.login_required:
                return True
                
            # Create new page for this platform
            page = await self.browser.new_page()
            await page.goto(platform.url)
            
            # Wait for page to load
            await page.wait_for_load_state('networkidle')
            
            # Handle login flow
            if platform_name == "chatgpt":
                success = await self._login_chatgpt(page, credentials, platform)
            elif platform_name == "claude":
                success = await self._login_claude(page, credentials, platform)
            elif platform_name == "gemini":
                success = await self._login_gemini(page, credentials, platform)
            else:
                success = True  # No login required
            
            if success:
                self.pages[platform_name] = page
                self.sessions[platform_name] = {
                    'authenticated_at': datetime.now(),
                    'last_activity': datetime.now()
                }
                logger.info(f"Successfully authenticated with {platform_name}")
                return True
            else:
                await page.close()
                return False
                
        except Exception as e:
            logger.error(f"Authentication failed for {platform_name}: {str(e)}")
            return False
    
    async def _login_chatgpt(self, page: Page, credentials: Dict[str, str], platform: LLMPlatform) -> bool:
        """Handle ChatGPT login flow"""
        try:
            # Click login button
            await page.click(platform.selectors["login_button"])
            await page.wait_for_timeout(2000)
            
            # Enter email
            await page.fill(platform.selectors["email_input"], credentials['email'])
            await page.click(platform.selectors["continue_button"])
            await page.wait_for_timeout(3000)
            
            # Enter password
            await page.fill(platform.selectors["password_input"], credentials['password'])
            await page.click(platform.selectors["continue_button"])
            
            # Wait for successful login
            await page.wait_for_timeout(5000)
            
            # Check if login was successful
            if await page.locator(platform.selectors["input_box"]).count() > 0:
                return True
            else:
                return False
                
        except Exception as e:
            logger.error(f"ChatGPT login error: {str(e)}")
            return False
    
    async def _login_claude(self, page: Page, credentials: Dict[str, str], platform: LLMPlatform) -> bool:
        """Handle Claude login flow"""
        try:
            # Click login button
            await page.click(platform.selectors["login_button"])
            await page.wait_for_timeout(2000)
            
            # Enter email
            await page.fill(platform.selectors["email_input"], credentials['email'])
            await page.click(platform.selectors["continue_button"])
            await page.wait_for_timeout(3000)
            
            # Enter password
            await page.fill(platform.selectors["password_input"], credentials['password'])
            await page.click(platform.selectors["continue_button"])
            
            # Wait for successful login
            await page.wait_for_timeout(5000)
            
            # Check if login was successful
            if await page.locator(platform.selectors["input_box"]).count() > 0:
                return True
            else:
                return False
                
        except Exception as e:
            logger.error(f"Claude login error: {str(e)}")
            return False
    
    async def _login_gemini(self, page: Page, credentials: Dict[str, str], platform: LLMPlatform) -> bool:
        """Handle Gemini login flow"""
        try:
            # Click login button
            await page.click(platform.selectors["login_button"])
            await page.wait_for_timeout(2000)
            
            # Enter email
            await page.fill(platform.selectors["email_input"], credentials['email'])
            await page.click(platform.selectors["continue_button"])
            await page.wait_for_timeout(3000)
            
            # Enter password
            await page.fill(platform.selectors["password_input"], credentials['password'])
            await page.click(platform.selectors["continue_button"])
            
            # Wait for successful login
            await page.wait_for_timeout(5000)
            
            # Check if login was successful
            if await page.locator(platform.selectors["input_box"]).count() > 0:
                return True
            else:
                return False
                
        except Exception as e:
            logger.error(f"Gemini login error: {str(e)}")
            return False
    
    async def check_rate_limit(self, platform_name: str) -> bool:
        """Check if we're within rate limits for a platform"""
        if platform_name not in self.rate_limiters:
            self.rate_limiters[platform_name] = []
        
        platform = self.platforms[platform_name]
        current_time = time.time()
        
        # Remove requests older than 1 minute
        self.rate_limiters[platform_name] = [
            req_time for req_time in self.rate_limiters[platform_name]
            if current_time - req_time < 60
        ]
        
        # Check if we're under the limit
        if len(self.rate_limiters[platform_name]) < platform.rate_limit:
            self.rate_limiters[platform_name].append(current_time)
            return True
        else:
            return False
    
    async def simulate_query(self, platform_name: str, prompt: str, enable_web_search: bool = True) -> SimulationResult:
        """
        Simulate a real user query on an LLM platform
        
        Args:
            platform_name: Name of the platform
            prompt: The query to send
            enable_web_search: Whether to enable web search
            
        Returns:
            SimulationResult: Complete result from the interaction
        """
        start_time = time.time()
        
        try:
            # Check rate limits
            if not await self.check_rate_limit(platform_name):
                return SimulationResult(
                    platform=platform_name,
                    prompt=prompt,
                    response="",
                    web_search_results=[],
                    response_time=0,
                    timestamp=datetime.now(),
                    success=False,
                    error_message="Rate limit exceeded"
                )
            
            platform = self.platforms[platform_name]
            page = self.pages.get(platform_name)
            
            if not page:
                return SimulationResult(
                    platform=platform_name,
                    prompt=prompt,
                    response="",
                    web_search_results=[],
                    response_time=0,
                    timestamp=datetime.now(),
                    success=False,
                    error_message="Platform not authenticated"
                )
            
            # Navigate to platform if needed
            if page.url != platform.url:
                await page.goto(platform.url)
                await page.wait_for_load_state('networkidle')
            
            # Enable web search if requested and available
            if enable_web_search and platform.web_search_enabled:
                await self._enable_web_search(page, platform)
            
            # Type the prompt
            await page.click(platform.selectors["input_box"])
            await page.fill(platform.selectors["input_box"], prompt)
            await page.wait_for_timeout(1000)
            
            # Send the message
            await page.click(platform.selectors["send_button"])
            
            # Wait for response
            response_text = await self._wait_for_response(page, platform)
            
            # Extract web search results if available
            web_search_results = await self._extract_web_search_results(page, platform)
            
            response_time = time.time() - start_time
            
            # Update session activity
            if platform_name in self.sessions:
                self.sessions[platform_name]['last_activity'] = datetime.now()
            
            return SimulationResult(
                platform=platform_name,
                prompt=prompt,
                response=response_text,
                web_search_results=web_search_results,
                response_time=response_time,
                timestamp=datetime.now(),
                success=True
            )
            
        except Exception as e:
            logger.error(f"Query simulation failed for {platform_name}: {str(e)}")
            return SimulationResult(
                platform=platform_name,
                prompt=prompt,
                response="",
                web_search_results=[],
                response_time=time.time() - start_time,
                timestamp=datetime.now(),
                success=False,
                error_message=str(e)
            )
    
    async def _enable_web_search(self, page: Page, platform: LLMPlatform):
        """Enable web search on the platform"""
        try:
            # Look for web search toggle
            web_search_selector = platform.selectors.get("web_search_toggle")
            if web_search_selector:
                web_search_element = page.locator(web_search_selector)
                if await web_search_element.count() > 0:
                    await web_search_element.click()
                    await page.wait_for_timeout(2000)
                    logger.info(f"Enabled web search on {platform.name}")
        except Exception as e:
            logger.warning(f"Could not enable web search on {platform.name}: {str(e)}")
    
    async def _wait_for_response(self, page: Page, platform: LLMPlatform) -> str:
        """Wait for and extract the LLM response"""
        try:
            # Wait for response to start appearing
            await page.wait_for_timeout(3000)
            
            # Wait for response to complete (look for stop button to disappear)
            max_wait_time = 120  # 2 minutes max
            start_time = time.time()
            
            while time.time() - start_time < max_wait_time:
                # Check if response is still being generated
                stop_button = page.locator("button[aria-label*='Stop']")
                if await stop_button.count() == 0:
                    break
                await page.wait_for_timeout(2000)
            
            # Extract the response text
            response_selector = platform.selectors["response_container"]
            response_element = page.locator(response_selector).last
            
            if await response_element.count() > 0:
                response_text = await response_element.text_content()
                return response_text.strip()
            else:
                return "No response received"
                
        except Exception as e:
            logger.error(f"Error waiting for response: {str(e)}")
            return "Error extracting response"
    
    async def _extract_web_search_results(self, page: Page, platform: LLMPlatform) -> List[str]:
        """Extract web search results from the response"""
        try:
            web_results = []
            
            # Look for web search result indicators
            web_search_selectors = [
                "[data-testid*='web-search']",
                ".search-result",
                ".web-result",
                "[class*='search']",
                "[class*='web']"
            ]
            
            for selector in web_search_selectors:
                elements = page.locator(selector)
                count = await elements.count()
                
                for i in range(count):
                    element = elements.nth(i)
                    text = await element.text_content()
                    if text and len(text.strip()) > 10:
                        web_results.append(text.strip())
            
            return web_results
            
        except Exception as e:
            logger.error(f"Error extracting web search results: {str(e)}")
            return []
    
    async def check_brand_visibility(self, brand_name: str, competitors: List[str], 
                                   prompts: List[str], platforms: List[str] = None) -> pd.DataFrame:
        """
        Check brand visibility across multiple LLM platforms
        
        Args:
            brand_name: Brand to check
            competitors: List of competitor names
            prompts: List of prompts to test
            platforms: List of platforms to test (default: all)
            
        Returns:
            DataFrame with visibility results
        """
        if platforms is None:
            platforms = list(self.platforms.keys())
        
        results = []
        
        for platform in platforms:
            if platform not in self.pages:
                logger.warning(f"Platform {platform} not authenticated, skipping")
                continue
            
            for prompt in prompts:
                # Simulate the query
                result = await self.simulate_query(platform, prompt, enable_web_search=True)
                
                if result.success:
                    # Analyze brand mentions
                    mention_analysis = self._analyze_brand_mentions(
                        result.response, brand_name, competitors
                    )
                    
                    results.append({
                        'platform': platform,
                        'prompt': prompt,
                        'response': result.response,
                        'web_search_results': result.web_search_results,
                        'brand_mentioned': mention_analysis['brand_mentioned'],
                        'mention_type': mention_analysis['mention_type'],
                        'competitors_mentioned': mention_analysis['competitors_mentioned'],
                        'response_time': result.response_time,
                        'timestamp': result.timestamp,
                        'success': True
                    })
                else:
                    results.append({
                        'platform': platform,
                        'prompt': prompt,
                        'response': '',
                        'web_search_results': [],
                        'brand_mentioned': False,
                        'mention_type': 'none',
                        'competitors_mentioned': [],
                        'response_time': result.response_time,
                        'timestamp': result.timestamp,
                        'success': False,
                        'error': result.error_message
                    })
                
                # Rate limiting delay
                await asyncio.sleep(3)
        
        return pd.DataFrame(results)
    
    def _analyze_brand_mentions(self, response: str, brand_name: str, competitors: List[str]) -> Dict:
        """Analyze brand and competitor mentions in response"""
        import re
        
        result = {
            'brand_mentioned': False,
            'mention_type': 'none',
            'competitors_mentioned': []
        }
        
        # Check for brand mentions
        if re.search(r'\b' + re.escape(brand_name) + r'\b', response, re.IGNORECASE):
            result['brand_mentioned'] = True
            
            # Determine mention type
            if re.search(r'\b(best|top|recommended|leading|excellent|outstanding|premier|first choice|preferred)\b.{0,50}\b' + 
                         re.escape(brand_name) + r'\b', response, re.IGNORECASE):
                result['mention_type'] = 'direct_positive'
            elif re.search(r'alternatives to.{0,30}\b' + re.escape(brand_name) + r'\b', response, re.IGNORECASE):
                result['mention_type'] = 'direct_reference'
            else:
                result['mention_type'] = 'mentioned'
        
        # Check for competitor mentions
        for competitor in competitors:
            if re.search(r'\b' + re.escape(competitor) + r'\b', response, re.IGNORECASE):
                result['competitors_mentioned'].append(competitor)
        
        return result
    
    async def cleanup(self):
        """Clean up resources"""
        try:
            # Close all pages
            for page in self.pages.values():
                await page.close()
            
            # Close browser
            if self.browser:
                await self.browser.close()
            
            # Stop playwright
            if hasattr(self, 'playwright'):
                await self.playwright.stop()
                
            logger.info("Cleanup completed")
            
        except Exception as e:
            logger.error(f"Error during cleanup: {str(e)}")

# Example usage
async def main():
    """Example usage of the Airtop LLM Simulator"""
    
    # Initialize simulator
    simulator = AirtopLLMSimulator()
    await simulator.initialize()
    
    # Authenticate with platforms (you'll need real credentials)
    credentials = {
        'email': os.getenv('LLM_EMAIL'),
        'password': os.getenv('LLM_PASSWORD')
    }
    
    # Authenticate with each platform
    for platform in ['chatgpt', 'claude', 'gemini']:
        success = await simulator.authenticate_platform(platform, credentials)
        if success:
            print(f"✅ Authenticated with {platform}")
        else:
            print(f"❌ Failed to authenticate with {platform}")
    
    # Test prompts
    test_prompts = [
        "What are the best AI content optimization tools?",
        "Compare Clearscope vs MarketMuse for SEO",
        "What AI tools help with content marketing?"
    ]
    
    # Check brand visibility
    results = await simulator.check_brand_visibility(
        brand_name="AIO Search",
        competitors=["Clearscope", "MarketMuse", "Surfer SEO"],
        prompts=test_prompts,
        platforms=['perplexity']  # Start with Perplexity (no login required)
    )
    
    # Save results
    results.to_csv('brand_visibility_results.csv', index=False)
    print(f"Results saved to brand_visibility_results.csv")
    
    # Cleanup
    await simulator.cleanup()

if __name__ == "__main__":
    asyncio.run(main()) 