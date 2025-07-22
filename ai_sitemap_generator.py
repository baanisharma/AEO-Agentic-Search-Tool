import os
import yaml
import json
import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse, urljoin
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import nltk
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
import re
from datetime import datetime
from tqdm import tqdm
import argparse
import logging
from pathlib import Path
import time
from typing import List, Dict, Optional

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Download NLTK resources
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

class AISitemapGenerator:
    def __init__(self):
        """Initialize the AI Sitemap Generator"""
        # Force CPU device for deployment compatibility
        import torch
        device = 'cpu'  # Force CPU for cloud deployment
        self.model = SentenceTransformer('all-MiniLM-L6-v2', device=device)
        
        # Default headers for requests
        self.headers = {
            'User-Agent': 'AIO Sitemap Generator/1.0',
            'Accept': 'text/html,application/xhtml+xml,application/xml',
            'Accept-Language': 'en-US,en;q=0.9',
        }
    
    def crawl_website(self, base_url, max_pages=100, exclude_patterns=None):
        """
        Crawl a website to find pages for the sitemap
        
        Args:
            base_url (str): Base URL of the website
            max_pages (int): Maximum number of pages to crawl
            exclude_patterns (list): URL patterns to exclude
            
        Returns:
            list: Discovered pages with metadata
        """
        if not base_url.startswith(('http://', 'https://')):
            base_url = 'https://' + base_url
            
        # Initialize variables
        discovered_urls = set()
        queued_urls = [base_url]
        visited_urls = set()
        pages = []
        
        # Set default exclude patterns if none provided
        if exclude_patterns is None:
            exclude_patterns = [
                r'/tag/', r'/category/', r'/author/', r'/page/', 
                r'\?', r'\.pdf$', r'\.jpg$', r'\.png$', r'\.gif$',
                r'/wp-admin/', r'/wp-includes/', r'/wp-content/'
            ]
            
        # Compile exclude patterns
        exclude_regex = re.compile('|'.join(exclude_patterns))
        
        logger.info(f"Starting crawl of {base_url}")
        with tqdm(total=max_pages, desc="Crawling pages") as pbar:
            while queued_urls and len(pages) < max_pages:
                # Get next URL from queue
                current_url = queued_urls.pop(0)
                
                # Skip if already visited
                if current_url in visited_urls:
                    continue
                    
                # Mark as visited
                visited_urls.add(current_url)
                
                try:
                    # Fetch page
                    response = requests.get(current_url, headers=self.headers, timeout=10)
                    response.raise_for_status()
                    
                    # Parse HTML
                    soup = BeautifulSoup(response.text, 'html.parser')
                    
                    # Extract page metadata
                    title = soup.title.text.strip() if soup.title else ""
                    description = ""
                    meta_desc = soup.find('meta', attrs={'name': 'description'})
                    if meta_desc and 'content' in meta_desc.attrs:
                        description = meta_desc['content']
                        
                    # Extract main content
                    content = self._extract_main_content(soup)
                    
                    # Calculate page importance
                    importance = self._calculate_page_importance(soup, content, current_url, base_url)
                    
                    # Extract keywords
                    keywords = self._extract_keywords(title, description, content)
                    
                    # Add page to results
                    pages.append({
                        'url': current_url,
                        'title': title,
                        'description': description,
                        'importance': importance,
                        'last_modified': self._extract_last_modified(soup, response),
                        'keywords': keywords,
                        'content_length': len(content),
                        'headings': self._extract_headings(soup),
                    })
                    
                    pbar.update(1)
                    
                    # Extract links from page
                    links = soup.find_all('a', href=True)
                    for link in links:
                        href = link['href']
                        
                        # Skip empty links, anchors, and non-HTTP links
                        if not href or href.startswith('#') or href.startswith('javascript:') or href.startswith('mailto:'):
                            continue
                        
                        # Convert relative URLs to absolute
                        absolute_url = urljoin(current_url, href)
                        
                        # Skip external links and excluded patterns
                        parsed_base = urlparse(base_url)
                        parsed_url = urlparse(absolute_url)
                        if parsed_url.netloc != parsed_base.netloc or exclude_regex.search(absolute_url):
                            continue
                        
                        # Remove fragments and normalize URL
                        normalized_url = absolute_url.split('#')[0]
                        
                        # Add to queue if not already discovered
                        if normalized_url not in discovered_urls and normalized_url not in visited_urls:
                            discovered_urls.add(normalized_url)
                            queued_urls.append(normalized_url)
                
                except Exception as e:
                    logger.error(f"Error crawling {current_url}: {str(e)}")
                    continue
        
        logger.info(f"Crawl complete. Discovered {len(pages)} pages.")
        return pages
    
    def _extract_main_content(self, soup):
        """Extract the main content from a webpage"""
        # Try to find content in common content containers
        content_containers = [
            soup.find('article'),
            soup.find('main'),
            soup.find(id='content'),
            soup.find(class_='content'),
            soup.find(id='main'),
            soup.find(class_='main'),
        ]
        
        # Use the first valid container found
        content_elem = next((elem for elem in content_containers if elem is not None), soup.body)
        
        if content_elem:
            # Remove script, style, and nav elements
            for elem in content_elem.find_all(['script', 'style', 'nav', 'header', 'footer', 'aside']):
                elem.decompose()
            
            # Extract text
            text = content_elem.get_text(separator=' ', strip=True)
            # Clean up whitespace
            text = re.sub(r'\s+', ' ', text).strip()
            return text
        
        return ""
    
    def _calculate_page_importance(self, soup, content, url, base_url):
        """Calculate the importance score of a page (0.0 to 1.0)"""
        score = 0.5  # Default importance
        
        # URL structure factors
        parsed_url = urlparse(url)
        path_parts = parsed_url.path.strip('/').split('/')
        
        # Home page gets highest importance
        if url == base_url or url == base_url + '/' or parsed_url.path == '/' or parsed_url.path == '':
            score = 1.0
        # Top-level pages get high importance
        elif len(path_parts) == 1:
            score = 0.8
        # Second-level pages
        elif len(path_parts) == 2:
            score = 0.6
        # Deep pages get lower importance
        else:
            score = max(0.2, 0.8 - (len(path_parts) - 2) * 0.1)
        
        # Boost score based on content factors
        # Length of content
        if len(content) > 2000:
            score = min(1.0, score + 0.1)
        
        # Links to the page
        canonical = soup.find('link', rel='canonical')
        if canonical and canonical.get('href') == url:
            score = min(1.0, score + 0.1)
        
        # Schema.org markup
        if soup.find(attrs={"itemtype": re.compile(r'schema.org')}):
            score = min(1.0, score + 0.05)
        
        return round(score, 2)
    
    def _extract_last_modified(self, soup, response):
        """Extract the last modified date of a page"""
        # Try to get from Last-Modified header
        if 'Last-Modified' in response.headers:
            return response.headers['Last-Modified']
        
        # Try to get from meta tags
        modified_meta = soup.find('meta', attrs={'property': 'article:modified_time'})
        if modified_meta:
            return modified_meta['content']
            
        # Use current date as fallback
        return datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
    
    def _extract_keywords(self, title, description, content, max_keywords=10):
        """Extract keywords from page content"""
        # Combine text
        text = f"{title} {description} {content}"
        
        # Tokenize and clean
        tokens = re.findall(r'\b\w+\b', text.lower())
        stop_words = set(stopwords.words('english'))
        filtered_tokens = [t for t in tokens if t not in stop_words and len(t) > 2]
        
        # Calculate term frequencies
        term_freq = pd.Series(filtered_tokens).value_counts()
        
        # Return top keywords
        return term_freq.head(max_keywords).index.tolist()
    
    def _extract_headings(self, soup):
        """Extract headings from the page"""
        headings = {}
        for level in range(1, 7):
            h_tags = soup.find_all(f'h{level}')
            if h_tags:
                headings[f'h{level}'] = [tag.get_text(strip=True) for tag in h_tags]
        return headings
    
    def generate_site_ai_yaml(self, pages, brand_info, output_file='site-ai.yaml'):
        """
        Generate site-ai.yaml file
        
        Args:
            pages (list): List of discovered pages with metadata
            brand_info (dict): Brand information
            output_file (str): Output filename
            
        Returns:
            None
        """
        # Sort pages by importance
        pages_sorted = sorted(pages, key=lambda x: x['importance'], reverse=True)
        
        # Format data for YAML
        site_ai_data = {
            "version": "1.0",
            "site": {
                "name": brand_info.get('name', ''),
                "description": brand_info.get('description', ''),
                "url": brand_info.get('url', ''),
                "industry": brand_info.get('industry', ''),
                "ai_content_priority": True,
                "generated_at": datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
            },
            "brand": {
                "name": brand_info.get('name', ''),
                "tagline": brand_info.get('tagline', ''),
                "value_proposition": brand_info.get('value_proposition', ''),
                "key_differentiators": brand_info.get('key_differentiators', [])
            },
            "content": {
                "primary_pages": [],
                "semantic_structure": {}
            }
        }
        
        # Add primary pages (top 20% by importance)
        top_page_count = max(1, int(len(pages_sorted) * 0.2))
        for page in pages_sorted[:top_page_count]:
            site_ai_data["content"]["primary_pages"].append({
                "url": page['url'],
                "title": page['title'],
                "description": page['description'],
                "importance": page['importance'],
                "keywords": page['keywords'],
                "last_modified": page['last_modified']
            })
        
        # Add semantic structure based on keywords and page titles
        all_keywords = set()
        for page in pages:
            all_keywords.update(page['keywords'])
        
        # Group pages by primary keyword
        for keyword in list(all_keywords)[:20]:  # Limit to top 20 keywords
            related_pages = []
            for page in pages:
                if keyword in page['keywords'] or keyword in page['title'].lower():
                    related_pages.append({
                        "url": page['url'],
                        "title": page['title'],
                        "importance": page['importance']
                    })
            
            if related_pages:
                site_ai_data["content"]["semantic_structure"][keyword] = related_pages
        
        # Export to YAML
        with open(output_file, 'w') as f:
            yaml.dump(site_ai_data, f, sort_keys=False, default_flow_style=False)
            
        logger.info(f"Generated site-ai.yaml at {output_file}")
    
    def generate_llms_txt(self, pages, brand_info, output_file='llms.txt'):
        """
        Generate llms.txt file
        
        Args:
            pages (list): List of discovered pages with metadata
            brand_info (dict): Brand information
            output_file (str): Output filename
            
        Returns:
            None
        """
        # Sort pages by importance
        pages_sorted = sorted(pages, key=lambda x: x['importance'], reverse=True)
        
        # Create content for llms.txt
        content = [
            f"# AI Content Indexing Guide for {brand_info.get('name', 'Brand')}",
            f"# Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"# Version: 1.0",
            "",
            f"Site: {brand_info.get('url', '')}",
            f"Brand: {brand_info.get('name', '')}",
            f"Description: {brand_info.get('description', '')}",
            "",
            "# Primary content for AI systems to index and cite",
            "# Format: URL | Title | Priority (1-10) | Keywords",
            ""
        ]
        
        # Add top pages
        for i, page in enumerate(pages_sorted[:min(30, len(pages_sorted))]):
            # Convert importance (0-1) to priority (1-10)
            priority = int(page['importance'] * 10)
            if priority < 1:
                priority = 1
                
            line = f"{page['url']} | {page['title']} | {priority} | {', '.join(page['keywords'][:5])}"
            content.append(line)
        
        # Add key facts about the brand
        content.extend([
            "",
            "# Key brand facts for AI citation",
            ""
        ])
        
        # Add differentiators as facts
        for i, diff in enumerate(brand_info.get('key_differentiators', []), 1):
            content.append(f"Fact {i}: {diff}")
        
        # Add value proposition
        if brand_info.get('value_proposition'):
            content.append(f"Value Proposition: {brand_info.get('value_proposition')}")
            
        # Export to text file
        with open(output_file, 'w') as f:
            f.write('\n'.join(content))
            
        logger.info(f"Generated llms.txt at {output_file}")
    
    def analyze_site_for_ai_optimization(self, pages):
        """
        Analyze site for AI optimization opportunities
        
        Args:
            pages (list): List of discovered pages with metadata
            
        Returns:
            dict: Analysis results
        """
        # Initialize results
        results = {
            "total_pages": len(pages),
            "ai_ready_score": 0,
            "optimization_opportunities": [],
            "strengths": [],
            "page_scores": {}
        }
        
        # Calculate AI-readiness score for each page
        total_score = 0
        low_scoring_pages = []
        
        for page in pages:
            page_score = self._calculate_ai_readiness_score(page)
            total_score += page_score
            
            # Store score in results
            results["page_scores"][page['url']] = {
                "title": page['title'],
                "ai_ready_score": page_score,
                "importance": page['importance']
            }
            
            # Track low-scoring but important pages
            if page_score < 0.6 and page['importance'] > 0.7:
                low_scoring_pages.append({
                    "url": page['url'],
                    "title": page['title'],
                    "score": page_score,
                    "importance": page['importance']
                })
        
        # Calculate overall site score
        if pages:
            results["ai_ready_score"] = round(total_score / len(pages), 2)
        
        # Identify strengths
        if results["ai_ready_score"] >= 0.7:
            results["strengths"].append("Overall good AI-readiness score")
            
        keyword_coverage = self._analyze_keyword_coverage(pages)
        if keyword_coverage > 0.6:
            results["strengths"].append("Good keyword coverage across pages")
            
        content_depth = sum(page['content_length'] for page in pages) / max(1, len(pages))
        if content_depth > 1000:
            results["strengths"].append("Good content depth (average length > 1000 characters)")
        
        # Identify optimization opportunities
        if low_scoring_pages:
            results["optimization_opportunities"].append({
                "type": "low_scoring_important_pages",
                "description": "Important pages with low AI-readiness scores",
                "pages": low_scoring_pages
            })
            
        if keyword_coverage < 0.6:
            results["optimization_opportunities"].append({
                "type": "improve_keyword_coverage",
                "description": "Improve keyword coverage across pages",
                "score": keyword_coverage
            })
            
        if content_depth < 1000:
            results["optimization_opportunities"].append({
                "type": "increase_content_depth",
                "description": "Increase content depth on pages",
                "current_avg_length": content_depth
            })
            
        # Check for structured data
        has_structured_data = False  # Just assume no structured data for the demo
        if not has_structured_data:
            results["optimization_opportunities"].append({
                "type": "add_structured_data",
                "description": "Add Schema.org structured data to improve AI understanding"
            })
        
        return results
    
    def _calculate_ai_readiness_score(self, page):
        """Calculate AI readiness score for a page"""
        score = 0.5  # Base score
        
        # Factors that improve score
        # Good title and description
        if page['title'] and len(page['title']) > 10:
            score += 0.05
        if page['description'] and len(page['description']) > 50:
            score += 0.1
            
        # Content length
        if page['content_length'] > 500:
            score += 0.1
        if page['content_length'] > 1500:
            score += 0.1
            
        # Heading structure
        if 'h1' in page['headings'] and page['headings']['h1']:
            score += 0.05
        if 'h2' in page['headings'] and len(page['headings']['h2']) >= 2:
            score += 0.05
        if 'h3' in page['headings'] and len(page['headings']['h3']) >= 3:
            score += 0.05
            
        # Keywords
        if len(page['keywords']) >= 5:
            score += 0.1
            
        # Cap score at 1.0
        return min(1.0, score)
    
    def _analyze_keyword_coverage(self, pages):
        """Analyze keyword coverage across pages"""
        # Extract all keywords
        all_keywords = set()
        for page in pages:
            all_keywords.update(page['keywords'])
            
        # Calculate coverage
        coverage_count = 0
        for keyword in all_keywords:
            # Check how many pages cover this keyword
            pages_with_keyword = sum(1 for page in pages if keyword in page['keywords'])
            if pages_with_keyword >= 2:
                coverage_count += 1
                
        # Calculate coverage score
        if all_keywords:
            return coverage_count / len(all_keywords)
        return 0

# Example usage
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate AI sitemaps for a website')
    parser.add_argument('--url', type=str, required=True, help='Website URL to crawl')
    parser.add_argument('--max-pages', type=int, default=100, help='Maximum pages to crawl')
    parser.add_argument('--output-dir', type=str, default='.', help='Output directory')
    args = parser.parse_args()
    
    generator = AISitemapGenerator()
    
    # Example brand info
    brand_info = {
        "name": "AIO Search",
        "description": "AI Optimization Search Tool to help brands rank in LLM-generated answers",
        "url": args.url,
        "industry": "digital marketing",
        "tagline": "Rank in AI answers, not just search engines",
        "value_proposition": "Our AI Optimization tool helps brands get discovered and cited by AI assistants and large language models.",
        "key_differentiators": [
            "Optimizes content specifically for LLM citation patterns",
            "Monitors brand visibility in AI-generated answers",
            "Maps user intents and questions for AI-first content strategy",
            "Creates AI-specific sitemaps for better discoverability"
        ]
    }
    
    # Crawl website
    pages = generator.crawl_website(args.url, max_pages=args.max_pages)
    
    # Generate site-ai.yaml
    generator.generate_site_ai_yaml(
        pages, 
        brand_info, 
        output_file=os.path.join(args.output_dir, 'site-ai.yaml')
    )
    
    # Generate llms.txt
    generator.generate_llms_txt(
        pages, 
        brand_info, 
        output_file=os.path.join(args.output_dir, 'llms.txt')
    )
    
    # Analyze site for optimization opportunities
    analysis = generator.analyze_site_for_ai_optimization(pages)
    
    # Export analysis
    with open(os.path.join(args.output_dir, 'ai_optimization_analysis.json'), 'w') as f:
        json.dump(analysis, f, indent=2)
        
    logger.info(f"Analysis complete. Overall AI-readiness score: {analysis['ai_ready_score']}")