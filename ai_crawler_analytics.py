"""
AI Crawler Analytics - Website AI Bot Tracking
==============================================

This module tracks how AI crawlers (ChatGPT bot, Perplexity bot, Google AI, etc.) 
visit and index your website. It analyzes which pages are being accessed and provides 
recommendations to optimize your site for better AI visibility and citation.

🎯 GOAL: Monitor AI bot behavior on your website to understand what content 
AI systems are indexing and how to optimize for better citation.

🔧 HOW WE DO THIS:
• Parse server logs to detect AI crawler visits
• Track which pages AI bots access most frequently  
• Analyze AI bot crawl patterns and behavior
• Identify pages that AI bots find valuable
• Recommend optimizations for better AI accessibility

✅ WHY THIS MATTERS:
• AI systems cite content they've recently crawled
• Understanding bot behavior helps optimize content strategy
• Detect if AI bots are missing important pages
• Improve site structure for better AI discoverability

🚀 RESULT: Complete AI crawler intelligence for better AI visibility
"""

import os
import re
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from collections import defaultdict, Counter
import logging
from pathlib import Path
import geoip2.database
import geoip2.errors

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AICrawlerAnalytics:
    """
    AI Crawler Analytics - Track AI Bot Website Visits
    
    Monitors how ChatGPT, Perplexity, Google AI, and other AI crawlers
    interact with your website to optimize for better AI citation.
    """
    
    def __init__(self):
        """Initialize AI Crawler Analytics"""
        self.ai_crawler_patterns = self._get_ai_crawler_signatures()
        self.analysis_results = {}
        
    def _get_ai_crawler_signatures(self) -> Dict:
        """Get known AI crawler user agents and IP patterns"""
        return {
            "chatgpt": {
                "name": "ChatGPT Bot",
                "user_agents": [
                    r"ChatGPT-User",
                    r"CCBot.*ChatGPT",
                    r"GPTBot",
                    r"OpenAI.*",
                ],
                "ips": ["20.15.240.", "20.171."],  # OpenAI IP ranges
                "purpose": "Content indexing for ChatGPT responses"
            },
            "perplexity": {
                "name": "Perplexity Bot", 
                "user_agents": [
                    r"PerplexityBot",
                    r"Perplexity.*",
                ],
                "ips": ["35.89.", "54.187."],  # Perplexity IP ranges
                "purpose": "Real-time content crawling for search responses"
            },
            "google_ai": {
                "name": "Google AI",
                "user_agents": [
                    r"Googlebot.*AI",
                    r"Google-Extended",
                    r"GoogleOther.*AI",
                ],
                "ips": ["66.249.", "64.233."],  # Google IP ranges
                "purpose": "Content indexing for AI Overviews"
            },
            "microsoft_copilot": {
                "name": "Microsoft Copilot",
                "user_agents": [
                    r"Microsoft.*Copilot",
                    r"BingBot.*AI",
                    r"msnbot.*copilot",
                ],
                "ips": ["40.77.", "207.46."],  # Microsoft IP ranges
                "purpose": "Content indexing for Copilot responses"
            },
            "claude": {
                "name": "Claude Bot",
                "user_agents": [
                    r"Claude.*Bot",
                    r"Anthropic.*",
                ],
                "ips": ["54.230.", "52.85."],  # Anthropic/Claude IP ranges
                "purpose": "Content analysis for Claude responses"
            },
            "generic_ai": {
                "name": "Other AI Crawlers",
                "user_agents": [
                    r".*AI.*Bot",
                    r".*LLM.*",
                    r".*Assistant.*Bot",
                    r".*Crawler.*AI",
                ],
                "ips": [],
                "purpose": "Various AI systems content indexing"
            }
        }
    
    def parse_access_logs(self, log_file_path: str, days_back: int = 30) -> Dict:
        """
        Parse web server access logs to identify AI crawler visits
        
        Args:
            log_file_path: Path to access log file (Apache/Nginx format)
            days_back: Number of days to analyze
            
        Returns:
            Dict: AI crawler analysis results
        """
        logger.info(f"📊 Analyzing access logs for AI crawler activity...")
        
        if not os.path.exists(log_file_path):
            logger.error(f"❌ Log file not found: {log_file_path}")
            return {"error": "Log file not found"}
        
        # Initialize results
        results = {
            "analysis_period": f"{days_back} days",
            "total_requests": 0,
            "ai_requests": 0,
            "crawlers_detected": {},
            "top_crawled_pages": {},
            "crawl_patterns": {},
            "recommendations": [],
            "timestamp": datetime.now().isoformat()
        }
        
        # Date threshold for analysis
        cutoff_date = datetime.now() - timedelta(days=days_back)
        
        # Track AI crawler activity
        ai_requests = []
        all_requests = []
        
        try:
            with open(log_file_path, 'r', encoding='utf-8', errors='ignore') as f:
                for line_num, line in enumerate(f):
                    if line_num % 10000 == 0:
                        logger.info(f"📈 Processed {line_num} log entries...")
                    
                    # Parse log entry
                    parsed = self._parse_log_entry(line)
                    if not parsed or not self._is_within_date_range(parsed['timestamp'], cutoff_date):
                        continue
                    
                    all_requests.append(parsed)
                    
                    # Check if this is an AI crawler
                    crawler_info = self._identify_ai_crawler(parsed['user_agent'], parsed['ip'])
                    if crawler_info:
                        parsed['crawler_type'] = crawler_info['type']
                        parsed['crawler_name'] = crawler_info['name']
                        ai_requests.append(parsed)
            
            logger.info(f"✅ Processed {len(all_requests)} total requests, {len(ai_requests)} AI crawler requests")
            
        except Exception as e:
            logger.error(f"❌ Error reading log file: {str(e)}")
            return {"error": str(e)}
        
        # Analyze results
        results["total_requests"] = len(all_requests)
        results["ai_requests"] = len(ai_requests)
        results["ai_request_percentage"] = round((len(ai_requests) / max(1, len(all_requests))) * 100, 2)
        
        # Analyze by crawler type
        results["crawlers_detected"] = self._analyze_crawler_types(ai_requests)
        
        # Analyze most crawled pages
        results["top_crawled_pages"] = self._analyze_crawled_pages(ai_requests)
        
        # Analyze crawl patterns
        results["crawl_patterns"] = self._analyze_crawl_patterns(ai_requests)
        
        # Generate recommendations
        results["recommendations"] = self._generate_crawler_recommendations(results)
        
        self.analysis_results = results
        return results
    
    def _parse_log_entry(self, line: str) -> Optional[Dict]:
        """Parse a single access log entry (Common Log Format)"""
        # Common Log Format: IP - - [timestamp] "method url protocol" status size "referer" "user-agent"
        pattern = r'(\S+) \S+ \S+ \[([^\]]+)\] "(\S+) (\S+) (\S+)" (\d+) (\S+) "([^"]*)" "([^"]*)"'
        match = re.match(pattern, line)
        
        if not match:
            return None
        
        try:
            ip, timestamp_str, method, url, protocol, status, size, referer, user_agent = match.groups()
            
            # Parse timestamp
            timestamp = datetime.strptime(timestamp_str, '%d/%b/%Y:%H:%M:%S %z')
            
            return {
                'ip': ip,
                'timestamp': timestamp,
                'method': method,
                'url': url,
                'protocol': protocol,
                'status': int(status),
                'size': size if size != '-' else 0,
                'referer': referer,
                'user_agent': user_agent
            }
        except Exception:
            return None
    
    def _is_within_date_range(self, timestamp: datetime, cutoff_date: datetime) -> bool:
        """Check if timestamp is within analysis range"""
        return timestamp >= cutoff_date
    
    def _identify_ai_crawler(self, user_agent: str, ip: str) -> Optional[Dict]:
        """Identify if a request is from an AI crawler"""
        for crawler_type, crawler_info in self.ai_crawler_patterns.items():
            # Check user agent patterns
            for pattern in crawler_info['user_agents']:
                if re.search(pattern, user_agent, re.IGNORECASE):
                    return {
                        'type': crawler_type,
                        'name': crawler_info['name'],
                        'purpose': crawler_info['purpose']
                    }
            
            # Check IP patterns
            for ip_pattern in crawler_info['ips']:
                if ip.startswith(ip_pattern):
                    return {
                        'type': crawler_type,
                        'name': crawler_info['name'],
                        'purpose': crawler_info['purpose']
                    }
        
        return None
    
    def _analyze_crawler_types(self, ai_requests: List[Dict]) -> Dict:
        """Analyze AI crawler activity by type"""
        crawler_stats = defaultdict(lambda: {
            'requests': 0,
            'unique_pages': set(),
            'success_rate': 0,
            'avg_requests_per_day': 0
        })
        
        for request in ai_requests:
            crawler_type = request.get('crawler_type', 'unknown')
            crawler_stats[crawler_type]['requests'] += 1
            crawler_stats[crawler_type]['unique_pages'].add(request['url'])
            
        # Convert sets to counts and calculate metrics
        for crawler_type, stats in crawler_stats.items():
            stats['unique_pages'] = len(stats['unique_pages'])
            stats['success_rate'] = round(
                len([r for r in ai_requests if r.get('crawler_type') == crawler_type and r['status'] == 200]) / 
                max(1, stats['requests']) * 100, 1
            )
            # Approximate requests per day (assumes 30-day period)
            stats['avg_requests_per_day'] = round(stats['requests'] / 30, 1)
        
        return dict(crawler_stats)
    
    def _analyze_crawled_pages(self, ai_requests: List[Dict]) -> Dict:
        """Analyze which pages AI crawlers visit most"""
        page_stats = defaultdict(lambda: {
            'visits': 0,
            'crawlers': set(),
            'last_crawled': None,
            'success_rate': 0
        })
        
        for request in ai_requests:
            url = request['url']
            page_stats[url]['visits'] += 1
            page_stats[url]['crawlers'].add(request.get('crawler_name', 'Unknown'))
            
            # Track most recent crawl
            if not page_stats[url]['last_crawled'] or request['timestamp'] > page_stats[url]['last_crawled']:
                page_stats[url]['last_crawled'] = request['timestamp']
        
        # Convert to list and sort by visits
        top_pages = []
        for url, stats in page_stats.items():
            stats['crawlers'] = list(stats['crawlers'])
            stats['last_crawled'] = stats['last_crawled'].isoformat() if stats['last_crawled'] else None
            
            # Calculate success rate
            url_requests = [r for r in ai_requests if r['url'] == url]
            successful = len([r for r in url_requests if r['status'] == 200])
            stats['success_rate'] = round(successful / max(1, len(url_requests)) * 100, 1)
            
            top_pages.append({
                'url': url,
                'visits': stats['visits'],
                'crawlers': stats['crawlers'],
                'last_crawled': stats['last_crawled'],
                'success_rate': stats['success_rate']
            })
        
        # Sort by visits and return top 20
        top_pages.sort(key=lambda x: x['visits'], reverse=True)
        return {'pages': top_pages[:20]}
    
    def _analyze_crawl_patterns(self, ai_requests: List[Dict]) -> Dict:
        """Analyze AI crawler behavior patterns"""
        patterns = {
            'hourly_distribution': defaultdict(int),
            'daily_distribution': defaultdict(int),
            'response_codes': defaultdict(int),
            'crawl_depth': defaultdict(int),
            'file_types': defaultdict(int)
        }
        
        for request in ai_requests:
            # Hourly distribution
            hour = request['timestamp'].hour
            patterns['hourly_distribution'][hour] += 1
            
            # Daily distribution
            day = request['timestamp'].strftime('%Y-%m-%d')
            patterns['daily_distribution'][day] += 1
            
            # Response codes
            patterns['response_codes'][request['status']] += 1
            
            # Crawl depth (number of path segments)
            depth = len([p for p in request['url'].split('/') if p])
            patterns['crawl_depth'][depth] += 1
            
            # File types
            if '.' in request['url'].split('/')[-1]:
                ext = request['url'].split('.')[-1].lower()
                patterns['file_types'][ext] += 1
            else:
                patterns['file_types']['html'] += 1
        
        # Convert to regular dicts and sort
        return {
            'hourly_distribution': dict(sorted(patterns['hourly_distribution'].items())),
            'daily_distribution': dict(sorted(patterns['daily_distribution'].items())),
            'response_codes': dict(sorted(patterns['response_codes'].items())),
            'crawl_depth': dict(sorted(patterns['crawl_depth'].items())),
            'file_types': dict(sorted(patterns['file_types'].items(), key=lambda x: x[1], reverse=True))
        }
    
    def _generate_crawler_recommendations(self, results: Dict) -> List[Dict]:
        """Generate recommendations for improving AI crawler accessibility"""
        recommendations = []
        
        # Check AI crawler presence
        if results['ai_requests'] == 0:
            recommendations.append({
                'priority': 'high',
                'category': 'crawler_accessibility',
                'title': 'No AI Crawler Activity Detected',
                'description': 'AI crawlers are not visiting your site. Improve discoverability.',
                'actions': [
                    'Submit sitemap to search engines',
                    'Ensure robots.txt allows AI crawlers',
                    'Add structured data markup',
                    'Create quality, linkable content'
                ]
            })
        
        # Check for high error rates
        total_ai = results['ai_requests']
        if total_ai > 0:
            error_rate = sum(
                stats['requests'] for crawler, stats in results['crawlers_detected'].items()
                if stats.get('success_rate', 100) < 90
            ) / total_ai
            
            if error_rate > 0.1:  # >10% error rate
                recommendations.append({
                    'priority': 'medium',
                    'category': 'technical_issues',
                    'title': 'High Error Rate for AI Crawlers',
                    'description': f'AI crawlers experiencing {error_rate*100:.1f}% error rate',
                    'actions': [
                        'Fix broken internal links',
                        'Improve server response times',
                        'Check for 404 errors on important pages',
                        'Optimize page loading speed'
                    ]
                })
        
        # Check crawler diversity
        unique_crawlers = len(results['crawlers_detected'])
        if unique_crawlers < 3:
            recommendations.append({
                'priority': 'medium',
                'category': 'crawler_diversity',
                'title': 'Limited AI Crawler Diversity',
                'description': f'Only {unique_crawlers} AI crawler types detected',
                'actions': [
                    'Improve content quality and freshness',
                    'Add more structured data',
                    'Increase internal linking',
                    'Build more external backlinks'
                ]
            })
        
        # Check for popular page accessibility
        top_pages = results.get('top_crawled_pages', {}).get('pages', [])
        if top_pages:
            low_success_pages = [p for p in top_pages[:10] if p['success_rate'] < 95]
            if low_success_pages:
                recommendations.append({
                    'priority': 'high',
                    'category': 'page_accessibility',
                    'title': 'Important Pages Have Crawler Issues',
                    'description': f'{len(low_success_pages)} frequently crawled pages have accessibility issues',
                    'actions': [
                        'Fix server errors on popular pages',
                        'Improve page load times',
                        'Ensure consistent URL structure',
                        'Remove redirect chains'
                    ]
                })
        
        return recommendations
    
    def generate_crawler_report(self, output_file: str = None) -> str:
        """Generate a comprehensive AI crawler analytics report"""
        if not self.analysis_results:
            return "No analysis results available. Run parse_access_logs() first."
        
        results = self.analysis_results
        
        report = f"""
# 🤖 AI Crawler Analytics Report
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## 📊 Summary
- **Analysis Period**: {results['analysis_period']}
- **Total Requests**: {results['total_requests']:,}
- **AI Crawler Requests**: {results['ai_requests']:,} ({results.get('ai_request_percentage', 0)}%)
- **AI Crawlers Detected**: {len(results['crawlers_detected'])}

## 🤖 AI Crawler Activity

"""
        
        for crawler_type, stats in results['crawlers_detected'].items():
            crawler_name = self.ai_crawler_patterns.get(crawler_type, {}).get('name', crawler_type)
            report += f"### {crawler_name}\n"
            report += f"- **Requests**: {stats['requests']:,}\n"
            report += f"- **Unique Pages**: {stats['unique_pages']}\n" 
            report += f"- **Success Rate**: {stats['success_rate']}%\n"
            report += f"- **Avg Daily Requests**: {stats['avg_requests_per_day']}\n\n"
        
        report += "## 📄 Most Crawled Pages\n\n"
        for i, page in enumerate(results['top_crawled_pages']['pages'][:10], 1):
            report += f"{i}. **{page['url']}**\n"
            report += f"   - Visits: {page['visits']}\n"
            report += f"   - Crawlers: {', '.join(page['crawlers'])}\n"
            report += f"   - Success Rate: {page['success_rate']}%\n"
            report += f"   - Last Crawled: {page['last_crawled']}\n\n"
        
        report += "## 🔧 Recommendations\n\n"
        for rec in results['recommendations']:
            priority_emoji = "🔴" if rec['priority'] == 'high' else "🟡" if rec['priority'] == 'medium' else "🟢"
            report += f"{priority_emoji} **{rec['title']}** ({rec['priority']} priority)\n"
            report += f"{rec['description']}\n\n"
            report += "**Actions:**\n"
            for action in rec['actions']:
                report += f"- {action}\n"
            report += "\n"
        
        if output_file:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(report)
            logger.info(f"✅ AI Crawler report saved to {output_file}")
        
        return report
    
    def create_robots_txt_for_ai(self, sitemap_url: str = None) -> str:
        """Generate AI-friendly robots.txt content"""
        robots_content = """# AI-Optimized robots.txt
# Generated by AIO Search Tool

# Allow all major AI crawlers
User-agent: ChatGPT-User
Allow: /

User-agent: CCBot
Allow: /

User-agent: GPTBot  
Allow: /

User-agent: PerplexityBot
Allow: /

User-agent: Google-Extended
Allow: /

User-agent: BingBot
Allow: /

User-agent: *
Allow: /

# Block problematic paths for AI crawlers
Disallow: /admin/
Disallow: /private/
Disallow: /wp-admin/
Disallow: /wp-includes/
Disallow: /*.pdf$
Disallow: /search/
Disallow: /cart/
Disallow: /checkout/

# Crawl delay to be respectful
Crawl-delay: 1

"""
        
        if sitemap_url:
            robots_content += f"# Sitemap for AI discovery\nSitemap: {sitemap_url}\n"
        
        return robots_content

# Example usage
if __name__ == "__main__":
    analytics = AICrawlerAnalytics()
    
    # Example: Analyze server logs
    # results = analytics.parse_access_logs('/var/log/apache2/access.log', days_back=30)
    
    # Generate example report
    print("🤖 AI Crawler Analytics Module Ready!")
    print("📊 Use analytics.parse_access_logs(log_file_path) to analyze your server logs")
    print("📄 Use analytics.generate_crawler_report() to create comprehensive reports")
    print("🤖 Use analytics.create_robots_txt_for_ai() to generate AI-friendly robots.txt") 