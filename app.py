# app.py
import os
import argparse
import logging
import json
import pandas as pd
from datetime import datetime
import streamlit as st
import asyncio

# Import module classes
from content_optimizer import ContentOptimizer
from airtop_integration import AirtopLLMVisibility as AIVisibilityChecker
from question_intent_mapper import QuestionIntentMapper
from ai_sitemap_generator import AISitemapGenerator

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

class AIOSearchTool:
    def __init__(self):
        """Initialize the AI Optimization Search Tool"""
        # Initialize modules
        self.content_optimizer = ContentOptimizer()
        self.visibility_checker = AIVisibilityChecker()
        self.question_mapper = QuestionIntentMapper()
        self.sitemap_generator = AISitemapGenerator()
        
        # Create output directory
        self.output_dir = os.path.join(os.getcwd(), 'aio_output')
        os.makedirs(self.output_dir, exist_ok=True)
        
    def run_content_optimizer(self, content_file, brand_name, keywords, industry):
        """
        Run the Content Optimizer module
        
        Args:
            content_file (str): Path to content file
            brand_name (str): Brand name
            keywords (list): List of keywords
            industry (str): Industry category
            
        Returns:
            tuple: (str, str) Paths to optimized content and optimization report
        """
        logger.info(f"Running Content Optimizer for {content_file}")
        
        try:
            # Read content file
            with open(content_file, 'r', encoding='utf-8') as f:
                content = f.read()
                
            # Optimize content
            optimized_content, scores, changes = self.content_optimizer.optimize_content(
                content, brand_name, keywords, industry
            )
            
            # Create output filenames
            filename = os.path.basename(content_file)
            base_name, ext = os.path.splitext(filename)
            output_file = os.path.join(self.output_dir, f"{base_name}_optimized.md")
            report_file = os.path.join(self.output_dir, f"{base_name}_optimization_report.md")
            
            # Write optimized content
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(optimized_content)
            
            # Generate and write optimization report
            report_content = self._generate_optimization_report(scores, changes, content, optimized_content)
            with open(report_file, 'w', encoding='utf-8') as f:
                f.write(report_content)
                
            logger.info(f"Content optimization complete. Output saved to {output_file}")
            return output_file, report_file
            
        except Exception as e:
            logger.error(f"Error in content optimization: {str(e)}")
            return None, None
    
    def _generate_optimization_report(self, scores, changes, original_content, optimized_content):
        """Generate a detailed optimization report"""
        report = [
            "# Content Optimization Report\n",
            "## Optimization Scores\n",
            "| Category | Score |",
            "|----------|--------|"
        ]
        
        # Add scores table
        for category, score in scores.items():
            report.append(f"| {category.replace('_', ' ').title()} | {score:.1f}/100 |")
        
        report.append("\n## Changes Made\n")
        
        # Add changes for each category
        for category, category_changes in changes.items():
            if category_changes:
                report.append(f"### {category.replace('_', ' ').title()}\n")
                for change in category_changes:
                    if 'similarity' in change:
                        report.append(f"- Similarity: {change['similarity']}%")
                    if 'length_change' in change:
                        report.append(f"- Length Change: {change['length_change']} characters")
                    if 'sample' in change and change['sample'].get('original'):
                        report.append("- Sample Change:")
                        report.append(f"  - Before: \"{change['sample']['original']}\"")
                        report.append(f"  - After: \"{change['sample']['modified']}\"")
                    if 'improvement' in change:
                        report.append(f"- Improvement: {change['improvement']}")
                report.append("")
        
        # Add content length comparison
        report.extend([
            "## Content Statistics\n",
            "| Metric | Original | Optimized |",
            "|--------|-----------|-----------|",
            f"| Length (chars) | {len(original_content)} | {len(optimized_content)} |",
            f"| Words | {len(original_content.split())} | {len(optimized_content.split())} |",
            f"| Paragraphs | {len([p for p in original_content.split('\n\n') if p.strip()])} | {len([p for p in optimized_content.split('\n\n') if p.strip()])} |"
        ])
        
        return "\n".join(report)
    
    async def run_visibility_checker(self, brand_name, competitor_names, industry, product_categories, num_prompts=5):
        """
        Run the AI Visibility Checker module using Airtop browser automation
        
        Args:
            brand_name (str): Brand name
            competitor_names (list): List of competitor names
            industry (str): Industry category
            product_categories (list): List of product categories
            num_prompts (int): Number of prompts per category
            
        Returns:
            str: Path to visibility report
        """
        logger.info(f"Running Airtop AI Visibility Checker for {brand_name}")
        
        try:
            # Generate test queries based on industry and product categories
            queries = []
            for category in product_categories[:2]:  # Limit to 2 categories for demo
                queries.append(f"What are the best {category} tools for {industry}?")
                queries.append(f"Compare top {category} solutions in {industry}")
            
            # Add competitor comparison query
            if competitor_names:
                queries.append(f"Alternatives to {competitor_names[0]} for {industry}")
            
            # Run Airtop visibility check with browser automation
            airtop_results = await self.visibility_checker.run_visibility_check(
                brand_name=brand_name,
                competitors=competitor_names,
                queries=queries,
                platforms=["perplexity"]  # Start with Perplexity (no login required)
            )
            
            if not airtop_results.get('success', False):
                logger.error(f"Airtop visibility check failed: {airtop_results.get('error', 'Unknown error')}")
                return None
            
            # Convert Airtop results to DataFrame format
            results_data = []
            for result in airtop_results.get('results', []):
                results_data.append({
                    "prompt": result.get('query', ''),
                    "provider": result.get('platform', 'unknown'),
                    "model": f"{result.get('platform', 'unknown')}_web",
                    "response": result.get('response', ''),
                    "brand_mentioned": result.get('brand_mentioned', False),
                    "mention_type": result.get('mention_type', 'none'),
                    "competitors_mentioned": result.get('competitors_mentioned', []),
                    "timestamp": result.get('timestamp', '')
                })
            
            results_df = pd.DataFrame(results_data)
            
            # Create output filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = os.path.join(self.output_dir, f"airtop_visibility_report_{timestamp}.csv")
            
            # Save results
            results_df.to_csv(output_file, index=False)
            
            # Create summary report
            summary_file = os.path.join(self.output_dir, f"airtop_visibility_summary_{timestamp}.json")
            self._create_airtop_visibility_summary(airtop_results, brand_name, competitor_names, summary_file)
            
            logger.info(f"Airtop visibility check complete. Output saved to {output_file}")
            return output_file
            
        except Exception as e:
            logger.error(f"Error in Airtop visibility check: {str(e)}")
            return None

    def _create_airtop_visibility_summary(self, airtop_results, brand_name, competitor_names, output_file):
        """Create a summary of Airtop visibility results"""
        summary = airtop_results.get('summary', {})
        
        # Add additional analysis
        summary.update({
            "brand": brand_name,
            "competitors": competitor_names,
            "platforms_tested": summary.get('platforms_tested', []),
            "airtop_session_id": airtop_results.get('session_id', ''),
            "total_queries": summary.get('total_queries', 0),
            "successful_queries": summary.get('successful_queries', 0),
            "brand_mentions": summary.get('brand_mentions', 0),
            "brand_mention_rate": summary.get('brand_mention_rate', 0),
            "success_rate": summary.get('success_rate', 0),
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })
        
        # Save summary
        with open(output_file, 'w') as f:
            json.dump(summary, f, indent=2)
    
    def run_question_mapper(self, topics, industry, brand_name, competitors, product_features, num_questions=100):
        """
        Run the Question & Intent Mapper module
        
        Args:
            topics (list): Main topics
            industry (str): Industry category
            brand_name (str): Brand name
            competitors (list): List of competitor names
            product_features (list): List of product features
            num_questions (int): Number of questions to generate
            
        Returns:
            str: Path to question intents file
        """
        logger.info(f"Running Question & Intent Mapper for {brand_name}")
        
        try:
            # Generate questions
            questions = self.question_mapper.generate_questions(
                topics, industry, brand_name, competitors, product_features, num_questions
            )
            
            # Cluster questions
            clusters = self.question_mapper.cluster_questions(questions)
            
            # Create output filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = os.path.join(self.output_dir, f"question_intents_{timestamp}.json")
            
            # Export results
            self.question_mapper.export_results(clusters, output_file)
            
            # Generate visualization
            try:
                self.question_mapper.visualize_clusters(questions, clusters)
                viz_file = os.path.join(self.output_dir, "question_clusters.png")
                logger.info(f"Visualization saved to {viz_file}")
            except Exception as viz_error:
                logger.warning(f"Could not generate visualization: {str(viz_error)}")
            
            logger.info(f"Question mapping complete. Output saved to {output_file}")
            return output_file
            
        except Exception as e:
            logger.error(f"Error in question mapping: {str(e)}")
            return None
    
    def run_sitemap_generator(self, url, brand_info, max_pages=100):
        """
        Run the AI Sitemap Generator module
        
        Args:
            url (str): Website URL
            brand_info (dict): Brand information
            max_pages (int): Maximum pages to crawl
            
        Returns:
            tuple: Paths to site-ai.yaml and llms.txt files
        """
        logger.info(f"Running AI Sitemap Generator for {url}")
        
        try:
            # Crawl website
            pages = self.sitemap_generator.crawl_website(url, max_pages=max_pages)
            
            # Generate site-ai.yaml
            yaml_file = os.path.join(self.output_dir, 'site-ai.yaml')
            self.sitemap_generator.generate_site_ai_yaml(pages, brand_info, output_file=yaml_file)
            
            # Generate llms.txt
            txt_file = os.path.join(self.output_dir, 'llms.txt')
            self.sitemap_generator.generate_llms_txt(pages, brand_info, output_file=txt_file)
            
            # Analyze site for optimization opportunities
            analysis = self.sitemap_generator.analyze_site_for_ai_optimization(pages)
            
            # Export analysis
            analysis_file = os.path.join(self.output_dir, 'ai_optimization_analysis.json')
            with open(analysis_file, 'w') as f:
                json.dump(analysis, f, indent=2)
                
            logger.info(f"Sitemap generation complete. Files saved to {self.output_dir}")
            return (yaml_file, txt_file, analysis_file)
            
        except Exception as e:
            logger.error(f"Error in sitemap generation: {str(e)}")
            return None

def main():
    """Main entry point for the AIO Search Tool"""
    parser = argparse.ArgumentParser(description='AI Optimization Search Tool')
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Content Optimizer command
    content_parser = subparsers.add_parser('optimize', help='Optimize content for LLM citation')
    content_parser.add_argument('--file', required=True, help='Path to content file')
    content_parser.add_argument('--brand', required=True, help='Brand name')
    content_parser.add_argument('--keywords', required=True, help='Comma-separated keywords')
    content_parser.add_argument('--industry', required=True, help='Industry category')
    
    # Visibility Checker command
    visibility_parser = subparsers.add_parser('check', help='Check AI visibility')
    visibility_parser.add_argument('--brand', required=True, help='Brand name')
    visibility_parser.add_argument('--competitors', required=True, help='Comma-separated competitor names')
    visibility_parser.add_argument('--industry', required=True, help='Industry category')
    visibility_parser.add_argument('--categories', required=True, help='Comma-separated product categories')
    visibility_parser.add_argument('--prompts', type=int, default=5, help='Number of prompts per category')
    
    # Question Mapper command
    question_parser = subparsers.add_parser('map', help='Map questions and intents')
    question_parser.add_argument('--topics', required=True, help='Comma-separated topics')
    question_parser.add_argument('--industry', required=True, help='Industry category')
    question_parser.add_argument('--brand', required=True, help='Brand name')
    question_parser.add_argument('--competitors', required=True, help='Comma-separated competitor names')
    question_parser.add_argument('--features', required=True, help='Comma-separated product features')
    question_parser.add_argument('--questions', type=int, default=100, help='Number of questions to generate')
    
    # Sitemap Generator command
    sitemap_parser = subparsers.add_parser('sitemap', help='Generate AI sitemaps')
    sitemap_parser.add_argument('--url', required=True, help='Website URL')
    sitemap_parser.add_argument('--brand', required=True, help='Brand name')
    sitemap_parser.add_argument('--description', required=True, help='Brand description')
    sitemap_parser.add_argument('--industry', required=True, help='Industry category')
    sitemap_parser.add_argument('--tagline', required=True, help='Brand tagline')
    sitemap_parser.add_argument('--value', required=True, help='Value proposition')
    sitemap_parser.add_argument('--differentiators', required=True, help='Comma-separated differentiators')
    sitemap_parser.add_argument('--max-pages', type=int, default=100, help='Maximum pages to crawl')
    
    # All-in-one command
    all_parser = subparsers.add_parser('all', help='Run all modules')
    all_parser.add_argument('--config', required=True, help='Path to config JSON file')
    
    args = parser.parse_args()
    
    # Initialize the tool
    tool = AIOSearchTool()
    
    if args.command == 'optimize':
        # Run Content Optimizer
        keywords = [k.strip() for k in args.keywords.split(',')]
        tool.run_content_optimizer(args.file, args.brand, keywords, args.industry)
        
    elif args.command == 'check':
        # Run Visibility Checker
        competitors = [c.strip() for c in args.competitors.split(',')]
        categories = [c.strip() for c in args.categories.split(',')]
        asyncio.run(tool.run_visibility_checker(args.brand, competitors, args.industry, categories, args.prompts))
        
    elif args.command == 'map':
        # Run Question Mapper
        topics = [t.strip() for t in args.topics.split(',')]
        competitors = [c.strip() for c in args.competitors.split(',')]
        features = [f.strip() for f in args.features.split(',')]
        tool.run_question_mapper(topics, args.industry, args.brand, competitors, features, args.questions)
        
    elif args.command == 'sitemap':
        # Run Sitemap Generator
        differentiators = [d.strip() for d in args.differentiators.split(',')]
        brand_info = {
            "name": args.brand,
            "description": args.description,
            "url": args.url,
            "industry": args.industry,
            "tagline": args.tagline,
            "value_proposition": args.value,
            "key_differentiators": differentiators
        }
        tool.run_sitemap_generator(args.url, brand_info, args.max_pages)
        
    elif args.command == 'all':
        # Run all modules from config
        try:
            with open(args.config, 'r') as f:
                config = json.load(f)
                
            logger.info("Running all modules from config")
            
            # Run Content Optimizer if configured
            if 'content_optimizer' in config:
                co_config = config['content_optimizer']
                tool.run_content_optimizer(
                    co_config['file'],
                    co_config['brand'],
                    co_config['keywords'],
                    co_config['industry']
                )
                
            # Run Visibility Checker if configured
            if 'visibility_checker' in config:
                vc_config = config['visibility_checker']
                asyncio.run(tool.run_visibility_checker(
                    vc_config['brand'],
                    vc_config['competitors'],
                    vc_config['industry'],
                    vc_config['categories'],
                    vc_config.get('prompts', 5)
                ))
                
            # Run Question Mapper if configured
            if 'question_mapper' in config:
                qm_config = config['question_mapper']
                tool.run_question_mapper(
                    qm_config['topics'],
                    qm_config['industry'],
                    qm_config['brand'],
                    qm_config['competitors'],
                    qm_config['features'],
                    qm_config.get('questions', 100)
                )
                
            # Run Sitemap Generator if configured
            if 'sitemap_generator' in config:
                sg_config = config['sitemap_generator']
                brand_info = {
                    "name": sg_config['brand'],
                    "description": sg_config['description'],
                    "url": sg_config['url'],
                    "industry": sg_config['industry'],
                    "tagline": sg_config['tagline'],
                    "value_proposition": sg_config['value_proposition'],
                    "key_differentiators": sg_config['differentiators']
                }
                tool.run_sitemap_generator(
                    sg_config['url'],
                    brand_info,
                    sg_config.get('max_pages', 100)
                )
                
            logger.info("All modules completed successfully")
                
        except Exception as e:
            logger.error(f"Error running all modules: {str(e)}")
    
    else:
        parser.print_help()

def show_content_optimizer(optimizer):
    st.title("Content Optimizer")
    st.subheader("Transform your content for better LLM citation")
    
    with st.expander("How it works", expanded=False):
        st.markdown("""
        The Content Optimizer rewrites your existing content to be better suited for LLM citation. It:
        1. Improves semantic clarity for better understanding by AI systems
        2. Adds structured Q&A-style phrasing that matches how users ask questions
        3. Inserts branded and quotable statements that are more likely to be cited
        4. Formats content specifically for vector-based RAG systems
        5. Adds metadata that helps AI systems understand the content's authority and relevance
        """)
    
    # Input section
    st.markdown("### Input Content")
    
    input_method = st.radio("Select input method:", ["Enter text", "Upload file"])
    
    content = ""
    if input_method == "Enter text":
        content = st.text_area("Enter your content:", height=200, placeholder="Paste your content here...")
    else:
        uploaded_file = st.file_uploader("Choose a file", type=['txt', 'md'])
        if uploaded_file is not None:
            content = uploaded_file.getvalue().decode("utf-8")
            st.text_area("File content:", content, height=200)
    
    # Configuration
    st.markdown("### Configuration")
    col1, col2 = st.columns(2)
    
    with col1:
        brand_name = st.text_input("Brand name:", value="AIO Search")
        industry = st.text_input("Industry:", value="digital marketing")
    
    with col2:
        keywords_input = st.text_input("Keywords (comma-separated):", value="AI content optimization,LLM SEO,semantic relevance")
        keywords = [k.strip() for k in keywords_input.split(',')]
    
    # Optimization button and results
    if st.button("Optimize Content"):
        if content:
            with st.spinner("Optimizing content..."):
                try:
                    # Create temporary file for content
                    temp_file = "temp_content.md"
                    with open(temp_file, "w") as f:
                        f.write(content)
                    
                    # Run optimization
                    optimized_content, scores, changes = optimizer.content_optimizer.optimize_content(
                        content, brand_name, keywords, industry
                    )
                    
                    # Generate report
                    report_content = optimizer._generate_optimization_report(
                        scores, changes, content, optimized_content
                    )
                    
                    # Display results in tabs
                    tab1, tab2 = st.tabs(["Optimized Content", "Optimization Report"])
                    
                    with tab1:
                        st.markdown(optimized_content)
                        
                    with tab2:
                        st.markdown(report_content)
                    
                    # Cleanup temporary file
                    if os.path.exists(temp_file):
                        os.remove(temp_file)
                        
                except Exception as e:
                    st.error(f"Error optimizing content: {str(e)}")
        else:
            st.warning("Please enter some content to optimize.")

if __name__ == "__main__":
    main()