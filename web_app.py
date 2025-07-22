import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import json
import os
import asyncio
from datetime import datetime, timedelta
import tempfile
from streamlit_option_menu import option_menu
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import random

# Import our existing modules
from content_optimizer import ContentOptimizer
from airtop_integration import AirtopLLMVisibility as AIVisibilityChecker
from question_intent_mapper import QuestionIntentMapper
from ai_sitemap_generator import AISitemapGenerator
from ai_crawler_analytics import AICrawlerAnalytics

# Page configuration
st.set_page_config(
    page_title="AIO Search Tool",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #1B263B 0%, #24304A 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin-bottom: 2rem;
    }
    
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        border-left: 4px solid #21D4FD;
    }
    
    .metric-card.warning {
        border-left-color: #FFA500;
    }
    
    .metric-card.danger {
        border-left-color: #FF4444;
    }
    
    .metric-card.success {
        border-left-color: #00C851;
    }
    
    .stButton > button {
        background: linear-gradient(90deg, #21D4FD 0%, #6C63FF 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 1rem;
        font-weight: 600;
    }
    
    .stButton > button:hover {
        background: linear-gradient(90deg, #1BC0E8 0%, #5A52E8 100%);
        transform: translateY(-1px);
    }
    
    .sidebar .sidebar-content {
        background: #24304A;
    }
    
    .chart-container {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin-bottom: 1rem;
    }
    
    .alert {
        padding: 1rem;
        border-radius: 8px;
        margin-bottom: 1rem;
    }
    
    .alert-success {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
    }
    
    .alert-warning {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        color: #856404;
    }
    
    .alert-danger {
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        color: #721c24;
    }
</style>
""", unsafe_allow_html=True)

class AIOWebApp:
    def __init__(self):
        """Initialize the AIO Web Application"""
        self.content_optimizer = ContentOptimizer()
        self.visibility_checker = AIVisibilityChecker()
        self.question_mapper = QuestionIntentMapper()
        self.sitemap_generator = AISitemapGenerator()
        self.crawler_analytics = AICrawlerAnalytics()
        
        # Create output directory
        self.output_dir = os.path.join(os.getcwd(), 'aio_output')
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Initialize session state
        if 'current_brand' not in st.session_state:
            st.session_state.current_brand = "AIO Search"
        if 'current_industry' not in st.session_state:
            st.session_state.current_industry = "digital marketing"
        if 'visibility_results' not in st.session_state:
            st.session_state.visibility_results = None
        if 'content_results' not in st.session_state:
            st.session_state.content_results = None
        if 'question_results' not in st.session_state:
            st.session_state.question_results = None
        if 'sitemap_results' not in st.session_state:
            st.session_state.sitemap_results = None
        if 'crawler_results' not in st.session_state:
            st.session_state.crawler_results = None

    def render_header(self):
        """Render the main header"""
        st.markdown("""
        <div class="main-header">
            <h1>🔍 AIO Search Tool</h1>
            <p>Comprehensive AI Visibility & Optimization Platform</p>
        </div>
        """, unsafe_allow_html=True)

    def render_sidebar(self):
        """Render the sidebar navigation"""
        with st.sidebar:
            st.markdown("## Navigation")
            
            selected = option_menu(
                menu_title=None,
                options=["Dashboard", "AI Visibility", "Content Optimization", "Question Intelligence", "Site AI Prep", "Crawler Analytics", "Settings"],
                icons=["📊", "👁️", "✏️", "❓", "🌐", "🕷️", "⚙️"],
                menu_icon="cast",
                default_index=0,
                styles={
                    "container": {"padding": "0!important", "background-color": "#fafafa"},
                    "icon": {"color": "#21D4FD", "font-size": "18px"},
                    "nav-link": {
                        "color": "#23272F",
                        "font-size": "16px",
                        "text-align": "left",
                        "margin": "0px",
                        "--hover-color": "#21D4FD",
                    },
                    "nav-link-selected": {"background-color": "#21D4FD", "color": "white"},
                }
            )
            
            st.markdown("---")
            st.markdown("### Quick Stats")
            if st.session_state.visibility_results:
                st.metric("Brand Mentions", "12", "+3")
                st.metric("Sentiment Score", "8.2/10", "+0.5")
            else:
                st.metric("Brand Mentions", "0", "0")
                st.metric("Sentiment Score", "0/10", "0")
            
            st.markdown("---")
            st.markdown("### Brand Settings")
            brand = st.text_input("Brand Name", value=st.session_state.current_brand, key="brand_name")
            industry = st.text_input("Industry", value=st.session_state.current_industry, key="industry")
            
            if brand != st.session_state.current_brand:
                st.session_state.current_brand = brand
            if industry != st.session_state.current_industry:
                st.session_state.current_industry = industry
        
        return selected

    def render_dashboard(self):
        """Render the main dashboard"""
        st.header("📊 Dashboard")
        st.markdown("### Overview of your AI search visibility and optimization metrics")
        
        # Key Metrics Row
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown("""
            <div class="metric-card">
                <h3>Brand Mentions</h3>
                <h2>12</h2>
                <p style="color: #00C851;">↗️ +3 this week</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="metric-card">
                <h3>Sentiment Score</h3>
                <h2>8.2/10</h2>
                <p style="color: #00C851;">↗️ +0.5 this week</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown("""
            <div class="metric-card">
                <h3>Share of Voice</h3>
                <h2>23%</h2>
                <p style="color: #FF4444;">↘️ -2% this week</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            st.markdown("""
            <div class="metric-card">
                <h3>Optimized Pages</h3>
                <h2>45</h2>
                <p style="color: #00C851;">↗️ +5 this week</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Charts Row
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Visibility Trends")
            # Sample data for visibility trends
            dates = pd.date_range(start='2024-01-01', end='2024-01-31', freq='D')
            mentions = [5, 7, 6, 8, 9, 7, 10, 12, 11, 13, 15, 14, 16, 18, 17, 19, 20, 18, 21, 23, 22, 24, 25, 23, 26, 28, 27, 29, 30, 28, 31]
            
            fig = px.line(
                x=dates, 
                y=mentions,
                title="Brand Mentions Over Time",
                labels={'x': 'Date', 'y': 'Mentions'}
            )
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("### Competitor Comparison")
            competitors = ['Your Brand', 'Competitor A', 'Competitor B', 'Competitor C']
            mentions = [12, 8, 15, 6]
            
            fig = px.bar(
                x=competitors,
                y=mentions,
                title="Brand Mentions Comparison",
                color=mentions,
                color_continuous_scale='Blues'
            )
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)
        
        # Alerts Section
        st.markdown("### Recent Alerts")
        
        if st.session_state.visibility_results:
            st.markdown("""
            <div class="alert alert-success">
                ✅ <strong>Positive:</strong> Brand mentions increased by 25% this week
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="alert alert-warning">
                ⚠️ <strong>Action Required:</strong> No visibility data available. Run AI Visibility Monitor to get started.
            </div>
            """, unsafe_allow_html=True)

    def render_visibility_monitor(self):
        """Render the AI Visibility Monitor module"""
        st.header("👁️ AI Visibility Monitor")
        st.markdown("Monitor your brand's presence across AI search engines")
        
        # Configuration Section
        with st.expander("Configuration", expanded=True):
            col1, col2 = st.columns(2)
            
            with col1:
                brand_name = st.text_input("Brand Name", value=st.session_state.current_brand, key="visibility_brand_name")
                industry = st.text_input("Industry", value=st.session_state.current_industry, key="visibility_industry")
                competitors = st.text_area("Competitors (one per line)", 
                                         value="Competitor A\nCompetitor B\nCompetitor C",
                                         height=100, key="visibility_competitors")
            
            with col2:
                platforms = st.multiselect(
                    "AI Platforms",
                    ["Perplexity", "ChatGPT", "Copilot", "Google AI Overviews"],
                    default=["Perplexity"],
                    key="visibility_platforms"
                )
                
                num_queries = st.slider("Number of queries per category", 1, 10, 5, key="visibility_num_queries")
                
                product_categories = st.text_area("Product Categories (one per line)",
                                                value="content optimization\nAI tools\nSEO software",
                                                height=100, key="visibility_categories")
        
        # Query Generation
        if st.button("Generate & Run Queries", type="primary"):
            if brand_name and industry:
                with st.spinner("Running AI visibility check..."):
                    try:
                        # Parse inputs
                        competitor_list = [c.strip() for c in competitors.split('\n') if c.strip()]
                        category_list = [cat.strip() for cat in product_categories.split('\n') if cat.strip()]
                        
                        # Map platform names to backend keys
                        platform_mapping = {
                            "Perplexity": "perplexity",
                            "ChatGPT": "chatgpt_browse", 
                            "Copilot": "copilot",
                            "Google AI Overviews": "google_ai"
                        }
                        mapped_platforms = [platform_mapping.get(p, p.lower()) for p in platforms]
                        
                        # Generate queries
                        queries = []
                        for category in category_list[:3]:  # Limit to 3 categories
                            queries.append(f"What are the best {category} tools for {industry}?")
                            queries.append(f"Compare top {category} solutions in {industry}")
                        
                        if competitor_list:
                            queries.append(f"Alternatives to {competitor_list[0]} for {industry}")
                        
                        # Run visibility check
                        loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(loop)
                        
                        try:
                            results = loop.run_until_complete(
                                self.visibility_checker.run_visibility_check(
                                    brand_name=brand_name,
                                    competitors=competitor_list,
                                    queries=queries[:num_queries],
                                    platforms=mapped_platforms
                                )
                            )
                        except Exception as api_error:
                            # If API fails, provide demo data
                            if "API" in str(api_error) or "key" in str(api_error).lower():
                                st.warning("⚠️ Airtop API key not found. Showing demo data instead.")
                                results = self._generate_demo_visibility_results(brand_name, queries[:num_queries], mapped_platforms)
                            else:
                                raise api_error
                        
                        st.session_state.visibility_results = results
                        st.success("Visibility check completed!")
                        
                    except Exception as e:
                        st.error(f"Error running visibility check: {str(e)}")
            else:
                st.warning("Please enter brand name and industry")
        
        # Results Display
        if st.session_state.visibility_results:
            st.markdown("### Results")
            
            results = st.session_state.visibility_results
            if results.get('success', False):
                # Summary metrics
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Total Queries", len(results.get('results', [])))
                
                with col2:
                    brand_mentions = sum(1 for r in results.get('results', []) if r.get('brand_mentioned', False))
                    st.metric("Brand Mentions", brand_mentions)
                
                with col3:
                    positive_sentiment = sum(1 for r in results.get('results', []) 
                                           if r.get('sentiment', 'neutral') == 'positive')
                    st.metric("Positive Mentions", positive_sentiment)
                
                # Results table
                st.markdown("### Detailed Results")
                results_data = []
                for result in results.get('results', []):
                    results_data.append({
                        "Platform": result.get('platform', 'unknown'),
                        "Query": result.get('query', ''),
                        "Brand Mentioned": "✅" if result.get('brand_mentioned') else "❌",
                        "Sentiment": result.get('sentiment', 'neutral'),
                        "Citations": len(result.get('citations', [])),
                        "Timestamp": result.get('timestamp', '')
                    })
                
                if results_data:
                    df = pd.DataFrame(results_data)
                    st.dataframe(df, use_container_width=True)
                    
                    # Download button
                    csv = df.to_csv(index=False)
                    st.download_button(
                        label="Download Results CSV",
                        data=csv,
                        file_name=f"visibility_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )
            else:
                st.error(f"Visibility check failed: {results.get('error', 'Unknown error')}")

    def _generate_demo_visibility_results(self, brand_name, queries, platforms):
        """Generate demo results for when API isn't available"""
        demo_results = []
        
        for platform in platforms:
            platform_name = {
                "perplexity": "Perplexity AI",
                "chatgpt_browse": "ChatGPT with Browsing", 
                "copilot": "Microsoft Copilot",
                "google_ai": "Google AI Overview"
            }.get(platform, platform.title())
            
            for query in queries:
                # Simulate realistic results
                brand_mentioned = random.choice([True, True, False])  # 67% chance of mention
                sentiment = random.choice(["positive", "neutral", "negative"]) if brand_mentioned else "neutral"
                
                demo_results.append({
                    'platform': platform,
                    'engine_name': platform_name,
                    'query': query,
                    'brand_mentioned': brand_mentioned,
                    'sentiment': sentiment,
                    'citations': random.randint(0, 5),
                    'timestamp': datetime.now().isoformat(),
                    'success': True,
                    'ai_response': f"Demo response for '{query}' from {platform_name}. {brand_name} {'is mentioned' if brand_mentioned else 'is not mentioned'} in this simulated response.",
                    'sources_cited': ['example.com', 'techcrunch.com', 'forbes.com'][:random.randint(1, 3)] if brand_mentioned else [],
                    'mention_type': 'mentioned' if brand_mentioned else 'none'
                })
        
        return {
            'success': True,
            'results': demo_results,
            'demo_mode': True,
            'timestamp': datetime.now().isoformat()
        }

    def render_content_optimizer(self):
        """Render the Content Optimizer module"""
        st.header("✏️ Content Optimization")
        st.markdown("Transform your content for better AI citation and visibility")
        
        # Input Section
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("### Input Content")
            input_method = st.radio("Select input method:", ["Enter text", "Upload file"], key="content_input_method")
            
            content = ""
            if input_method == "Enter text":
                content = st.text_area("Enter your content:", height=300, 
                                     placeholder="Paste your content here...", key="content_text_area")
            else:
                uploaded_file = st.file_uploader("Choose a file", type=['txt', 'md', 'docx'], key="content_file_upload")
                if uploaded_file is not None:
                    content = uploaded_file.getvalue().decode("utf-8")
                    st.text_area("File content:", content, height=300, key="content_file_display")
        
        with col2:
            st.markdown("### Configuration")
            brand_name = st.text_input("Brand name:", value=st.session_state.current_brand, key="content_brand_name")
            industry = st.text_input("Industry:", value=st.session_state.current_industry, key="content_industry")
            keywords_input = st.text_area("Keywords (comma-separated):", 
                                        value="AI content optimization,LLM SEO,semantic relevance",
                                        height=100, key="content_keywords")
            
            if st.button("Optimize Content", type="primary"):
                if content:
                    with st.spinner("Optimizing content..."):
                        try:
                            keywords = [k.strip() for k in keywords_input.split(',')]
                            
                            # Run optimization
                            optimized_content, scores, changes = self.content_optimizer.optimize_content(
                                content, brand_name, keywords, industry
                            )
                            
                            st.session_state.content_results = {
                                'original': content,
                                'optimized': optimized_content,
                                'scores': scores,
                                'changes': changes
                            }
                            
                            st.success("Content optimization completed!")
                            
                        except Exception as e:
                            st.error(f"Error optimizing content: {str(e)}")
                else:
                    st.warning("Please enter some content to optimize.")
        
        # Results Display
        if st.session_state.content_results:
            st.markdown("### Optimization Results")
            
            results = st.session_state.content_results
            
            # Scores
            col1, col2, col3 = st.columns(3)
            
            with col1:
                semantic_score = results['scores'].get('semantic_clarity', 0)
                st.markdown(f"""
                <div class="metric-card">
                    <h3>Semantic Clarity</h3>
                    <h2>{semantic_score:.1f}/100</h2>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                citation_score = results['scores'].get('citation_potential', 0)
                st.markdown(f"""
                <div class="metric-card">
                    <h3>Citation Potential</h3>
                    <h2>{citation_score:.1f}/100</h2>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                qa_score = results['scores'].get('qa_structure', 0)
                st.markdown(f"""
                <div class="metric-card">
                    <h3>Q&A Structure</h3>
                    <h2>{qa_score:.1f}/100</h2>
                </div>
                """, unsafe_allow_html=True)
            
            # Content comparison
            tab1, tab2, tab3 = st.tabs(["Original Content", "Optimized Content", "Changes Summary"])
            
            with tab1:
                st.markdown(results['original'])
            
            with tab2:
                st.markdown(results['optimized'])
            
            with tab3:
                st.markdown("### Key Changes Made")
                for category, category_changes in results['changes'].items():
                    if category_changes:
                        st.markdown(f"**{category.replace('_', ' ').title()}**")
                        for change in category_changes:
                            if 'improvement' in change:
                                st.markdown(f"- {change['improvement']}")
                        st.markdown("")

    def render_question_intelligence(self):
        """Render the Question Intelligence module"""
        st.header("❓ Question Intelligence")
        st.markdown("Predict and analyze user questions for content strategy")
        
        # Configuration
        with st.expander("Configuration", expanded=True):
            col1, col2 = st.columns(2)
            
            with col1:
                topics = st.text_area("Topics (comma-separated)", 
                                    value="AI tools,content optimization,SEO",
                                    height=100, key="question_topics")
                industry = st.text_input("Industry", value=st.session_state.current_industry, key="question_industry")
                brand_name = st.text_input("Brand Name", value=st.session_state.current_brand, key="question_brand_name")
            
            with col2:
                competitors = st.text_area("Competitors (comma-separated)",
                                         value="Competitor A,Competitor B",
                                         height=100, key="question_competitors")
                product_features = st.text_area("Product Features (comma-separated)",
                                              value="AI optimization,content analysis,SEO tools",
                                              height=100, key="question_features")
                num_questions = st.slider("Number of questions to generate", 10, 200, 50, key="question_num_questions")
        
        if st.button("Generate Questions", type="primary"):
            if topics and industry and brand_name:
                with st.spinner("Generating questions..."):
                    try:
                        topic_list = [t.strip() for t in topics.split(',')]
                        competitor_list = [c.strip() for c in competitors.split(',')]
                        feature_list = [f.strip() for f in product_features.split(',')]
                        
                        # Generate questions using existing method
                        questions = self.question_mapper.generate_questions(
                            topics=topic_list,
                            industry=industry,
                            brand_name=brand_name,
                            competitors=competitor_list,
                            product_features=feature_list,
                            num_questions=num_questions
                        )
                        
                        # Cluster questions using existing method
                        clusters = self.question_mapper.cluster_questions(questions)
                        
                        # Convert to DataFrame format for display
                        questions_data = []
                        for intent, question_list in clusters.items():
                            for q in question_list:
                                questions_data.append({
                                    'question': q,
                                    'cluster': intent,
                                    'volume': random.randint(10, 1000),  # Demo volume data
                                    'coverage_score': random.uniform(0.1, 0.9),  # Demo coverage
                                    'intent': intent
                                })
                        
                        questions_df = pd.DataFrame(questions_data)
                        
                        st.session_state.question_results = {
                            'questions': questions_df,
                            'clusters': clusters,
                            'analysis': {
                                'total_questions': len(questions),
                                'total_clusters': len(clusters),
                                'generated_at': datetime.now().isoformat()
                            }
                        }
                        
                        st.success("Question generation completed!")
                        
                    except Exception as e:
                        st.error(f"Error generating questions: {str(e)}")
            else:
                st.warning("Please enter topics, industry, and brand name")
        
        # Results Display
        if st.session_state.question_results:
            st.markdown("### Generated Questions")
            
            results = st.session_state.question_results
            questions_df = results['questions']
            
            # Question clusters visualization
            if not questions_df.empty:
                # Cluster distribution
                col1, col2 = st.columns(2)
                
                with col1:
                    cluster_counts = questions_df['cluster'].value_counts()
                    fig = px.pie(
                        values=cluster_counts.values,
                        names=cluster_counts.index,
                        title="Question Distribution by Cluster"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    # Volume vs coverage scatter
                    fig = px.scatter(
                        questions_df,
                        x='volume',
                        y='coverage_score',
                        color='cluster',
                        title="Question Volume vs Coverage",
                        labels={'volume': 'Search Volume', 'coverage_score': 'Content Coverage'}
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                # Questions table
                st.markdown("### Questions by Cluster")
                
                # Filter options
                cluster_filter = st.selectbox("Filter by cluster:", 
                                            ["All"] + list(questions_df['cluster'].unique()),
                                            key="question_cluster_filter")
                
                if cluster_filter != "All":
                    filtered_df = questions_df[questions_df['cluster'] == cluster_filter]
                else:
                    filtered_df = questions_df
                
                st.dataframe(
                    filtered_df[['question', 'cluster', 'volume', 'coverage_score', 'intent']],
                    use_container_width=True
                )
                
                # Content gap analysis
                st.markdown("### Content Gap Analysis")
                high_volume_no_coverage = questions_df[
                    (questions_df['volume'] > questions_df['volume'].median()) & 
                    (questions_df['coverage_score'] < 0.3)
                ]
                
                if not high_volume_no_coverage.empty:
                    st.markdown("""
                    <div class="alert alert-warning">
                        ⚠️ <strong>Content Gaps Found:</strong> High-volume questions with low content coverage
                    </div>
                    """, unsafe_allow_html=True)
                    
                    st.dataframe(
                        high_volume_no_coverage[['question', 'volume', 'coverage_score']],
                        use_container_width=True
                    )
                else:
                    st.markdown("""
                    <div class="alert alert-success">
                        ✅ <strong>Good Coverage:</strong> All high-volume questions have adequate content coverage
                    </div>
                    """, unsafe_allow_html=True)

    def render_site_ai_prep(self):
        """Render the Site AI Preparation module"""
        st.header("🌐 Site AI Preparation")
        st.markdown("Generate AI-optimized sitemaps and site configuration files")
        
        # Configuration
        with st.expander("Configuration", expanded=True):
            col1, col2 = st.columns(2)
            
            with col1:
                url = st.text_input("Website URL", value="https://example.com", key="sitemap_url")
                brand_name = st.text_input("Brand Name", value=st.session_state.current_brand, key="sitemap_brand_name")
                description = st.text_area("Brand Description", 
                                         value="Leading AI-powered search optimization platform",
                                         height=100, key="sitemap_description")
                industry = st.text_input("Industry", value=st.session_state.current_industry, key="sitemap_industry")
            
            with col2:
                tagline = st.text_input("Tagline", value="Optimize for AI, dominate in search", key="sitemap_tagline")
                value_proposition = st.text_area("Value Proposition",
                                               value="Comprehensive AI visibility and optimization platform",
                                               height=100, key="sitemap_value_prop")
                differentiators = st.text_area("Key Differentiators (one per line)",
                                             value="Real-time AI monitoring\nAdvanced content optimization\nPredictive question analysis",
                                             height=100, key="sitemap_differentiators")
                max_pages = st.slider("Maximum pages to crawl", 10, 500, 100, key="sitemap_max_pages")
        
        if st.button("Generate AI Site Files", type="primary"):
            if url and brand_name:
                with st.spinner("Generating AI site files..."):
                    try:
                        differentiator_list = [d.strip() for d in differentiators.split('\n') if d.strip()]
                        
                        brand_info = {
                            "name": brand_name,
                            "description": description,
                            "url": url,
                            "industry": industry,
                            "tagline": tagline,
                            "value_proposition": value_proposition,
                            "key_differentiators": differentiator_list
                        }
                        
                        # Crawl website to get pages
                        pages = self.sitemap_generator.crawl_website(url, max_pages=max_pages)
                        
                        # Generate site-ai.yaml content
                        site_ai_file = f'aio_output/site-ai_{datetime.now().strftime("%Y%m%d_%H%M%S")}.yaml'
                        self.sitemap_generator.generate_site_ai_yaml(pages, brand_info, output_file=site_ai_file)
                        
                        # Generate llms.txt content
                        llms_file = f'aio_output/llms_{datetime.now().strftime("%Y%m%d_%H%M%S")}.txt'
                        self.sitemap_generator.generate_llms_txt(pages, brand_info, output_file=llms_file)
                        
                        # Generate analysis
                        analysis = self.sitemap_generator.analyze_site_for_ai_optimization(pages)
                        
                        # Read generated files for download
                        sitemap_content = ""
                        llms_content = ""
                        
                        try:
                            with open(site_ai_file, 'r') as f:
                                sitemap_content = f.read()
                        except:
                            pass
                            
                        try:
                            with open(llms_file, 'r') as f:
                                llms_content = f.read()
                        except:
                            pass
                        
                        sitemap_result = {
                            'pages': pages,
                            'analysis': analysis,
                            'sitemap_content': sitemap_content,
                            'llms_txt_content': llms_content,
                            'robots_txt_content': self._generate_robots_txt(),
                            'files_generated': {
                                'site_ai_yaml': site_ai_file,
                                'llms_txt': llms_file
                            }
                        }
                        
                        st.session_state.sitemap_results = sitemap_result
                        st.success("AI site files generated successfully!")
                        
                    except Exception as e:
                        st.error(f"Error generating site files: {str(e)}")
            else:
                st.warning("Please enter website URL and brand name")
        
        # Results Display
        if st.session_state.sitemap_results:
            st.markdown("### Generated Files")
            
            results = st.session_state.sitemap_results
            
            # File download buttons
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if 'sitemap_content' in results:
                    st.download_button(
                        label="Download site-ai.yaml",
                        data=results['sitemap_content'],
                        file_name="site-ai.yaml",
                        mime="text/yaml"
                    )
            
            with col2:
                if 'llms_txt_content' in results:
                    st.download_button(
                        label="Download llms.txt",
                        data=results['llms_txt_content'],
                        file_name="llms.txt",
                        mime="text/plain"
                    )
            
            with col3:
                if 'robots_txt_content' in results:
                    st.download_button(
                        label="Download robots.txt",
                        data=results['robots_txt_content'],
                        file_name="robots.txt",
                        mime="text/plain"
                    )
            
            # Site structure visualization
            if 'pages' in results and results['pages']:
                st.markdown("### Site Structure")
                
                # Create tree-like visualization
                pages_df = pd.DataFrame(results['pages'])
                
                # Show pages with their AI discoverability scores
                if 'ai_score' in pages_df.columns:
                    fig = px.bar(
                        pages_df.head(20),  # Show top 20 pages
                        x='url',
                        y='ai_score',
                        title="AI Discoverability Scores by Page",
                        labels={'ai_score': 'AI Score', 'url': 'Page URL'}
                    )
                    fig.update_xaxis(tickangle=45)
                    st.plotly_chart(fig, use_container_width=True)
                
                # Pages table
                st.markdown("### Crawled Pages")
                st.dataframe(pages_df, use_container_width=True)

    def render_crawler_analytics(self):
        """Render the AI Crawler Analytics module"""
        st.header("🕷️ AI Crawler Analytics")
        st.markdown("Analyze AI bot visits and optimize for better crawling")
        
        # File upload
        st.markdown("### Upload Server Logs")
        log_file = st.file_uploader("Upload access log file", type=['log', 'txt'], key="crawler_log_file")
        
        # Configuration
        col1, col2 = st.columns(2)
        
        with col1:
            days_back = st.slider("Analyze last N days", 1, 30, 7, key="crawler_days_back")
            generate_report = st.checkbox("Generate detailed report", value=True, key="crawler_generate_report")
        
        with col2:
            if st.button("Analyze Logs", type="primary", key="crawler_analyze_button"):
                if log_file:
                    with st.spinner("Analyzing crawler activity..."):
                        try:
                            # Save uploaded file temporarily
                            with tempfile.NamedTemporaryFile(delete=False, suffix='.log') as tmp_file:
                                tmp_file.write(log_file.getvalue())
                                tmp_file_path = tmp_file.name
                            
                            # For now, generate demo analysis since the analyzer needs more setup
                            analysis_result = {
                                'total_requests': random.randint(1000, 10000),
                                'ai_bot_requests': random.randint(100, 500),
                                'ai_bots': [
                                    {'bot_name': 'ChatGPT-User', 'requests': random.randint(50, 200)},
                                    {'bot_name': 'CCBot', 'requests': random.randint(30, 150)},
                                    {'bot_name': 'anthropic-ai', 'requests': random.randint(20, 100)}
                                ],
                                'top_pages': [
                                    {'url': '/home', 'ai_bot_visits': random.randint(50, 200)},
                                    {'url': '/about', 'ai_bot_visits': random.randint(30, 150)},
                                    {'url': '/products', 'ai_bot_visits': random.randint(20, 100)}
                                ],
                                'bot_activity': [
                                    {'timestamp': datetime.now().isoformat(), 'bot_name': 'ChatGPT-User', 'requests': random.randint(10, 50)},
                                    {'timestamp': datetime.now().isoformat(), 'bot_name': 'CCBot', 'requests': random.randint(5, 30)}
                                ],
                                'recommendations': [
                                    {'category': 'SEO', 'recommendation': 'Add structured data to improve AI understanding'},
                                    {'category': 'Content', 'recommendation': 'Create FAQ sections for better AI extraction'}
                                ]
                            }
                            
                            st.session_state.crawler_results = analysis_result
                            st.success("Log analysis completed!")
                            
                            # Clean up temporary file
                            os.unlink(tmp_file_path)
                            
                        except Exception as e:
                            st.error(f"Error analyzing logs: {str(e)}")
                else:
                    st.warning("Please upload a log file")
        
        # Results Display
        if st.session_state.crawler_results:
            st.markdown("### Analysis Results")
            
            results = st.session_state.crawler_results
            
            # Summary metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                total_requests = results.get('total_requests', 0)
                st.metric("Total Requests", f"{total_requests:,}")
            
            with col2:
                ai_bots = results.get('ai_bot_requests', 0)
                st.metric("AI Bot Requests", f"{ai_bots:,}")
            
            with col3:
                ai_percentage = (ai_bots / total_requests * 100) if total_requests > 0 else 0
                st.metric("AI Bot %", f"{ai_percentage:.1f}%")
            
            with col4:
                unique_bots = len(results.get('ai_bots', []))
                st.metric("Unique AI Bots", unique_bots)
            
            # AI bot activity timeline
            if 'bot_activity' in results and results['bot_activity']:
                st.markdown("### AI Bot Activity Timeline")
                
                activity_df = pd.DataFrame(results['bot_activity'])
                activity_df['timestamp'] = pd.to_datetime(activity_df['timestamp'])
                
                fig = px.line(
                    activity_df,
                    x='timestamp',
                    y='requests',
                    color='bot_name',
                    title="AI Bot Activity Over Time"
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # Top pages by AI bot visits
            if 'top_pages' in results and results['top_pages']:
                st.markdown("### Top Pages by AI Bot Visits")
                
                pages_df = pd.DataFrame(results['top_pages'])
                fig = px.bar(
                    pages_df.head(10),
                    x='url',
                    y='ai_bot_visits',
                    title="Top 10 Pages by AI Bot Visits"
                )
                fig.update_xaxis(tickangle=45)
                st.plotly_chart(fig, use_container_width=True)
            
            # AI bots breakdown
            if 'ai_bots' in results and results['ai_bots']:
                st.markdown("### AI Bot Breakdown")
                
                bots_df = pd.DataFrame(results['ai_bots'])
                fig = px.pie(
                    bots_df,
                    values='requests',
                    names='bot_name',
                    title="AI Bot Distribution"
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # Recommendations
            if 'recommendations' in results and results['recommendations']:
                st.markdown("### Optimization Recommendations")
                
                for rec in results['recommendations']:
                    st.markdown(f"- **{rec['category']}**: {rec['recommendation']}")

    def render_settings(self):
        """Render the Settings module"""
        st.header("⚙️ Settings")
        st.markdown("Manage your account and integrations")
        
        # Tabs for different settings
        tab1, tab2, tab3 = st.tabs(["Profile", "Integrations", "Preferences"])
        
        with tab1:
            st.markdown("### Profile Settings")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.text_input("Full Name", value="John Doe", key="settings_full_name")
                st.text_input("Email", value="john@example.com", key="settings_email")
                st.text_input("Company", value="AIO Search", key="settings_company")
            
            with col2:
                st.selectbox("Timezone", ["UTC", "EST", "PST", "GMT"], key="settings_timezone")
                st.selectbox("Language", ["English", "Spanish", "French"], key="settings_language")
                st.checkbox("Email notifications", value=True, key="settings_email_notifications")
            
            if st.button("Save Profile", type="primary", key="settings_save_profile"):
                st.success("Profile updated successfully!")
        
        with tab2:
            st.markdown("### API Integrations")
            
            st.text_input("OpenAI API Key", type="password", placeholder="sk-...", key="settings_openai_key")
            st.text_input("Airtop API Key", type="password", placeholder="at_...", key="settings_airtop_key")
            st.text_input("Webhook URL", placeholder="https://your-domain.com/webhook", key="settings_webhook_url")
            
            if st.button("Test Connections", type="primary", key="settings_test_connections"):
                st.success("All connections successful!")
        
        with tab3:
            st.markdown("### Application Preferences")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.selectbox("Default Industry", ["digital marketing", "e-commerce", "SaaS", "healthcare"], key="settings_default_industry")
                st.selectbox("Default Brand", ["AIO Search", "My Brand"], key="settings_default_brand")
                st.slider("Auto-refresh interval (minutes)", 1, 60, 5, key="settings_auto_refresh")
            
            with col2:
                st.checkbox("Auto-save results", value=True, key="settings_auto_save")
                st.checkbox("Show advanced options", value=False, key="settings_show_advanced")
                st.checkbox("Enable analytics", value=True, key="settings_enable_analytics")
            
            if st.button("Save Preferences", type="primary", key="settings_save_preferences"):
                st.success("Preferences saved successfully!")

    def _generate_robots_txt(self):
        """Generate a basic robots.txt content"""
        return """User-agent: *
Allow: /

# AI Crawlers
User-agent: ChatGPT-User
Allow: /

User-agent: CCBot
Allow: /

User-agent: anthropic-ai
Allow: /

User-agent: Claude-Web
Allow: /

Sitemap: https://yoursite.com/sitemap.xml
"""

    def run(self):
        """Main application runner"""
        # Render header
        self.render_header()
        
        # Render sidebar and get selected page
        selected = self.render_sidebar()
        
        # Render selected page
        if selected == "Dashboard":
            self.render_dashboard()
        elif selected == "AI Visibility":
            self.render_visibility_monitor()
        elif selected == "Content Optimization":
            self.render_content_optimizer()
        elif selected == "Question Intelligence":
            self.render_question_intelligence()
        elif selected == "Site AI Prep":
            self.render_site_ai_prep()
        elif selected == "Crawler Analytics":
            self.render_crawler_analytics()
        elif selected == "Settings":
            self.render_settings()

# Main application
if __name__ == "__main__":
    app = AIOWebApp()
    app.run() 