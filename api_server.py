#!/usr/bin/env python3
"""
AIO Search Tool API Server
=========================

FastAPI server that exposes all AIO Search Tool modules as REST APIs
for the Lovable frontend to consume.

Endpoints:
- /api/content/optimize - Content Optimizer
- /api/visibility/check - AI Visibility Checker (Airtop)
- /api/questions/map - Question & Intent Mapper
- /api/sitemap/generate - AI Sitemap Generator
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import asyncio
import os
import json
import uuid
from datetime import datetime
import logging

# Import your modules
from content_optimizer import ContentOptimizer
from airtop_integration import AirtopLLMVisibility
from question_intent_mapper import QuestionIntentMapper
from ai_sitemap_generator import AISitemapGenerator
from ai_crawler_analytics import AICrawlerAnalytics

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI
app = FastAPI(
    title="AIO Search Tool Backend API",
    description="AI Optimization Search Tool - Backend APIs for content optimization and visibility analysis",
    version="1.0.0"
)

# Enable CORS for any frontend that might connect
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure with your domain in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize modules
content_optimizer = ContentOptimizer()
visibility_checker = AirtopLLMVisibility()
question_mapper = QuestionIntentMapper()
sitemap_generator = AISitemapGenerator()
crawler_analytics = AICrawlerAnalytics()

# Create output directory
os.makedirs('aio_output', exist_ok=True)

# Pydantic models for API requests

class ContentOptimizeRequest(BaseModel):
    content: str
    brand_name: str
    keywords: List[str]
    industry: str

class VisibilityCheckRequest(BaseModel):
    brand_name: str
    competitors: List[str]
    industry: str
    categories: List[str]
    queries: Optional[List[str]] = None
    platforms: Optional[List[str]] = ["perplexity"]

class QuestionMapRequest(BaseModel):
    topics: List[str]
    industry: str
    brand_name: str
    competitors: List[str]
    product_features: List[str]
    num_questions: Optional[int] = 50

class SitemapGenerateRequest(BaseModel):
    url: str
    brand_name: str
    description: str
    industry: str
    tagline: str
    value_proposition: str
    differentiators: List[str]
    max_pages: Optional[int] = 100

class CrawlerAnalyticsRequest(BaseModel):
    log_file_path: str
    days_back: Optional[int] = 30
    generate_report: Optional[bool] = True

# API Endpoints

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "message": "AIO Search Tool API Server",
        "status": "running",
        "modules": {
            "content_optimizer": "ready",
            "ai_visibility_checker": "ready", 
            "question_mapper": "ready",
            "sitemap_generator": "ready",
            "crawler_analytics": "ready"
        },
        "timestamp": datetime.now().isoformat()
    }

@app.get("/api/health")
async def health_check():
    """Detailed health check for all modules"""
    health_status = {
        "api_server": "healthy",
        "content_optimizer": "ready",
        "question_mapper": "ready", 
        "sitemap_generator": "ready",
        "crawler_analytics": "ready",
        "visibility_checker": "checking..."
    }
    
    # Check if Airtop is configured
    try:
        airtop_key = os.getenv('AIRTOP_API_KEY')
        if airtop_key:
            health_status["visibility_checker"] = "ready"
        else:
            health_status["visibility_checker"] = "needs_api_key"
    except Exception as e:
        health_status["visibility_checker"] = f"error: {str(e)}"
    
    return health_status

# Content Optimizer API
@app.post("/api/content/optimize")
async def optimize_content(request: ContentOptimizeRequest):
    """
    🎯 Content Optimizer API
    
    Optimizes content for better LLM citation and AI visibility
    """
    try:
        logger.info(f"Optimizing content for brand: {request.brand_name}")
        
        # Run content optimization
        optimized_content, scores, changes = content_optimizer.optimize_content(
            request.content,
            request.brand_name, 
            request.keywords,
            request.industry
        )
        
        # Generate unique ID for this optimization
        optimization_id = str(uuid.uuid4())
        
        # Save result to file
        output_file = f"aio_output/optimization_{optimization_id}.md"
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(optimized_content)
        
        return {
            "success": True,
            "optimization_id": optimization_id,
            "optimized_content": optimized_content,
            "scores": scores,
            "changes": changes,
            "output_file": output_file,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Content optimization error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# AI Visibility Checker API  
@app.post("/api/visibility/check")
async def check_visibility(request: VisibilityCheckRequest, background_tasks: BackgroundTasks):
    """
    🎯 AI Visibility Checker API
    
    Monitors brand presence across AI search engines using browser automation
    """
    try:
        logger.info(f"Starting AI visibility check for: {request.brand_name}")
        
        # Generate check ID
        check_id = str(uuid.uuid4())
        
        # Run visibility check asynchronously 
        result = await visibility_checker.run_visibility_check(
            brand_name=request.brand_name,
            competitors=request.competitors,
            queries=request.queries,
            platforms=request.platforms
        )
        
        # Save results
        results_file = f"aio_output/visibility_{check_id}.json"
        with open(results_file, 'w') as f:
            json.dump(result, f, indent=2)
        
        return {
            "success": result.get('success', False),
            "check_id": check_id,
            "results": result,
            "results_file": results_file,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Visibility check error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Question Mapper API
@app.post("/api/questions/map")
async def map_questions(request: QuestionMapRequest):
    """
    🎯 Question & Intent Mapper API
    
    Generates and clusters questions users might ask about your brand/industry
    """
    try:
        logger.info(f"Mapping questions for: {request.brand_name}")
        
        # Generate questions
        questions = question_mapper.generate_questions(
            request.topics,
            request.industry,
            request.brand_name,
            request.competitors,
            request.product_features,
            request.num_questions
        )
        
        # Cluster questions
        clusters = question_mapper.cluster_questions(questions)
        
        # Generate mapping ID
        mapping_id = str(uuid.uuid4())
        
        # Save results
        results_file = f"aio_output/questions_{mapping_id}.json"
        question_mapper.export_results(clusters, results_file)
        
        # Generate visualization (optional)
        try:
            question_mapper.visualize_clusters(questions, clusters)
            viz_file = f"aio_output/question_clusters_{mapping_id}.png"
        except Exception as viz_error:
            logger.warning(f"Visualization failed: {viz_error}")
            viz_file = None
        
        return {
            "success": True,
            "mapping_id": mapping_id,
            "questions": questions,
            "clusters": clusters,
            "results_file": results_file,
            "visualization_file": viz_file,
            "total_questions": len(questions),
            "total_clusters": len(clusters),
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Question mapping error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# AI Sitemap Generator API
@app.post("/api/sitemap/generate")
async def generate_sitemap(request: SitemapGenerateRequest):
    """
    🎯 AI Sitemap Generator API
    
    Creates AI-optimized sitemaps and files for better discoverability
    """
    try:
        logger.info(f"Generating AI sitemap for: {request.url}")
        
        # Prepare brand info
        brand_info = {
            "name": request.brand_name,
            "description": request.description,
            "url": request.url,
            "industry": request.industry,
            "tagline": request.tagline,
            "value_proposition": request.value_proposition,
            "key_differentiators": request.differentiators
        }
        
        # Crawl website
        pages = sitemap_generator.crawl_website(request.url, max_pages=request.max_pages)
        
        # Generate files
        generation_id = str(uuid.uuid4())
        
        # Generate site-ai.yaml
        yaml_file = f'aio_output/site-ai_{generation_id}.yaml'
        sitemap_generator.generate_site_ai_yaml(pages, brand_info, output_file=yaml_file)
        
        # Generate llms.txt
        txt_file = f'aio_output/llms_{generation_id}.txt'
        sitemap_generator.generate_llms_txt(pages, brand_info, output_file=txt_file)
        
        # Analyze site
        analysis = sitemap_generator.analyze_site_for_ai_optimization(pages)
        analysis_file = f'aio_output/analysis_{generation_id}.json'
        with open(analysis_file, 'w') as f:
            json.dump(analysis, f, indent=2)
        
        return {
            "success": True,
            "generation_id": generation_id,
            "pages_crawled": len(pages),
            "files_generated": {
                "site_ai_yaml": yaml_file,
                "llms_txt": txt_file,
                "analysis_json": analysis_file
            },
            "analysis": analysis,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Sitemap generation error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# AI Crawler Analytics API
@app.post("/api/crawler/analyze")
async def analyze_crawler_activity(request: CrawlerAnalyticsRequest):
    """
    🎯 AI Crawler Analytics API
    
    Analyzes server logs to track AI crawler activity and optimize for better AI visibility
    """
    try:
        logger.info(f"Analyzing AI crawler activity in: {request.log_file_path}")
        
        # Parse access logs for AI crawler activity
        results = crawler_analytics.parse_access_logs(
            request.log_file_path, 
            days_back=request.days_back
        )
        
        if "error" in results:
            raise HTTPException(status_code=400, detail=results["error"])
        
        # Generate analysis ID
        analysis_id = str(uuid.uuid4())
        
        # Save results
        analysis_file = f'aio_output/crawler_analysis_{analysis_id}.json'
        with open(analysis_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        response_data = {
            "success": True,
            "analysis_id": analysis_id,
            "results": results,
            "analysis_file": analysis_file,
            "timestamp": datetime.now().isoformat()
        }
        
        # Generate report if requested
        if request.generate_report:
            report_file = f'aio_output/crawler_report_{analysis_id}.md'
            report_content = crawler_analytics.generate_crawler_report(output_file=report_file)
            response_data["report_file"] = report_file
            response_data["report_preview"] = report_content[:500] + "..." if len(report_content) > 500 else report_content
        
        return response_data
        
    except Exception as e:
        logger.error(f"Crawler analytics error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/crawler/robots-txt")
async def generate_robots_txt(sitemap_url: Optional[str] = None):
    """
    🎯 Generate AI-Optimized robots.txt
    
    Creates robots.txt content optimized for AI crawler accessibility
    """
    try:
        robots_content = crawler_analytics.create_robots_txt_for_ai(sitemap_url)
        
        return {
            "success": True,
            "robots_txt": robots_content,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Robots.txt generation error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# File download endpoints
@app.get("/api/files/{filename}")
async def download_file(filename: str):
    """Download generated files"""
    file_path = f"aio_output/{filename}"
    if os.path.exists(file_path):
        return FileResponse(file_path)
    else:
        raise HTTPException(status_code=404, detail="File not found")

@app.get("/api/results/{result_id}")
async def get_results(result_id: str):
    """Get results by ID"""
    # Look for various file types
    possible_files = [
        f"aio_output/optimization_{result_id}.md",
        f"aio_output/visibility_{result_id}.json", 
        f"aio_output/questions_{result_id}.json",
        f"aio_output/analysis_{result_id}.json"
    ]
    
    for file_path in possible_files:
        if os.path.exists(file_path):
            if file_path.endswith('.json'):
                with open(file_path, 'r') as f:
                    return json.load(f)
            else:
                with open(file_path, 'r', encoding='utf-8') as f:
                    return {"content": f.read()}
    
    raise HTTPException(status_code=404, detail="Results not found")

# Demo/Testing endpoints
@app.get("/api/demo/content")
async def demo_content_optimizer():
    """Demo endpoint for content optimizer"""
    return {
        "demo_request": {
            "content": "AI content optimization helps improve search rankings...",
            "brand_name": "Demo Brand",
            "keywords": ["AI optimization", "content strategy"],
            "industry": "digital marketing"
        },
        "endpoint": "/api/content/optimize",
        "method": "POST"
    }

@app.get("/api/demo/visibility")
async def demo_visibility_checker():
    """Demo endpoint for visibility checker"""
    return {
        "demo_request": {
            "brand_name": "Demo Brand",
            "competitors": ["Competitor 1", "Competitor 2"],
            "industry": "technology",
            "categories": ["AI tools", "content optimization"],
            "platforms": ["perplexity"]
        },
        "endpoint": "/api/visibility/check", 
        "method": "POST",
        "note": "Requires AIRTOP_API_KEY environment variable"
    }

@app.get("/api/demo/crawler")
async def demo_crawler_analytics():
    """Demo endpoint for crawler analytics"""
    return {
        "demo_request": {
            "log_file_path": "/var/log/apache2/access.log",
            "days_back": 30,
            "generate_report": True
        },
        "endpoint": "/api/crawler/analyze",
        "method": "POST",
        "note": "Requires access to server log files"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 