# 🎯 AIO Search Tool - Answer Engine Optimization Backend

**The complete backend for AI visibility monitoring and optimization.**

Direct competitor to Profound with 4x more features at competitive pricing.

## 🏆 Competitive Advantages

| **Feature** | **Profound** | **AIO Search Tool** |
|-------------|--------------|-------------------|
| **AI Monitoring** | ✅ | ✅ |
| **Content Optimization** | ❌ | ✅ **ADVANTAGE** |
| **Question Prediction** | ❌ | ✅ **ADVANTAGE** |
| **Site AI Preparation** | ❌ | ✅ **ADVANTAGE** |
| **Real Browser Data** | ✅ | ✅ (Airtop) |
| **Pricing** | Enterprise $$$$$ | Competitive $$ |

**Positioning**: *"Don't just track AI mentions - dominate them"*

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Set Environment Variables
```bash
# .env
AIRTOP_API_KEY=your_airtop_key_here
```

### 3. Start the API Server
```bash
python start_api.py
```

**Server runs on**: http://localhost:8000  
**API Documentation**: http://localhost:8000/docs

## 📡 API Endpoints

### 🎯 Content Optimizer
**POST** `/api/content/optimize`
Transform content for better AI citation and visibility.

```json
{
  "content": "Your content here...",
  "brand_name": "Your Brand",
  "keywords": ["keyword1", "keyword2"],
  "industry": "technology"
}
```

### 👁️ AI Visibility Monitor  
**POST** `/api/visibility/check`
Monitor brand presence across AI search engines using real browser automation.

```json
{
  "brand_name": "Your Brand",
  "competitors": ["Competitor 1", "Competitor 2"],
  "industry": "technology",
  "categories": ["AI tools"],
  "platforms": ["perplexity", "chatgpt"]
}
```

### 🧠 Question Intelligence
**POST** `/api/questions/map`
Generate and cluster questions users might ask about your brand.

```json
{
  "topics": ["AI optimization", "content strategy"],
  "industry": "digital marketing", 
  "brand_name": "Your Brand",
  "competitors": ["Comp1", "Comp2"],
  "product_features": ["feature1", "feature2"],
  "num_questions": 50
}
```

### 🗺️ AI Sitemap Generator
**POST** `/api/sitemap/generate`
Create AI-optimized sitemaps and discovery files.

```json
{
  "url": "https://yourdomain.com",
  "brand_name": "Your Brand",
  "description": "Brand description",
  "industry": "technology",
  "tagline": "Your tagline",
  "value_proposition": "Your value prop",
  "differentiators": ["diff1", "diff2"],
  "max_pages": 100
}
```

### 🤖 AI Crawler Analytics
**POST** `/api/crawler/analyze`
Analyze server logs to track AI crawler visits and optimize for AI visibility.

```json
{
  "log_file_path": "/var/log/apache2/access.log",
  "days_back": 30,
  "generate_report": true
}
```

**GET** `/api/crawler/robots-txt`
Generate AI-optimized robots.txt for better crawler accessibility.

## 🔧 Core Features

### ✨ Content Optimizer (`content_optimizer.py`)
- AI-friendly content transformation
- Semantic clarity scoring
- Q&A structure optimization
- Citation potential analysis

### 🤖 AI Visibility Monitor (`airtop_integration.py`)
- Real browser automation via Airtop
- Multi-platform monitoring (ChatGPT, Perplexity, Copilot, Google AI)
- Sentiment analysis
- Citation tracking
- Competitor benchmarking

### 🎯 Question Intelligence (`question_intent_mapper.py`)
- Predictive question generation
- Intent clustering and analysis
- Content gap identification
- User behavior prediction

### 🗺️ Site Preparation (`ai_sitemap_generator.py`)
- AI-optimized sitemap generation
- `site-ai.yaml` creation
- `llms.txt` generation
- AI discoverability analysis

### 🤖 AI Crawler Analytics (`ai_crawler_analytics.py`)
- Track AI bot visits to your website
- Analyze crawler behavior patterns
- Identify most valuable pages for AI
- Generate optimization recommendations

## 🏗️ Architecture

```
AIO Search Tool Backend
├── 🌐 FastAPI Server (api_server.py)
├── 🎯 Content Optimizer (content_optimizer.py)
├── 🤖 Airtop Integration (airtop_integration.py)
├── 🧠 Question Mapper (question_intent_mapper.py)
├── 🗺️ Sitemap Generator (ai_sitemap_generator.py)
├── 🤖 Crawler Analytics (ai_crawler_analytics.py)
└── 📚 Documentation & Guides
```

## 🚀 Production Deployment

### Quick Deploy Options:
- **Railway**: `railway init && railway add`
- **Render**: Connect GitHub → Deploy
- **Docker**: Use provided Dockerfiles

Deploy this FastAPI backend to any cloud provider that supports Python applications.

## 📊 Market Opportunity

**Profound charges enterprise prices for monitoring-only.**  
**We offer 4x the features at competitive pricing.**

### Suggested Pricing:
- **Starter**: $99/month
- **Professional**: $299/month  
- **Enterprise**: $599/month

## 🎯 Integration Examples

### Python
```python
import requests

# Optimize content
response = requests.post('http://localhost:8000/api/content/optimize', json={
    'content': 'Your content...',
    'brand_name': 'Your Brand',
    'keywords': ['AI', 'optimization'],
    'industry': 'technology'
})

print(response.json())
```

### cURL
```bash
curl -X POST "http://localhost:8000/api/visibility/check" \
  -H "Content-Type: application/json" \
  -d '{
    "brand_name": "Your Brand",
    "competitors": ["Competitor"],
    "industry": "technology",
    "categories": ["AI tools"],
    "platforms": ["perplexity"]
  }'
```

### JavaScript/Node.js
```javascript
const response = await fetch('http://localhost:8000/api/questions/map', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    topics: ['AI optimization'],
    industry: 'technology',
    brand_name: 'Your Brand',
    competitors: ['Competitor'],
    product_features: ['feature1'],
    num_questions: 25
  })
});

const data = await response.json();
console.log(data);
```

## 🔒 Environment Variables

```bash
# Required
AIRTOP_API_KEY=your_airtop_key

# Optional (Production)
DATABASE_URL=postgresql://...
REDIS_URL=redis://...
CORS_ORIGINS=https://yourdomain.com
```

## 📈 Performance

- **Startup Time**: ~10 seconds
- **Content Optimization**: ~2-5 seconds
- **Visibility Check**: ~30-60 seconds (real browser automation)
- **Question Generation**: ~10-20 seconds
- **Sitemap Generation**: ~20-40 seconds

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

## 📄 License

This project is proprietary. All rights reserved.

## 🎯 Next Steps

1. **Deploy to production** using any cloud provider (Railway, Render, AWS, etc.)
2. **Integrate with your existing tools** via the REST API
3. **Set up monitoring** and analytics
4. **Scale** based on usage patterns

**You now have a complete Profound competitor backend!** 🚀
