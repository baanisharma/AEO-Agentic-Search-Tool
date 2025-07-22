# AIO Search Tool - Web Interface

A comprehensive AI visibility and optimization platform with a modern web interface built with Streamlit.

## 🚀 Quick Start

### Option 1: Using the startup script (Recommended)
```bash
python run_web_app.py
```

### Option 2: Direct Streamlit command
```bash
streamlit run web_app.py
```

The application will automatically open in your default browser at `http://localhost:8501`.

## 📋 Prerequisites

- Python 3.8 or higher
- All dependencies from `requirements.txt`
- API keys for external services (optional but recommended)

## 🏗️ Architecture

The web interface is built with a modular architecture that integrates all existing AIO Search Tool modules:

### Core Modules
1. **Dashboard** - Overview and key metrics
2. **AI Visibility Monitor** - Monitor brand presence across AI search engines
3. **Content Optimization** - Transform content for better AI citation
4. **Question Intelligence** - Predict and analyze user questions
5. **Site AI Preparation** - Generate AI-optimized site files
6. **Crawler Analytics** - Analyze AI bot visits and behavior
7. **Settings** - Manage account and integrations

### Design System
- **Colors**: Deep blue (#1B263B), teal (#21D4FD), violet (#6C63FF)
- **Typography**: Inter/Roboto
- **Layout**: Responsive design with sidebar navigation
- **Components**: Cards, charts, tables, alerts

## 🎯 Features by Module

### 📊 Dashboard
- **Key Metrics**: Brand mentions, sentiment score, share of voice, optimized pages
- **Trend Charts**: Visibility trends over time, competitor comparison
- **Alerts**: Real-time notifications for important changes
- **Quick Stats**: Sidebar with current performance indicators

### 👁️ AI Visibility Monitor
- **Multi-Platform Support**: Perplexity, ChatGPT, Copilot, Google AI Overviews
- **Query Generation**: Automatic query generation based on industry and categories
- **Results Analysis**: Sentiment analysis, citation tracking, competitor benchmarking
- **Export Options**: CSV download with detailed results

**Usage:**
1. Enter brand name and industry
2. Add competitors and product categories
3. Select AI platforms to monitor
4. Generate and run queries
5. View results and download reports

### ✏️ Content Optimization
- **Input Methods**: Text input or file upload (TXT, MD, DOCX)
- **Optimization Scores**: Semantic clarity, citation potential, Q&A structure
- **Before/After Comparison**: Side-by-side content comparison
- **Change Tracking**: Detailed breakdown of optimizations made

**Usage:**
1. Input content via text or file upload
2. Configure brand name, industry, and keywords
3. Run optimization
4. Review scores and changes
5. Download optimized content

### ❓ Question Intelligence
- **Question Generation**: AI-powered question prediction
- **Cluster Analysis**: Group questions by intent and topic
- **Content Gap Analysis**: Identify high-volume questions without content coverage
- **Visualization**: Interactive charts and graphs

**Usage:**
1. Define topics, industry, and brand
2. Add competitors and product features
3. Generate questions
4. Analyze clusters and coverage gaps
5. Export question lists for content planning

### 🌐 Site AI Preparation
- **AI Sitemap Generation**: Create `site-ai.yaml` files
- **LLMs.txt Creation**: Generate AI crawler instructions
- **Robots.txt Optimization**: AI-optimized robots.txt files
- **Site Structure Analysis**: Visualize page hierarchy and AI scores

**Usage:**
1. Enter website URL and brand information
2. Configure brand details and differentiators
3. Generate AI site files
4. Download generated files
5. Review site structure and scores

### 🕷️ Crawler Analytics
- **Log Analysis**: Upload and analyze server access logs
- **AI Bot Detection**: Identify and track AI crawler visits
- **Activity Timeline**: Visualize bot activity over time
- **Optimization Recommendations**: Technical suggestions for better crawling

**Usage:**
1. Upload server access log file
2. Configure analysis parameters
3. Run analysis
4. Review bot activity and recommendations
5. Download analysis reports

### ⚙️ Settings
- **Profile Management**: User information and preferences
- **API Integrations**: Manage external service connections
- **Application Preferences**: Customize default settings and behavior

## 🎨 UI Components

### Navigation
- **Sidebar**: Collapsible navigation with icons and labels
- **Breadcrumbs**: Clear navigation hierarchy
- **Quick Actions**: Common tasks accessible from sidebar

### Data Visualization
- **Charts**: Line charts, bar charts, pie charts, scatter plots
- **Tables**: Sortable, filterable data tables
- **Metrics**: Key performance indicators with trend indicators
- **Alerts**: Color-coded notification banners

### Forms and Inputs
- **Text Inputs**: Brand names, URLs, descriptions
- **File Uploads**: Support for various file formats
- **Multi-select**: Platform and category selection
- **Sliders**: Numeric parameter configuration

## 🔧 Configuration

### Environment Variables
```bash
export AIO_OUTPUT_DIR="/path/to/output"
export AIO_LOG_DIR="/path/to/logs"
export AIO_TEMP_DIR="/path/to/temp"
```

### API Keys (Optional)
- OpenAI API Key: For advanced content optimization
- Airtop API Key: For AI visibility monitoring
- Webhook URL: For external integrations

## 📊 Data Flow

1. **Input**: User provides configuration and data
2. **Processing**: Backend modules process the data
3. **Analysis**: AI algorithms analyze and optimize
4. **Visualization**: Results are displayed in charts and tables
5. **Export**: Users can download results and reports

## 🚀 Deployment

### Local Development
```bash
# Install dependencies
pip install -r requirements.txt

# Run the application
python run_web_app.py
```

### Production Deployment
```bash
# Using Streamlit Cloud
streamlit run web_app.py --server.port 8501 --server.address 0.0.0.0

# Using Docker
docker build -t aio-search-tool .
docker run -p 8501:8501 aio-search-tool
```

## 📈 Performance

- **Real-time Updates**: Live data refresh and notifications
- **Caching**: Session state management for better performance
- **Async Processing**: Non-blocking operations for long-running tasks
- **Responsive Design**: Works on desktop, tablet, and mobile

## 🔒 Security

- **Input Validation**: All user inputs are validated
- **File Upload Security**: Secure file handling and validation
- **API Key Protection**: Secure storage of sensitive credentials
- **Session Management**: Secure session state handling

## 🐛 Troubleshooting

### Common Issues

1. **Application won't start**
   - Check Python version (3.8+ required)
   - Install missing dependencies: `pip install -r requirements.txt`
   - Check port availability (default: 8501)

2. **Module import errors**
   - Ensure all backend modules are in the same directory
   - Check file permissions
   - Verify Python path

3. **API errors**
   - Verify API keys are correctly configured
   - Check network connectivity
   - Review API rate limits

4. **Performance issues**
   - Reduce the number of queries or pages to crawl
   - Check system resources
   - Use smaller file uploads

### Logs and Debugging
- Application logs are stored in the `logs/` directory
- Enable debug mode by setting `STREAMLIT_DEBUG=1`
- Check browser console for frontend errors

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Support

For support and questions:
- Check the troubleshooting section
- Review the existing issues
- Create a new issue with detailed information

## 🔄 Updates

The web interface is regularly updated with:
- New features and modules
- UI/UX improvements
- Performance optimizations
- Security enhancements

Stay updated by checking the repository regularly.

## 🔄 Updates

The web interface is regularly updated with:
- New features and modules
- UI/UX improvements
- Performance optimizations
- Security enhancements

Stay updated by checking the repository regularly. 