#!/usr/bin/env python3
"""
Model Context Protocol (MCP) Server for Financial NLP
Enables standardized LLM integration with modular tools for financial data parsing.
"""

import asyncio
import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass
from enum import Enum

import aiohttp
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn
import re
import yfinance as yf
from textblob import TextBlob
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

# Download required NLTK data
try:
    nltk.download('vader_lexicon', quiet=True)
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
except:
    pass

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="MCP Financial NLP Server",
    description="Model Context Protocol server for financial data analysis",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Data Models
class ToolType(str, Enum):
    SUMMARIZATION = "summarization"
    CLASSIFICATION = "classification"
    EXTRACTION = "extraction"
    SENTIMENT = "sentiment"
    ENTITY_RECOGNITION = "entity_recognition"

class DataFormat(str, Enum):
    TABULAR = "tabular"
    UNSTRUCTURED = "unstructured"
    JSON = "json"
    TEXT = "text"

@dataclass
class MCPRequest(BaseModel):
    tool_type: ToolType
    data_format: DataFormat
    input_data: Union[str, Dict, List]
    parameters: Optional[Dict[str, Any]] = {}
    context: Optional[str] = ""

@dataclass
class MCPResponse(BaseModel):
    success: bool
    tool_type: str
    result: Any
    metadata: Dict[str, Any]
    timestamp: str

class FinancialNLPProcessor:
    """Core processor for financial NLP tasks"""
    
    def __init__(self):
        self.sentiment_analyzer = SentimentIntensityAnalyzer()
        self.financial_keywords = {
            'positive': ['profit', 'growth', 'increase', 'bullish', 'gain', 'revenue', 'earnings', 'up'],
            'negative': ['loss', 'decline', 'decrease', 'bearish', 'drop', 'deficit', 'down', 'bankruptcy'],
            'neutral': ['report', 'announcement', 'filing', 'statement', 'update', 'forecast']
        }
    
    async def summarize_financial_text(self, text: str, max_length: int = 150) -> Dict[str, Any]:
        """Summarize financial text using extractive summarization"""
        sentences = text.split('.')
        sentences = [s.strip() for s in sentences if len(s.strip()) > 10]
        
        # Score sentences based on financial keywords and position
        scored_sentences = []
        for i, sentence in enumerate(sentences):
            score = 0
            sentence_lower = sentence.lower()
            
            # Financial keyword scoring
            for category, keywords in self.financial_keywords.items():
                for keyword in keywords:
                    if keyword in sentence_lower:
                        score += 2 if category in ['positive', 'negative'] else 1
            
            # Position scoring (earlier sentences get higher score)
            position_score = max(0, 3 - i * 0.1)
            score += position_score
            
            # Length penalty for very short/long sentences
            length_penalty = abs(len(sentence) - 100) / 1000
            score -= length_penalty
            
            scored_sentences.append((sentence, score))
        
        # Select top sentences
        scored_sentences.sort(key=lambda x: x[1], reverse=True)
        num_sentences = min(3, len(scored_sentences))
        summary_sentences = [s[0] for s in scored_sentences[:num_sentences]]
        
        summary = '. '.join(summary_sentences)
        if len(summary) > max_length:
            summary = summary[:max_length] + "..."
        
        return {
            'summary': summary,
            'key_sentences': len(summary_sentences),
            'confidence': sum(s[1] for s in scored_sentences[:num_sentences]) / num_sentences if num_sentences > 0 else 0
        }
    
    async def classify_financial_text(self, text: str) -> Dict[str, Any]:
        """Classify financial text into categories"""
        text_lower = text.lower()
        
        categories = {
            'earnings_report': ['earnings', 'quarterly', 'revenue', 'eps', 'profit'],
            'market_analysis': ['market', 'trend', 'analysis', 'forecast', 'outlook'],
            'company_news': ['company', 'merger', 'acquisition', 'ceo', 'management'],
            'regulatory': ['sec', 'regulation', 'compliance', 'filing', 'legal'],
            'economic_indicator': ['gdp', 'inflation', 'unemployment', 'fed', 'interest rate']
        }
        
        scores = {}
        for category, keywords in categories.items():
            score = sum(1 for keyword in keywords if keyword in text_lower)
            scores[category] = score / len(keywords)  # Normalize
        
        # Get the highest scoring category
        primary_category = max(scores, key=scores.get)
        confidence = scores[primary_category]
        
        return {
            'primary_category': primary_category,
            'confidence': confidence,
            'all_scores': scores,
            'is_financial': any(score > 0.1 for score in scores.values())
        }
    
    async def extract_financial_entities(self, text: str) -> Dict[str, Any]:
        """Extract financial entities from text"""
        entities = {
            'companies': [],
            'currencies': [],
            'amounts': [],
            'dates': [],
            'percentages': [],
            'stock_symbols': []
        }
        
        # Extract stock symbols (basic pattern)
        stock_pattern = r'\b[A-Z]{1,5}\b'
        potential_stocks = re.findall(stock_pattern, text)
        entities['stock_symbols'] = list(set(potential_stocks))
        
        # Extract monetary amounts
        amount_pattern = r'\$[\d,]+\.?\d*[MBK]?|\d+\.?\d*\s*(million|billion|thousand)'
        amounts = re.findall(amount_pattern, text, re.IGNORECASE)
        entities['amounts'] = amounts
        
        # Extract percentages
        percentage_pattern = r'\d+\.?\d*\s*%'
        percentages = re.findall(percentage_pattern, text)
        entities['percentages'] = percentages
        
        # Extract dates (basic pattern)
        date_pattern = r'\b\d{1,2}[-/]\d{1,2}[-/]\d{2,4}\b|\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2},?\s+\d{4}\b'
        dates = re.findall(date_pattern, text, re.IGNORECASE)
        entities['dates'] = dates
        
        # Extract currency mentions
        currencies = ['USD', 'EUR', 'GBP', 'JPY', 'CAD', 'AUD', 'CHF']
        found_currencies = [curr for curr in currencies if curr in text.upper()]
        entities['currencies'] = found_currencies
        
        return entities
    
    async def analyze_sentiment(self, text: str) -> Dict[str, Any]:
        """Analyze sentiment with financial context"""
        # VADER sentiment analysis
        vader_scores = self.sentiment_analyzer.polarity_scores(text)
        
        # TextBlob sentiment
        blob = TextBlob(text)
        textblob_sentiment = blob.sentiment
        
        # Financial-specific sentiment scoring
        financial_score = 0
        text_lower = text.lower()
        
        for keyword in self.financial_keywords['positive']:
            financial_score += text_lower.count(keyword) * 1
        for keyword in self.financial_keywords['negative']:
            financial_score -= text_lower.count(keyword) * 1
        
        # Normalize financial score
        word_count = len(text.split())
        financial_score = financial_score / max(word_count, 1) if word_count > 0 else 0
        
        # Combined sentiment
        combined_sentiment = (vader_scores['compound'] + textblob_sentiment.polarity + financial_score) / 3
        
        # Determine overall sentiment
        if combined_sentiment > 0.1:
            overall = 'positive'
        elif combined_sentiment < -0.1:
            overall = 'negative'
        else:
            overall = 'neutral'
        
        return {
            'overall_sentiment': overall,
            'confidence': abs(combined_sentiment),
            'scores': {
                'vader': vader_scores,
                'textblob': {
                    'polarity': textblob_sentiment.polarity,
                    'subjectivity': textblob_sentiment.subjectivity
                },
                'financial_context': financial_score,
                'combined': combined_sentiment
            }
        }

# Initialize processor
processor = FinancialNLPProcessor()

class ToolRouter:
    """Routes requests to appropriate tools based on MCP specifications"""
    
    @staticmethod
    async def route_request(request: MCPRequest) -> MCPResponse:
        """Route request to appropriate tool"""
        try:
            timestamp = datetime.now().isoformat()
            
            if request.tool_type == ToolType.SUMMARIZATION:
                result = await ToolRouter.handle_summarization(request)
            elif request.tool_type == ToolType.CLASSIFICATION:
                result = await ToolRouter.handle_classification(request)
            elif request.tool_type == ToolType.EXTRACTION:
                result = await ToolRouter.handle_extraction(request)
            elif request.tool_type == ToolType.SENTIMENT:
                result = await ToolRouter.handle_sentiment(request)
            else:
                raise ValueError(f"Unsupported tool type: {request.tool_type}")
            
            return MCPResponse(
                success=True,
                tool_type=request.tool_type.value,
                result=result,
                metadata={
                    'data_format': request.data_format.value,
                    'processing_time': datetime.now().isoformat(),
                    'parameters_used': request.parameters
                },
                timestamp=timestamp
            )
            
        except Exception as e:
            logger.error(f"Error processing request: {str(e)}")
            return MCPResponse(
                success=False,
                tool_type=request.tool_type.value,
                result={'error': str(e)},
                metadata={'error_type': type(e).__name__},
                timestamp=datetime.now().isoformat()
            )
    
    @staticmethod
    async def handle_summarization(request: MCPRequest) -> Dict[str, Any]:
        """Handle summarization requests"""
        text = ToolRouter.extract_text(request.input_data, request.data_format)
        max_length = request.parameters.get('max_length', 150)
        
        result = await processor.summarize_financial_text(text, max_length)
        return result
    
    @staticmethod
    async def handle_classification(request: MCPRequest) -> Dict[str, Any]:
        """Handle classification requests"""
        text = ToolRouter.extract_text(request.input_data, request.data_format)
        result = await processor.classify_financial_text(text)
        return result
    
    @staticmethod
    async def handle_extraction(request: MCPRequest) -> Dict[str, Any]:
        """Handle entity extraction requests"""
        text = ToolRouter.extract_text(request.input_data, request.data_format)
        result = await processor.extract_financial_entities(text)
        return result
    
    @staticmethod
    async def handle_sentiment(request: MCPRequest) -> Dict[str, Any]:
        """Handle sentiment analysis requests"""
        text = ToolRouter.extract_text(request.input_data, request.data_format)
        result = await processor.analyze_sentiment(text)
        return result
    
    @staticmethod
    def extract_text(input_data: Union[str, Dict, List], data_format: DataFormat) -> str:
        """Extract text from various input formats"""
        if data_format == DataFormat.TEXT:
            return str(input_data)
        elif data_format == DataFormat.JSON:
            if isinstance(input_data, dict):
                # Look for common text fields
                text_fields = ['text', 'content', 'description', 'body', 'message']
                for field in text_fields:
                    if field in input_data:
                        return str(input_data[field])
                # If no common fields, stringify the whole object
                return json.dumps(input_data)
            return str(input_data)
        elif data_format == DataFormat.TABULAR:
            if isinstance(input_data, list):
                # Assume list of rows
                return ' '.join([str(item) for item in input_data])
            elif isinstance(input_data, dict):
                return ' '.join([str(value) for value in input_data.values()])
            return str(input_data)
        else:
            return str(input_data)

# API Endpoints

@app.get("/")
async def root():
    """Root endpoint with server information"""
    return {
        "name": "MCP Financial NLP Server",
        "version": "1.0.0",
        "protocol": "Model Context Protocol",
        "supported_tools": [tool.value for tool in ToolType],
        "supported_formats": [fmt.value for fmt in DataFormat]
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

@app.post("/mcp/process")
async def process_mcp_request(request: MCPRequest):
    """Main MCP processing endpoint"""
    logger.info(f"Processing MCP request: {request.tool_type}")
    response = await ToolRouter.route_request(request)
    return response

@app.get("/tools")
async def list_tools():
    """List available tools and their capabilities"""
    return {
        "tools": {
            "summarization": {
                "description": "Summarize financial text content",
                "parameters": ["max_length"],
                "supported_formats": ["text", "json", "tabular"]
            },
            "classification": {
                "description": "Classify financial text into categories",
                "parameters": [],
                "supported_formats": ["text", "json", "tabular"]
            },
            "extraction": {
                "description": "Extract financial entities from text",
                "parameters": [],
                "supported_formats": ["text", "json", "tabular"]
            },
            "sentiment": {
                "description": "Analyze sentiment with financial context",
                "parameters": [],
                "supported_formats": ["text", "json", "tabular"]
            }
        }
    }

@app.get("/market/quote/{symbol}")
async def get_market_quote(symbol: str):
    """Get real-time market data for analysis"""
    try:
        ticker = yf.Ticker(symbol)
        info = ticker.info
        hist = ticker.history(period="1d")
        
        if hist.empty:
            raise HTTPException(status_code=404, detail="Symbol not found")
        
        current_price = hist['Close'].iloc[-1]
        
        return {
            "symbol": symbol.upper(),
            "current_price": float(current_price),
            "company_name": info.get('longName', 'N/A'),
            "market_cap": info.get('marketCap', 0),
            "pe_ratio": info.get('trailingPE', 0),
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Example usage endpoints for testing

@app.post("/examples/summarize")
async def example_summarize(text: str):
    """Example summarization endpoint"""
    request = MCPRequest(
        tool_type=ToolType.SUMMARIZATION,
        data_format=DataFormat.TEXT,
        input_data=text,
        parameters={"max_length": 150}
    )
    return await ToolRouter.route_request(request)

@app.post("/examples/classify")
async def example_classify(text: str):
    """Example classification endpoint"""
    request = MCPRequest(
        tool_type=ToolType.CLASSIFICATION,
        data_format=DataFormat.TEXT,
        input_data=text
    )
    return await ToolRouter.route_request(request)

@app.post("/examples/extract")
async def example_extract(text: str):
    """Example entity extraction endpoint"""
    request = MCPRequest(
        tool_type=ToolType.EXTRACTION,
        data_format=DataFormat.TEXT,
        input_data=text
    )
    return await ToolRouter.route_request(request)

@app.post("/examples/sentiment")
async def example_sentiment(text: str):
    """Example sentiment analysis endpoint"""
    request = MCPRequest(
        tool_type=ToolType.SENTIMENT,
        data_format=DataFormat.TEXT,
        input_data=text
    )
    return await ToolRouter.route_request(request)

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
