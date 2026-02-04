#!/usr/bin/env python3
"""
Pipeline 3: Parallel Extraction with Reconciliation + High-Accuracy PPT Integration
(Server-Side Version with GPU Support)
"""

import sys
import os
import io
import json
import logging
import re
import shutil
import subprocess
import cv2
import numpy as np
import fitz  # PyMuPDF
from PIL import Image
import torch

import paddle
import paddle.inference

# --- UNIVERSAL PADDLE PATCH START ---
# Monkey-patch paddle.inference.create_predictor to force config safety globally
# This prevents "Illegal Instruction" crashes on CPUs by disabling AVX-512 based IR optimizations
try:
    _original_create_predictor = paddle.inference.create_predictor
except AttributeError:
    _original_create_predictor = None

def _patched_create_predictor(config):
    # Depending on version, config might be AnalysisConfig or Config
    try:
        if hasattr(config, "switch_ir_optim"):
            config.switch_ir_optim(False)
        if hasattr(config, "disable_mkldnn"):
            config.disable_mkldnn()
        if hasattr(config, "enable_mkldnn"):
            # Ensure it's not re-enabled
            pass 
    except Exception:
        pass
    
    if _original_create_predictor:
        return _original_create_predictor(config)
    return None

if _original_create_predictor:
    paddle.inference.create_predictor = _patched_create_predictor
# --- UNIVERSAL PADDLE PATCH END ---
import gc
from typing import List, Dict, Optional, Tuple, Any
from collections import OrderedDict
from datetime import datetime

# Force PaddleOCR to CPU to avoid CUDNN version mismatch with PyTorch
os.environ["FLAGS_use_gpu"] = "0"
os.environ["FLAGS_use_mkldnn"] = "0"
# Limit CPU instruction set to AVX2 to avoid Illegal Instruction
os.environ["DNNL_MAX_CPU_ISA"] = "AVX2"
os.environ["ONEDNN_MAX_CPU_ISA"] = "AVX2"

# Add project root to path (robust drop-in)
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir)) # pipelines/advanced/ -> pipelines/ -> root
if project_root not in sys.path:
    sys.path.append(project_root)

from models.llm_interface import LLMInterface
from utils.document_processor import DocumentProcessor
from config import Config

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Suppress warnings
import warnings
warnings.filterwarnings('ignore')

# ==============================================================================
# PROMPTS & CONFIGURATION (Prompts from PPT Extractor)
# ==============================================================================

MAIN_EXTRACTION_PROMPT = '''You are an expert data analyst. Analyze this presentation slide and extract ALL information.

CRITICAL: Extract EVERY number, percentage, currency value, and data point visible.

Return a JSON object with this EXACT structure:
{
    "slide_title": "exact title of the slide",
    "slide_type": "data/chart/table/mixed/title",
    "company_brand": "company name if visible",

    "key_highlights": [
        "bullet point 1 with exact numbers",
        "bullet point 2 with exact numbers"
    ],

    "metrics": {
        "metric_name": {"value": "exact value", "unit": "unit", "context": "what it represents"}
    },

    "bar_charts": [
        {
            "title": "chart title",
            "x_axis": "what x-axis shows",
            "y_axis": "what y-axis shows (with unit)",
            "data_points": {"label1": "value1", "label2": "value2"},
            "insight": "what the chart shows"
        }
    ],

    "line_charts": [
        {
            "title": "chart title",
            "data_series": {"year/period": "value"},
            "trend": "increasing/decreasing/stable",
            "insight": "what it shows"
        }
    ],

    "tables": [
        {
            "title": "table title",
            "headers": ["col1", "col2"],
            "rows": [["row1col1", "row1col2"], ["row2col1", "row2col2"]],
            "summary": "key takeaway from table"
        }
    ],

    "comparisons": [
        {"entity1": "name", "entity1_value": "value", "entity2": "name", "entity2_value": "value", "metric": "what is compared"}
    ],

    "growth_rates": [
        {"metric": "what is growing", "rate": "growth rate", "period": "time period", "from": "start value", "to": "end value"}
    ],

    "targets_projections": [
        {"target": "what", "value": "target value", "by_when": "deadline/year"}
    ],

    "country_data": {
        "country_name": {"metric": "value"}
    },

    "investment_opportunities": [
        {"sector": "sector name", "amount": "investment amount", "timeline": "by when"}
    ],

    "all_numerical_facts": [
        "India GDP growth: 6.5% in FY25",
        "Population: 1.46 billion"
    ],

    "sources": ["source 1", "source 2"]
}

IMPORTANT RULES:
1. Extract EVERY visible number, percentage, and currency value
2. Include units (%, GW, Bn, $, etc.)
3. Preserve exact values - do not round or estimate
4. For charts, extract ALL data points visible
5. For tables, extract ALL rows and columns
6. Return ONLY valid JSON, no other text'''

CHART_EXTRACTION_PROMPT = '''Extract ALL data from the charts in this slide.

For EACH chart visible, provide:
{
    "charts": [
        {
            "chart_number": 1,
            "chart_type": "bar/line/pie/area",
            "title": "exact chart title",
            "subtitle": "subtitle if any",
            "x_axis_label": "label",
            "y_axis_label": "label with unit",
            "data": {
                "category1": "value1",
                "category2": "value2"
            },
            "annotations": ["any text annotations on chart"],
            "insight": "what the chart communicates"
        }
    ]
}

Extract EVERY data point. Return ONLY JSON.'''

TABLE_EXTRACTION_PROMPT = '''Extract ALL data from tables in this slide.

For EACH table:
{
    "tables": [
        {
            "table_number": 1,
            "title": "table title",
            "headers": ["header1", "header2", "header3"],
            "data_rows": [
                ["row1_col1", "row1_col2", "row1_col3"],
                ["row2_col1", "row2_col2", "row2_col3"]
            ],
            "totals_row": ["total values if present"],
            "key_values": {"row_label": "important_value"}
        }
    ]
}

Include ALL rows and columns. Return ONLY JSON.'''

NUMBERS_VERIFICATION_PROMPT = '''List EVERY number visible in this slide.

Return JSON:
{
    "percentages": ["6.7%", "6.4%"],
    "currency_amounts": ["$500 Bn+", "$35 Tn"],
    "quantities": ["1.46 Bn", "186 bn"],
    "growth_rates": ["11% CAGR", "14x increase"],
    "years": ["FY25", "FY32", "2047"],
    "other_numbers": ["112 GW+", "648 ckms"]
}

Include EVERY number. Return ONLY JSON.'''

# ==============================================================================
# PPT EXTRACTOR CLASSES (from ppt_extractor_v6.py)
# ==============================================================================

class ComprehensiveOCRExtractor:
    """Extract ALL text, numbers, tables with maximum precision using PaddleOCR"""

    def __init__(self, ocr_engine, structure_engine):
        self.ocr = ocr_engine
        self.structure = structure_engine

    def extract(self, image: np.ndarray) -> Dict:
        """Complete OCR extraction with multiple passes"""

        result = {
            'text_blocks': [],
            'tables': [],
            'layout_regions': [],
            'all_text': [],
            'all_numbers': [],
            'percentages': [],
            'currency_values': [],
            'years': [],
            'raw_text': '',
            'confidence': 0.0
        }

        # Pass 1: Standard OCR
        try:
            ocr_result = self.ocr.ocr(image, cls=True)

            if ocr_result and ocr_result[0]:
                confidences = []
                for line in ocr_result[0]:
                    bbox = line[0]
                    text = line[1][0]
                    conf = float(line[1][1])

                    # Calculate position for sorting
                    center_y = sum(p[1] for p in bbox) / 4
                    center_x = sum(p[0] for p in bbox) / 4

                    result['text_blocks'].append({
                        'text': text,
                        'bbox': bbox,
                        'confidence': conf,
                        'center_x': center_x,
                        'center_y': center_y
                    })
                    result['all_text'].append(text)
                    confidences.append(conf)

                    # Extract ALL numbers
                    self._extract_numbers(text, result)

                result['confidence'] = sum(confidences) / len(confidences) if confidences else 0
        except Exception as e:
            logger.error(f"OCR Pass 1 error: {e}")

        # Pass 2: Structure analysis
        try:
            structure_result = self.structure(image)

            for item in structure_result:
                item_type = item.get('type', 'unknown')

                result['layout_regions'].append({
                    'type': item_type,
                    'bbox': item.get('bbox', []),
                    'score': item.get('score', 0.0)
                })

                if item_type == 'table':
                    table_data = self._parse_table(item)
                    if table_data:
                        result['tables'].append(table_data)
                        # Extract numbers from table
                        for row in table_data.get('all_data', []):
                            for cell in row:
                                self._extract_numbers(str(cell), result)
        except Exception as e:
            logger.error(f"Structure Pass error: {e}")

        # Sort text by position (top-to-bottom, left-to-right)
        result['text_blocks'].sort(key=lambda x: (x['center_y'], x['center_x']))
        result['raw_text'] = ' '.join([b['text'] for b in result['text_blocks']])

        # Remove duplicates from number lists
        result['all_numbers'] = list(OrderedDict.fromkeys(result['all_numbers']))
        result['percentages'] = list(OrderedDict.fromkeys(result['percentages']))
        result['currency_values'] = list(OrderedDict.fromkeys(result['currency_values']))
        result['years'] = list(OrderedDict.fromkeys(result['years']))

        return result

    def _extract_numbers(self, text: str, result: Dict):
        """Extract all numerical values from text"""

        # Percentages
        percentages = re.findall(r'[\d,]+\.?\d*\s*%', text)
        result['percentages'].extend(percentages)

        # Currency
        currencies = re.findall(r'\$[\d,]+\.?\d*\s*[BMKTn]*\+?', text, re.IGNORECASE)
        result['currency_values'].extend(currencies)

        # Years
        years = re.findall(r"FY\d{2}E?|20\d{2}|'\d{2}", text)
        result['years'].extend(years)

        # General numbers
        numbers = re.findall(r'[\d,]+\.?\d*\s*(?:bn|Bn|BN|mn|Mn|MN|GW|MW|MWh|ckms|x|Tn|TN)?\+?', text)
        result['all_numbers'].extend([n.strip() for n in numbers if n.strip()])

    def _parse_table(self, table_item: Dict) -> Optional[Dict]:
        """Parse table with all data"""
        try:
            res = table_item.get('res', {})
            html = res.get('html', '')

            if not html:
                return None

            from html.parser import HTMLParser

            class TableParser(HTMLParser):
                def __init__(self):
                    super().__init__()
                    self.rows = []
                    self.current_row = []
                    self.current_cell = ""
                    self.in_cell = False

                def handle_starttag(self, tag, attrs):
                    if tag in ['td', 'th']:
                        self.in_cell = True
                        self.current_cell = ""

                def handle_endtag(self, tag):
                    if tag in ['td', 'th']:
                        self.in_cell = False
                        self.current_row.append(self.current_cell.strip())
                    elif tag == 'tr':
                        if self.current_row:
                            self.rows.append(self.current_row)
                        self.current_row = []

                def handle_data(self, data):
                    if self.in_cell:
                        self.current_cell += data

            parser = TableParser()
            parser.feed(html)

            if parser.rows:
                return {
                    'headers': parser.rows[0] if parser.rows else [],
                    'rows': parser.rows[1:] if len(parser.rows) > 1 else [],
                    'all_data': parser.rows,
                    'num_rows': len(parser.rows),
                    'num_cols': len(parser.rows[0]) if parser.rows else 0
                }
            return None
        except:
            return None

class MultiPassVLMExtractor:
    """Extract with multiple focused passes for maximum accuracy using Qwen2-VL"""

    def __init__(self, model, processor):
        self.model = model
        self.processor = processor
        self.max_tokens = 4096

    def _run_vlm(self, image: Image.Image, prompt: str) -> str:
        """Run VLM with given prompt"""
        try:
            from qwen_vl_utils import process_vision_info
            
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": prompt}
                    ]
                }
            ]

            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )

            image_inputs, video_inputs = process_vision_info(messages)

            inputs = self.processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt"
            ).to(self.model.device)

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=self.max_tokens,
                    do_sample=False,
                )

            response = self.processor.batch_decode(
                outputs[:, inputs['input_ids'].shape[1]:],
                skip_special_tokens=True
            )[0]

            return response
        except Exception as e:
            logger.error(f"VLM Generation Error: {e}")
            return "{}"

    def _parse_json(self, text: str) -> Dict:
        """Robust JSON parsing"""
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0]
        elif "```" in text:
            parts = text.split("```")
            if len(parts) >= 2:
                text = parts[1]

        text = text.strip()

        start = text.find('{')
        end = text.rfind('}') + 1

        if start >= 0 and end > start:
            json_str = text[start:end]
            # Fix common JSON issues
            json_str = re.sub(r',\s*}', '}', json_str)
            json_str = re.sub(r',\s*]', ']', json_str)
            json_str = re.sub(r'\n', ' ', json_str)

            try:
                return json.loads(json_str)
            except json.JSONDecodeError:
                # Try to fix more issues
                json_str = re.sub(r"'([^']*)':", r'"\1":', json_str)
                try:
                    return json.loads(json_str)
                except:
                    pass

        return {'raw_response': text, 'parse_error': True}

    def extract_comprehensive(self, image: Image.Image, ocr_context: str = "") -> Dict:
        """Multi-pass extraction for maximum accuracy"""

        results = {
            'main_extraction': {},
            'chart_extraction': {},
            'table_extraction': {},
            'numbers_verification': {},
            'combined': {}
        }

        # Add OCR context to main prompt
        main_prompt = MAIN_EXTRACTION_PROMPT
        if ocr_context:
            main_prompt += f"\\n\\nOCR detected text for reference:\\n{ocr_context[:2000]}"

        # Pass 1: Main comprehensive extraction
        logger.info("      → Pass 1: Main extraction...")
        try:
            response = self._run_vlm(image, main_prompt)
            results['main_extraction'] = self._parse_json(response)
        except Exception as e:
            logger.error(f"      ⚠️ Main extraction error: {str(e)[:50]}")

        gc.collect()
        torch.cuda.empty_cache()

        # Pass 2: Focused chart extraction
        logger.info("      → Pass 2: Chart extraction...")
        try:
            response = self._run_vlm(image, CHART_EXTRACTION_PROMPT)
            results['chart_extraction'] = self._parse_json(response)
        except Exception as e:
            logger.error(f"      ⚠️ Chart extraction error: {str(e)[:50]}")

        gc.collect()
        torch.cuda.empty_cache()

        # Pass 3: Focused table extraction
        logger.info("      → Pass 3: Table extraction...")
        try:
            response = self._run_vlm(image, TABLE_EXTRACTION_PROMPT)
            results['table_extraction'] = self._parse_json(response)
        except Exception as e:
            logger.error(f"      ⚠️ Table extraction error: {str(e)[:50]}")

        gc.collect()
        torch.cuda.empty_cache()

        # Pass 4: Numbers verification
        logger.info("      → Pass 4: Numbers verification...")
        try:
            response = self._run_vlm(image, NUMBERS_VERIFICATION_PROMPT)
            results['numbers_verification'] = self._parse_json(response)
        except Exception as e:
            logger.error(f"      ⚠️ Numbers verification error: {str(e)[:50]}")

        # Combine all results
        results['combined'] = self._combine_results(results)

        return results

    def _combine_results(self, results: Dict) -> Dict:
        """Combine results from all passes"""
        combined = {}

        main = results.get('main_extraction', {})
        charts = results.get('chart_extraction', {})
        tables = results.get('table_extraction', {})
        numbers = results.get('numbers_verification', {})

        # Basic info from main
        combined['slide_title'] = main.get('slide_title', '')
        combined['slide_type'] = main.get('slide_type', 'unknown')
        combined['company_brand'] = main.get('company_brand', '')
        combined['key_highlights'] = main.get('key_highlights', [])
        combined['metrics'] = main.get('metrics', {})
        combined['sources'] = main.get('sources', [])

        # Charts - prefer focused extraction
        combined['charts'] = charts.get('charts', []) or main.get('bar_charts', []) + main.get('line_charts', [])

        # Tables - prefer focused extraction
        combined['tables'] = tables.get('tables', []) or main.get('tables', [])

        # Comparisons and growth
        combined['comparisons'] = main.get('comparisons', [])
        combined['growth_rates'] = main.get('growth_rates', [])
        combined['targets_projections'] = main.get('targets_projections', [])
        combined['investment_opportunities'] = main.get('investment_opportunities', [])
        combined['country_data'] = main.get('country_data', {})

        # Numbers - combine from verification and main
        combined['verified_numbers'] = {
            'percentages': numbers.get('percentages', []),
            'currency_amounts': numbers.get('currency_amounts', []),
            'quantities': numbers.get('quantities', []),
            'growth_rates': numbers.get('growth_rates', []),
            'years': numbers.get('years', []),
            'other_numbers': numbers.get('other_numbers', [])
        }

        # All numerical facts
        combined['all_numerical_facts'] = main.get('all_numerical_facts', [])

        return combined

class IntelligentDataFusion:
    """Merge OCR and VLM results with cross-validation"""

    def fuse(self, page_num: int, ocr_result: Dict, vlm_result: Dict) -> Dict:
        """Combine all extraction results"""

        vlm_combined = vlm_result.get('combined', {})

        fused = {
            'page_number': page_num,

            # Slide info (from VLM)
            'slide_title': vlm_combined.get('slide_title', ''),
            'slide_type': vlm_combined.get('slide_type', 'unknown'),
            'company_brand': vlm_combined.get('company_brand', ''),

            # Key content (from VLM)
            'key_highlights': vlm_combined.get('key_highlights', []),
            'metrics': vlm_combined.get('metrics', {}),

            # Charts (from VLM - semantic understanding)
            'charts': vlm_combined.get('charts', []),

            # Tables - combine OCR and VLM
            'tables_ocr': ocr_result.get('tables', []),
            'tables_vlm': vlm_combined.get('tables', []),

            # Comparisons and analysis (from VLM)
            'comparisons': vlm_combined.get('comparisons', []),
            'growth_rates': vlm_combined.get('growth_rates', []),
            'targets_projections': vlm_combined.get('targets_projections', []),
            'investment_opportunities': vlm_combined.get('investment_opportunities', []),
            'country_data': vlm_combined.get('country_data', {}),

            # Numbers - combine and cross-validate
            'numbers_ocr': {
                'all': ocr_result.get('all_numbers', []),
                'percentages': ocr_result.get('percentages', []),
                'currency': ocr_result.get('currency_values', []),
                'years': ocr_result.get('years', [])
            },
            'numbers_vlm': vlm_combined.get('verified_numbers', {}),

            # All text (from OCR - accurate)
            'all_text_ocr': ocr_result.get('all_text', []),
            'raw_text': ocr_result.get('raw_text', ''),

            # Numerical facts (from VLM)
            'numerical_facts': vlm_combined.get('all_numerical_facts', []),

            # Layout (from OCR structure)
            'layout_regions': ocr_result.get('layout_regions', []),

            # Sources
            'sources': vlm_combined.get('sources', []),

            # Confidence
            'ocr_confidence': ocr_result.get('confidence', 0),
        }

        # Generate comprehensive factual statements
        fused['factual_statements'] = self._generate_factual_statements(fused)

        # Cross-validate numbers
        fused['validated_numbers'] = self._cross_validate_numbers(fused)

        # Assess extraction quality
        fused['extraction_quality'] = self._assess_quality(fused)

        return fused

    def _generate_factual_statements(self, fused: Dict) -> List[str]:
        """Generate all factual statements from extracted data"""
        facts = []

        # From key highlights
        facts.extend(fused.get('key_highlights', []))

        # From metrics
        for name, data in fused.get('metrics', {}).items():
            if isinstance(data, dict):
                value = data.get('value', '')
                unit = data.get('unit', '')
                context = data.get('context', '')
                facts.append(f"{name}: {value} {unit} - {context}")
            else:
                facts.append(f"{name}: {data}")

        # From charts
        for chart in fused.get('charts', []):
            if isinstance(chart, dict):
                title = chart.get('title', '')
                if title:
                    facts.append(f"Chart: {title}")
                for label, value in chart.get('data', {}).items():
                    facts.append(f"  {label}: {value}")
                for label, value in chart.get('data_points', {}).items():
                    facts.append(f"  {label}: {value}")
                if chart.get('insight'):
                    facts.append(f"  Insight: {chart['insight']}")

        # From tables
        for table in fused.get('tables_vlm', []):
            if isinstance(table, dict):
                if table.get('title'):
                    facts.append(f"Table: {table['title']}")
                for row in table.get('data_rows', table.get('rows', []))[:10]:
                    if row:
                        facts.append(f"  {' | '.join(str(c) for c in row)}")

        # From comparisons
        for comp in fused.get('comparisons', []):
            if isinstance(comp, dict):
                e1 = comp.get('entity1', '')
                v1 = comp.get('entity1_value', '')
                e2 = comp.get('entity2', '')
                v2 = comp.get('entity2_value', '')
                metric = comp.get('metric', '')
                facts.append(f"Comparison ({metric}): {e1}={v1} vs {e2}={v2}")

        # From growth rates
        for growth in fused.get('growth_rates', []):
            if isinstance(growth, dict):
                facts.append(f"Growth: {growth.get('metric', '')} - {growth.get('rate', '')} ({growth.get('period', '')})")

        # From targets
        for target in fused.get('targets_projections', []):
            if isinstance(target, dict):
                facts.append(f"Target: {target.get('target', '')} = {target.get('value', '')} by {target.get('by_when', '')}")

        # From investments
        for inv in fused.get('investment_opportunities', []):
            if isinstance(inv, dict):
                facts.append(f"Investment: {inv.get('sector', '')} - {inv.get('amount', '')} by {inv.get('timeline', '')}")

        # From country data
        for country, data in fused.get('country_data', {}).items():
            if isinstance(data, dict):
                for metric, value in data.items():
                    facts.append(f"{country} - {metric}: {value}")

        # From VLM numerical facts
        facts.extend(fused.get('numerical_facts', []))

        # Clean and dedupe
        clean_facts = []
        seen = set()
        for f in facts:
            if f and f.strip() and f.strip() not in seen:
                clean_facts.append(f.strip())
                seen.add(f.strip())

        return clean_facts

    def _cross_validate_numbers(self, fused: Dict) -> Dict:
        """Cross-validate numbers from OCR and VLM"""

        ocr_numbers = set(fused.get('numbers_ocr', {}).get('all', []))
        vlm_numbers = set()

        vlm_num_data = fused.get('numbers_vlm', {})
        for key in ['percentages', 'currency_amounts', 'quantities', 'growth_rates', 'other_numbers']:
            vlm_numbers.update(vlm_num_data.get(key, []))

        # Find overlapping (validated) numbers
        validated = ocr_numbers.intersection(vlm_numbers)

        return {
            'ocr_only': list(ocr_numbers - vlm_numbers),
            'vlm_only': list(vlm_numbers - ocr_numbers),
            'validated_both': list(validated),
            'all_unique': list(ocr_numbers.union(vlm_numbers))
        }

    def _assess_quality(self, fused: Dict) -> str:
        """Assess extraction quality"""
        score = 0

        if fused.get('slide_title'):
            score += 1
        if len(fused.get('key_highlights', [])) > 0:
            score += 2
        if len(fused.get('metrics', {})) > 0:
            score += 2
        if len(fused.get('charts', [])) > 0:
            score += 2
        if len(fused.get('tables_vlm', [])) > 0 or len(fused.get('tables_ocr', [])) > 0:
            score += 2
        if len(fused.get('factual_statements', [])) > 5:
            score += 2
        if len(fused.get('validated_numbers', {}).get('all_unique', [])) > 10:
            score += 2
        if fused.get('ocr_confidence', 0) > 0.8:
            score += 1

        if score >= 10:
            return 'excellent'
        elif score >= 7:
            return 'good'
        elif score >= 4:
            return 'fair'
        else:
            return 'poor'

class PPTExtractorModule:
    """Wrapper for Maximum Accuracy Extraction Pipeline"""

    def __init__(self):        
        # -------------------------------------------------------------------------
        # LOAD MODELS
        # -------------------------------------------------------------------------
        logger.info("🔄 Loading High-Fidelity Models (GPU/CUDA)...")
        
        # PaddleOCR
        try:
            # Enable GPU for Paddle if available
            if torch.cuda.is_available():
                logger.info("   -> GPU Detected. Enabling GPU for PaddleOCR...")
                os.environ["FLAGS_use_gpu"] = "1"
                os.environ["FLAGS_use_mkldnn"] = "0"
                try:
                    paddle.set_device("gpu")
                except:
                    pass
            else:
                logger.info("   -> No GPU Detected. Using CPU for PaddleOCR...")
                os.environ["FLAGS_use_gpu"] = "0"
                os.environ["FLAGS_use_mkldnn"] = "1"
            
            from paddleocr import PaddleOCR
            try:
                # Standard legacy import
                from paddleocr import PPStructure
            except ImportError:
                # Newer versions (e.g. 3.x) often expose V3 directly or change structure
                logger.info("   -> PPStructure not found directly. Attempting to use PPStructureV3...")
                from paddleocr import PPStructureV3 as PPStructure
            import paddleocr
            import paddle
            
            logger.info("   -> Loading PaddleOCR...")
            use_gpu = torch.cuda.is_available()
            
            ppocr_engine = PaddleOCR(
                use_angle_cls=True,
                lang='en',
                ocr_version='PP-OCRv4',
                use_gpu=use_gpu,
                enable_mkldnn=not use_gpu,
                ir_optim=True,
                det_db_thresh=0.3,
                det_db_box_thresh=0.5,
            )
            pp_structure = PPStructure(
                table=True,
                ocr=True,
                show_log=True,
                layout=True,
                use_gpu=use_gpu,
                enable_mkldnn=not use_gpu,
                ir_optim=True
            )
        except ImportError:
            logger.error("❌ PaddleOCR not installed. Please run: pip install paddlepaddle-gpu paddleocr")
            raise
            
        # Transformers & Qwen
        try:
            from transformers import Qwen2VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig
            from qwen_vl_utils import process_vision_info
            
            logger.info("   -> Loading Qwen2-VL-7B (4-bit)...")
            VLM_MODEL = "Qwen/Qwen2-VL-7B-Instruct"
            
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
            
            vlm_model = Qwen2VLForConditionalGeneration.from_pretrained(
                VLM_MODEL,
                quantization_config=quantization_config,
                device_map="auto",
                trust_remote_code=True
            )
            
            vlm_processor = AutoProcessor.from_pretrained(VLM_MODEL, trust_remote_code=True)
                
        except ImportError:
            logger.error("❌ Transformers/Qwen dependencies not installed.")
            raise
        except Exception as e:
            logger.error(f"❌ Error loading VLM: {e}")
            raise
            
        logger.info("✅ Models Loaded.")
        
        self.ocr_extractor = ComprehensiveOCRExtractor(ppocr_engine, pp_structure)
        self.vlm_extractor = MultiPassVLMExtractor(vlm_model, vlm_processor)
        self.fusion_engine = IntelligentDataFusion()

    def pdf_to_images(self, pdf_path: str, zoom: float = 2.5) -> List[Tuple]:
        """Convert PDF to high-res images"""
        doc = fitz.open(pdf_path)
        images = []
        
        logger.info(f"Converting PDF {pdf_path} to images...")
        for i in range(doc.page_count):
            page = doc[i]
            mat = fitz.Matrix(zoom, zoom)
            pix = page.get_pixmap(matrix=mat, alpha=False)
            pil_img = Image.open(io.BytesIO(pix.tobytes("png")))
            np_img = np.array(pil_img)
            if len(np_img.shape) == 2:
                np_img = cv2.cvtColor(np_img, cv2.COLOR_GRAY2RGB)
            images.append((i + 1, pil_img, np_img))

        doc.close()
        return images

    def process_slide(self, page_num: int, pil_img: Image.Image, np_img: np.ndarray) -> Dict:
        """Process single slide with maximum accuracy"""
        import time
        start = time.time()

        logger.info(f"📄 PROCESSING SLIDE {page_num}")

        # Step 1: Comprehensive OCR
        ocr_result = self.ocr_extractor.extract(np_img)

        # Step 2: Multi-pass VLM extraction
        vlm_result = self.vlm_extractor.extract_comprehensive(pil_img, ocr_result['raw_text'])

        # Step 3: Intelligent fusion
        fused = self.fusion_engine.fuse(page_num, ocr_result, vlm_result)

        elapsed = time.time() - start
        logger.info(f"⏱️  Completed Slide {page_num} in {elapsed:.1f}s")
        
        # Clean up per slide
        gc.collect()
        torch.cuda.empty_cache()

        return fused
        
    def process_document(self, pdf_path: str) -> List[Dict]:
        """Process entire document and return list of fused results"""
        slides = self.pdf_to_images(pdf_path)
        logger.info(f"✅ Converted {len(slides)} slides to high-res images")
        
        results = []
        for page_num, pil_img, np_img in slides:
            result = self.process_slide(page_num, pil_img, np_img)
            results.append(result)
            
        return results

# ==============================================================================
# PIPELINE 3 INTEGRATION (PARALLEL LOGIC + SEQUENTIAL EXECUTION)
# ==============================================================================

class Pipeline3ParallelPPT:
    """
    Pipeline 3: Parallel Extraction with Reconciliation + High-Accuracy PPT Integration
    
    Architecture:
    1. Independent Extraction Streams (Logical Parallelism):
       - Stream A: Transcript (LLM)
       - Stream B: Notes (LLM)
       - Stream C: PPT Visuals (V6 Extractor - OCR+VLM+Fusion)
    
    2. Reconciliation (Judge):
       - Receives Textual Extractions + Hybrid Visual Evidence (Facts + JSON)
    """
    
    def __init__(self, model_name: str = None):
        self.config = Config()
        self.model_name = model_name or self.config.DEFAULT_MODEL
        self.llm = LLMInterface(self.model_name)
        self.doc_processor = DocumentProcessor()
        
        # Initialize PPT Extractor
        logger.info("Initializing PPT Extractor Module (Heavy Load)...")
        try:
           self.ppt_extractor = PPTExtractorModule()
        except Exception as e:
            logger.error(f"Failed to initialize PPT Extractor: {e}")
            self.ppt_extractor = None
            
        self.current_folder = None
    
    def process_documents(self, target_folder: str = None) -> Dict[str, Any]:
        """
        Process documents: Transcript + Notes + PPT (Parallel Extraction)
        """
        logger.info("Pipeline 3 (PPT+): Starting document processing")
        
        # Handle Batch Folder Logic
        if target_folder:
            self.current_folder = target_folder
        else:
             if not hasattr(self, 'current_folder') or not self.current_folder:
                data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../data")
                if os.path.exists(data_dir):
                     subs = [os.path.join(data_dir, d) for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
                     if subs:
                         self.current_folder = subs[0]
            
        if not self.current_folder:
             logger.error("No target folder specified for processing.")
             return {}
             
        logger.info(f"Processing Folder: {self.current_folder}")
        
        # Helper to resolve paths from config
        def resolve_path(file_type):
            for fname in self.config.INPUT_FILES.get(file_type, []):
                path = os.path.join(self.current_folder, fname)
                if os.path.exists(path):
                    return path
            return None

        # Load Inputs
        transcript_path = resolve_path("transcript")
        notes_path = resolve_path("notes")
        qna_path = resolve_path("qna")
        
        # Validation
        if not transcript_path:
            logger.error(f"Missing transcript in {self.current_folder}")
            return {}
            
        # Load Content
        transcript = ""
        notes = ""
        qna = ""
        
        try:
             # Transcript (Required)
             if transcript_path.lower().endswith('.docx'):
                 transcript = self.doc_processor.preprocess_text(self.doc_processor.read_docx(transcript_path))
             else:
                 with open(transcript_path, 'rb') as f:
                     transcript = self.doc_processor.preprocess_text(self.doc_processor.read_pdf(f))
                 
             # Notes (Optional but recommended)
             if notes_path:
                 if notes_path.lower().endswith('.docx'):
                     notes = self.doc_processor.preprocess_text(self.doc_processor.read_docx(notes_path))
                 else:
                     with open(notes_path, 'rb') as f:
                         notes = self.doc_processor.preprocess_text(self.doc_processor.read_pdf(f))
             else:
                 logger.warning(f"No notes found in {self.current_folder}, proceeding without notes.")
                 
             # QnA (New Integration)
             if qna_path:
                 if qna_path.lower().endswith('.docx'):
                     qna = self.doc_processor.preprocess_text(self.doc_processor.read_docx(qna_path))
                 else:
                     with open(qna_path, 'rb') as f:
                         qna = self.doc_processor.preprocess_text(self.doc_processor.read_pdf(f))
                 logger.info(f"Loaded QnA: {len(qna)} chars")
             else:
                 logger.warning("No QnA file found. Skipping Stream D.")

        except Exception as e:
            logger.error(f"Error loading files: {e}")
            return {}
        
        logger.info(f"Loaded transcript: {len(transcript)} chars, Notes: {len(notes)} chars")
        
        # ==============================================================================
        # STEP 1: PARALLEL EXTRACTION STREAM (Multi-GPU Parallel Execution)
        # ==============================================================================
        logger.info("Pipeline 3 (PPT+): Starting Multi-Stream Extraction (Parallel Execution on 2 GPUs)")
        
        from concurrent.futures import ThreadPoolExecutor
        
        def run_visual_stream():
            logger.info(" >> Stream C: Extracting from Visuals (High Fidelity) on GPU 0...")
            # Path adjustment: Look in the current batch folder
            pdf_path = os.path.join(self.current_folder, "ppt.pdf")
            if not os.path.exists(pdf_path):
                 # Try capitalized
                 pdf_path = os.path.join(self.current_folder, "PPT.pdf")

            visual_text = ""
            structured_data = []
            
            if os.path.exists(pdf_path) and self.ppt_extractor:
                 try:
                    # Model pinned to cuda:0 in PPTExtractorModule
                    slide_results = self.ppt_extractor.process_document(pdf_path)
                    structured_data = slide_results
                    
                    context_parts = []
                    for res in slide_results:
                        page_num = res['page_number']
                        structured_json = json.dumps(res, indent=2, ensure_ascii=False)
                        factual_stats = "\n".join(res.get('factual_statements', []))
                        
                        slide_block = f"""
=== SLIDE {page_num} ===
[FACTUAL SUMMARY]
{factual_stats}

[RAW DATA (JSON)]
{structured_json}
"""
                        context_parts.append(slide_block)
                    
                    visual_text = "\n\n=== VISUAL EVIDENCE (HYBRID FORMAT) ===\n" + "\n".join(context_parts)
                    
                 except Exception as e:
                    logger.error(f"Error processing PPT Stream: {e}")
                    visual_text = "[ERROR IN VISUAL STREAM]"
            else:
                logger.warning("ppt.pdf not found or PPT model failed. Stream C empty.")
                visual_text = "[NO VISUAL EVIDENCE AVAILABLE]"
            
            return visual_text, structured_data

        def run_transcript_stream():
            logger.info(" >> Stream A: Extracting from Transcript (Text LLM)...")
            return self.llm.extract_information(transcript, "financial")

        def run_notes_stream():
            if not notes: return "No notes available."
            logger.info(" >> Stream B: Extracting from Notes (Text LLM)...")
            return self.llm.extract_information(notes, "financial")
            
        def run_qna_stream():
            if not qna: return "No QnA available."
            logger.info(" >> Stream D: Extracting from QnA (Text LLM)...")
            return self.llm.extract_information(qna, "financial, strategic, risk")

        # Execute in parallel
        # We have 4 potential streams now: Visual, Transcript, Notes, QnA
        with ThreadPoolExecutor(max_workers=4) as executor:
            future_visual = executor.submit(run_visual_stream)
            future_transcript = executor.submit(run_transcript_stream)
            future_notes = executor.submit(run_notes_stream)
            future_qna = executor.submit(run_qna_stream)
            
            visual_extraction_text, ppt_structured_data = future_visual.result()
            transcript_extraction = future_transcript.result()
            notes_extraction = future_notes.result()
            qna_extraction = future_qna.result()
            
        logger.info("Pipeline 3 (PPT+): Parallel Extraction Completed.")

        # ==============================================================================
        # STEP 2: RECONCILIATION
        # ==============================================================================
        logger.info("Pipeline 3 (PPT+): Reconciling parallel streams")
        reconciled_info = self.reconcile_extractions(
            transcript_extraction, 
            notes_extraction, 
            visual_extraction_text,
            qna_extraction
        )
        
        # Add metadata back
        reconciled_info['ppt_structured_data'] = ppt_structured_data
        
        logger.info("Pipeline 3 (PPT+): Document processing completed")
        return reconciled_info
    
    def reconcile_extractions(self, transcript_extraction: str, notes_extraction: str, visual_extraction: str = "", qna_extraction: str = "") -> Dict[str, Any]:
        """
        Reconcile information extracted from transcript, notes, visuals, and QnA.
        """
        # Updated Prompt for Hybrid JSON/Text Visuals + QnA
        reconciliation_prompt = f"""You are a strict Judge evaluating financial claims from multiple sources.
You have FOUR sources of information:
1. Transcript (Spoken by executives)
2. Notes (Written summary)
3. Visuals (Slides - Hybrid format: Natural language summaries + RAW JSON DATA)
4. QnA (Unscripted Questions & Answers - Critical for risks and strategic clarity)

Your Task:
1. Create a unified summary of the financial performance.
2. **CROSS-MODAL VERIFICATION (CRITICAL)**: Compare the Textual claims (Transcript/Notes) against the Visual Evidence.
   - **USE THE JSON DATA**: The 'Visuals' section contains raw JSON. Use this for precise number verification.
   - If the Text says "Revenue up" but Visual chart JSON shows a decline, FLAG THIS as a "Possible Hallucination" or "Contradiction".
   - Explicitly list any discrepancies between what was said and what was shown.
   - If Visuals confirm the Text, note that as "Visually Verified".
3. **QnA Insights**: Highlight any new risks, clarifications, or strategic shifts revealed ONLY in the QnA section that were missing from the main presentation.

Transcript Extraction:
{transcript_extraction}

Notes Extraction:
{notes_extraction}

QnA Extraction:
{qna_extraction}

Visual Evidence (Slides + JSON):
{visual_extraction}

Reconciled Summary & Verification Report:"""

        reconciled_response = self.llm.generate_single_response(
            reconciliation_prompt,
            options={
                'temperature': 0.1, # Lower temperature for judging
                'top_p': 0.9,
                'num_predict': 2048
            }
        )
        
        return {
            'reconciled_info': reconciled_response['response'],
            'visual_extraction': visual_extraction,
            'transcript_extraction': transcript_extraction,
            'notes_extraction': notes_extraction,
            'qna_extraction': qna_extraction
        }
    
    def parse_questions_from_text(self, text: str) -> List[str]:
        """Extract questions from QnA text (heuristic based)"""
        questions = []
        if not text:
            return []
        
        # Split by newlines and clean
        lines = text.split('\n')
        for line in lines:
            line = line.strip()
            # Heuristic: Starts with Q/Question/q/question or Number (1.), ends with ?
            if line.strip().endswith('?') and len(line) > 10:
                 # Clean up prefix if it looks like "1. " or "Q: "
                 q_clean = re.sub(r'^(Q\d*|Question|[\d\.]+)\s?[:\.]?', '', line, flags=re.IGNORECASE).strip()
                 if len(q_clean) > 5:
                     questions.append(q_clean)
        
        return questions

    def run_pipeline(self, question: str = None, target_folder: str = None) -> List[Dict[str, Any]]:
        """
        Run complete Pipeline 3.
        """
        start_time = datetime.now()
        
        # NOTE: We do NOT default to self.config.DEFAULT_QUESTION immediately here
        # so we can detect if we should run the batch list.
        current_cli_question = question 
        
        logger.info(f"Pipeline 3 (PPT+): Starting complete pipeline run")
        
        # Step 1: Parallel Document Processing (Run Once)
        docs_data = self.process_documents(target_folder=target_folder)
        if not docs_data:
             logger.error("Document processing failed.")
             return {}
        
        processed_info = docs_data['reconciled_info']
        visual_extraction = docs_data['visual_extraction']
        qna_text = docs_data.get('qna_extraction', "")
        transcript_notes_text = f"Transcript Ext: {docs_data['transcript_extraction']}\nNotes Ext: {docs_data['notes_extraction']}"
        
        # Determine Questions list
        questions_to_ask = []
        
        # 1. Explicit CLI Question (Priority 1)
        if current_cli_question:
            questions_to_ask.append(current_cli_question)
        
        # 2. Parse QnA Doc (Priority 2)
        if not questions_to_ask:
            parsed_questions = self.parse_questions_from_text(qna_text)
            if parsed_questions:
                logger.info(f"Found {len(parsed_questions)} questions in QnA document.")
                questions_to_ask.extend(parsed_questions)
        
        # 3. Manual Config List (Priority 3)
        if not questions_to_ask and hasattr(self.config, 'QUESTIONS_LIST') and self.config.QUESTIONS_LIST:
             logger.info(f"Using {len(self.config.QUESTIONS_LIST)} manual questions from Config.")
             questions_to_ask.extend(self.config.QUESTIONS_LIST)
             
        # 4. Default Single Question (Priority 4 - Fallback)
        if not questions_to_ask:
             logger.info("No questions found in QnA or CLI. Using default question.")
             questions_to_ask.append(self.config.DEFAULT_QUESTION)
             
        # Dedup
        questions_to_ask = list(OrderedDict.fromkeys(questions_to_ask))
        logger.info(f"Pipeline 3 (PPT+): Will process {len(questions_to_ask)} questions.")
        
        all_results = []
        
        for idx, q in enumerate(questions_to_ask):
            logger.info(f"--- Processing Question {idx+1}/{len(questions_to_ask)}: {q[:50]}... ---")
            
            # Step 2: Generate answers
            qa_results = self.llm.generate_qa_responses(processed_info, q)
            
            # Step 3: Compute Hallucination Score
            hallucination_results = self.llm.calculate_hallucination_score(
                answer=qa_results['best_response']['response'],
                visual_facts=visual_extraction, 
                textual_facts=transcript_notes_text
            )
            
            res_entry = {
                'pipeline_name': 'Pipeline 3 (High-Res PPT Integration)',
                'pipeline_type': 'parallel_extraction_ppt_high_res',
                'model_used': self.model_name,
                'question': q,
                'processed_information': processed_info,
                'best_answer': {
                    'answer': qa_results['best_response']['response'],
                    'temperature': qa_results['best_response']['parameters'].get('temperature', 0.1),
                    'generation_time': qa_results['best_response']['generation_time']
                },
                'hallucination_score': hallucination_results,
                'ppt_metadata': {
                    'slides_processed': len(docs_data.get('ppt_structured_data', []))
                }
            }
            all_results.append(res_entry)
        
        end_time = datetime.now()
        execution_time = (end_time - start_time).total_seconds()
        
        # Add timestamp to all (or just return list)
        for r in all_results:
            r['timestamp'] = end_time.isoformat()
            r['execution_time'] = execution_time # Total time for batch
        
        logger.info(f"Pipeline 3 (PPT+) Completed processing {len(all_results)} questions in {execution_time:.2f} seconds")
        return all_results
    
    def save_results(self, results: Dict[str, Any], output_dir: str = None) -> str:
        if output_dir is None:
            # Save into results/BatchFolder/
            batch_name = os.path.basename(self.current_folder) if self.current_folder else "default_batch"
            output_dir = os.path.join(self.config.RESULTS_DIR, batch_name)
        
        os.makedirs(output_dir, exist_ok=True)
        
        results_file = os.path.join(output_dir, "pipeline3_ppt_results.json")
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
            
        logger.info(f"Results saved to {results_file}")
        return results_file
    
    def cleanup(self):
        if hasattr(self, 'ppt_extractor'):
             del self.ppt_extractor
             gc.collect()
             torch.cuda.empty_cache()

def main():
    """Main function for running Pipeline 3 standalone."""
    logging.basicConfig(level=logging.INFO)
    
    # Validation Check
    if not torch.cuda.is_available():
        logger.warning("⚠️  WARNING: No CUDA device detected. This script is optimized for GPU servers.")
        
    # Run pipeline
    pipeline = Pipeline3ParallelPPT()
    results = pipeline.run_pipeline()
    pipeline.save_results(results)
    pipeline.cleanup()
    
    print(f"Pipeline 3 (PPT+) completed successfully!")

if __name__ == "__main__":
    main()
