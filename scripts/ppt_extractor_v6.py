#!/usr/bin/env python3
"""
PPT_Extractor_V6_MaxAccuracy.py

Converted from PPT_Extractor_V6_MaxAccuracy.ipynb
Designed for Complex Business Presentations with Dense Data.
"""

import sys
import os
import io
import json
import re
import cv2
import numpy as np
import fitz  # PyMuPDF
from PIL import Image
import torch
import gc
import argparse
import logging
from typing import List, Dict, Optional, Tuple, Any
from collections import OrderedDict
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Suppress warnings
import warnings
warnings.filterwarnings('ignore')

# Check GPU
if torch.cuda.is_available():
    gpu_name = torch.cuda.get_device_name(0)
    gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
    logger.info(f"✅ GPU: {gpu_name} ({gpu_mem:.1f} GB)")
else:
    logger.warning("❌ NO GPU DETECTED. Inference will be slow or might fail.")

# ==============================================================================
# PROMPTS
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
# COMPREHENSIVE OCR EXTRACTOR
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

        # Percentages: 6.7%, 13%, etc.
        percentages = re.findall(r'[\d,]+\.?\d*\s*%', text)
        result['percentages'].extend(percentages)

        # Currency: $500, $1.2Bn, $35 Tn, etc.
        currencies = re.findall(r'\$[\d,]+\.?\d*\s*[BMKTn]*\+?', text, re.IGNORECASE)
        result['currency_values'].extend(currencies)

        # Years: FY25, FY32E, 2024, 2047, etc.
        years = re.findall(r"FY\d{2}E?|20\d{2}|'\d{2}", text)
        result['years'].extend(years)

        # General numbers with units: 1.46, 186 bn, 112 GW+, etc.
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

# ==============================================================================
# MULTI-PASS VLM EXTRACTOR
# ==============================================================================

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

# ==============================================================================
# INTELLIGENT DATA FUSION
# ==============================================================================

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

# ==============================================================================
# MAIN PIPELINE
# ==============================================================================

class MaxAccuracyPipeline:
    """Maximum accuracy extraction pipeline"""

    def __init__(self, ocr_extractor, vlm_extractor, fusion_engine):
        self.ocr = ocr_extractor
        self.vlm = vlm_extractor
        self.fusion = fusion_engine

    def pdf_to_images(self, pdf_path: str, zoom: float = 2.5,
                      start_page: int = 0, end_page: int = None) -> List[Tuple]:
        """Convert PDF to high-res images"""
        doc = fitz.open(pdf_path)
        end = end_page if end_page else doc.page_count
        images = []
        
        # Validate ranges
        start_page = max(0, start_page)
        end = min(end, doc.page_count)

        logger.info(f"Converting pages {start_page} to {end}...")
        for i in range(start_page, end):
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

        print(f"\n{'═'*60}")
        print(f"📄 PROCESSING SLIDE {page_num}")
        print(f"{'═'*60}")

        # Step 1: Comprehensive OCR
        print("\n🔤 Step 1: OCR Extraction")
        ocr_result = self.ocr.extract(np_img)
        print(f"   ✓ Text blocks: {len(ocr_result['all_text'])}")
        print(f"   ✓ Tables detected: {len(ocr_result['tables'])}")
        print(f"   ✓ Numbers found: {len(ocr_result['all_numbers'])}")
        print(f"   ✓ Percentages: {len(ocr_result['percentages'])}")
        print(f"   ✓ Currency values: {len(ocr_result['currency_values'])}")
        print(f"   ✓ Confidence: {ocr_result['confidence']:.2%}")

        # Step 2: Multi-pass VLM extraction
        print("\n🧠 Step 2: VLM Multi-Pass Extraction")
        vlm_result = self.vlm.extract_comprehensive(pil_img, ocr_result['raw_text'])

        vlm_combined = vlm_result.get('combined', {})
        print(f"   ✓ Title: {vlm_combined.get('slide_title', 'N/A')[:60]}")
        print(f"   ✓ Type: {vlm_combined.get('slide_type', 'N/A')}")
        print(f"   ✓ Key highlights: {len(vlm_combined.get('key_highlights', []))}")
        print(f"   ✓ Metrics: {len(vlm_combined.get('metrics', {}))}")
        print(f"   ✓ Charts: {len(vlm_combined.get('charts', []))}")
        print(f"   ✓ Tables: {len(vlm_combined.get('tables', []))}")

        # Step 3: Intelligent fusion
        print("\n🔀 Step 3: Data Fusion")
        fused = self.fusion.fuse(page_num, ocr_result, vlm_result)
        print(f"   ✓ Factual statements: {len(fused['factual_statements'])}")
        print(f"   ✓ Validated numbers: {len(fused['validated_numbers'].get('all_unique', []))}")
        print(f"   ✓ Quality: {fused['extraction_quality'].upper()}")

        elapsed = time.time() - start
        print(f"\n⏱️  Completed in {elapsed:.1f}s")
        
        # Clean up per slide
        gc.collect()
        torch.cuda.empty_cache()

        return fused

    def process_document(self, pdf_path: str, zoom: float = 2.5,
                         start_page: int = 0, end_page: int = None, output_dir: str = None) -> Tuple[Dict, Dict]:
        """Process entire document"""
        import time

        print("\n" + "═"*70)
        print("🎯 MAXIMUM ACCURACY EXTRACTION PIPELINE")
        print("═"*70)
        print(f"📁 Document: {os.path.basename(pdf_path)}")
        print(f"🔧 Zoom: {zoom}x | Multi-pass VLM | Cross-validation enabled")
        print("═"*70)

        start_time = time.time()

        slides = self.pdf_to_images(pdf_path, zoom, start_page, end_page)
        print(f"\n✅ Converted {len(slides)} slides to high-res images")

        results = []
        stored_images = {}

        for page_num, pil_img, np_img in slides:
            stored_images[page_num] = pil_img.copy()
            result = self.process_slide(page_num, pil_img, np_img)
            results.append(result)

        total_time = time.time() - start_time

        output = {
            'document_info': {
                'source_file': os.path.basename(pdf_path),
                'total_pages': len(slides),
                'processing_time_seconds': total_time,
                'timestamp': datetime.now().isoformat(),
                'pipeline_version': 'V6-MaxAccuracy'
            },
            'pages': results
        }

        self._print_summary(output)
        
        # Save results
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            output_file = os.path.join(output_dir, f"{os.path.basename(pdf_path).split('.')[0]}_results.json")
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(output, f, indent=2, ensure_ascii=False)
            print(f"\n✅ Results saved to {output_file}")

        return output, stored_images

    def _print_summary(self, output: Dict):
        """Print comprehensive summary"""
        print("\n" + "═"*70)
        print("📊 EXTRACTION SUMMARY")
        print("═"*70)

        total_facts = 0
        total_numbers = 0
        total_charts = 0
        total_tables = 0

        for page in output['pages']:
            total_facts += len(page.get('factual_statements', []))
            total_numbers += len(page.get('validated_numbers', {}).get('all_unique', []))
            total_charts += len(page.get('charts', []))
            total_tables += len(page.get('tables_vlm', [])) + len(page.get('tables_ocr', []))

        print(f"📁 Document: {output['document_info']['source_file']}")
        print(f"📄 Pages processed: {output['document_info']['total_pages']}")
        print(f"⏱️  Total time: {output['document_info']['processing_time_seconds']:.1f}s")
        print(f"\n📈 Extracted Data:")
        print(f"   • Factual statements: {total_facts}")
        print(f"   • Unique numbers: {total_numbers}")
        print(f"   • Charts analyzed: {total_charts}")
        print(f"   • Tables detected: {total_tables}")

        qualities = [p.get('extraction_quality', 'unknown') for p in output['pages']]
        print(f"\n📊 Quality:")
        for q in ['excellent', 'good', 'fair', 'poor']:
            count = qualities.count(q)
            if count > 0:
                print(f"   • {q.capitalize()}: {count} slides")

        print("═"*70)

def main():
    parser = argparse.ArgumentParser(description="PPT Extractor V6 - Max Accuracy")
    parser.add_argument("--input", type=str, required=True, help="Path to input PDF file")
    parser.add_argument("--output", type=str, default="./results", help="Directory to save results")
    parser.add_argument("--zoom", type=float, default=2.5, help="Zoom factor for PDF to Image conversion")
    parser.add_argument("--start_page", type=int, default=0, help="Start page index (0-based)")
    parser.add_argument("--end_page", type=int, help="End page index (exclusive)")
    
    args = parser.parse_args()
    
    # -------------------------------------------------------------------------
    # LOAD MODELS
    # -------------------------------------------------------------------------
    print("🔄 Loading Models (this may take a few minutes)...")
    
    # PaddleOCR
    try:
        from paddleocr import PaddleOCR, PPStructure
        
        print("   -> Loading PaddleOCR...")
        ppocr_engine = PaddleOCR(
            use_angle_cls=True,
            lang='en',
            use_gpu=torch.cuda.is_available(),
            show_log=False,
            ocr_version='PP-OCRv4',
            det_db_thresh=0.3,
            det_db_box_thresh=0.5,
        )
        pp_structure = PPStructure(
            table=True,
            ocr=True,
            show_log=False,
            use_gpu=torch.cuda.is_available(),
            layout=True,
            lang='en' # Added to prevent errors if default is different
        )
    except ImportError:
        print("❌ PaddleOCR not installed. Please run: pip install paddlepaddle-gpu paddleocr")
        sys.exit(1)
        
    # Transformers & Qwen
    try:
        from transformers import Qwen2VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig
        from qwen_vl_utils import process_vision_info
        
        print("   -> Loading Qwen2-VL-7B...")
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
        print("❌ Transformers/Qwen dependencies not installed.")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error loading VLM: {e}")
        sys.exit(1)
        
    print("✅ Models Loaded.")
    
    # -------------------------------------------------------------------------
    # INIT PIPELINE
    # -------------------------------------------------------------------------
    ocr_extractor = ComprehensiveOCRExtractor(ppocr_engine, pp_structure)
    vlm_extractor = MultiPassVLMExtractor(vlm_model, vlm_processor)
    fusion_engine = IntelligentDataFusion()
    
    pipeline = MaxAccuracyPipeline(ocr_extractor, vlm_extractor, fusion_engine)
    
    # -------------------------------------------------------------------------
    # RUN
    # -------------------------------------------------------------------------
    if not os.path.exists(args.input):
        print(f"❌ Input file not found: {args.input}")
        sys.exit(1)
        
    pipeline.process_document(
        pdf_path=args.input,
        zoom=args.zoom,
        start_page=args.start_page,
        end_page=args.end_page,
        output_dir=args.output
    )

if __name__ == "__main__":
    main()
