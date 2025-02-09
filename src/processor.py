import argparse
import yaml
import logging
import re
import uuid
import html
from pathlib import Path
from datetime import datetime
from dateutil.parser import parse
from lxml import etree
from tqdm import tqdm
import json
from datasets import Dataset, Features, Value, Sequence
from FeatureHTMLExtractor import HTMLExtractor

logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO,
    filename='processor.log'
)
logger = logging.getLogger(__name__)

class MetadataExtractor:
    def __init__(self, config_path, label_map_path):
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)

        with open(label_map_path, 'r', encoding='utf-8') as f:
            self.label_map = json.load(f)
        
        self.label2id = self.label_map["label2id"]
        self.remove_tags = self.config.get('remove_tags', ['script', 'style'])
        self.max_html_length = self.config.get('max_html_length', 10000)

    def _clean_html(self, html_content):
        """Очистка HTML с сохранением структуры"""
        try:
            parser = etree.HTMLParser(remove_comments=True, encoding='utf-8')
            tree = etree.fromstring(html_content, parser)
            
            for tag in self.remove_tags:
                for element in tree.xpath(f'.//{tag}'):
                    parent = element.getparent()
                    if parent is not None:
                        parent.remove(element)
            
            return etree.tostring(tree, method='html', encoding='unicode')
        except Exception as e:
            logger.error(f"HTML cleaning error: {str(e)}")
            return html_content

    def _process_value(self, raw_value, selector_config):
        """Обработка значения с улучшенной обработкой ошибок"""
        try:
            value = str(raw_value).strip() if raw_value else ""
            
            if 'regex' in selector_config:
                match = re.search(selector_config['regex'], value)
                if match:
                    value = match.group(1 if match.lastindex else 0)
            
            processing_ops = {
                'strip': lambda x: x.strip(),
                'lower': lambda x: x.lower(),
                'upper': lambda x: x.upper(),
                'title': lambda x: x.title(),
                'unescape': lambda x: html.unescape(x),
                'normalize_date': self._normalize_date
            }
            
            for op in selector_config.get('post_process', []):
                if op in processing_ops:
                    try:
                        value = processing_ops[op](value)
                    except Exception as e:
                        logger.warning(f"Processing failed for '{op}': {str(e)}")
            
            return value
        except Exception as e:
            logger.error(f"Value processing error: {str(e)}")
            return None

    def _normalize_date(self, value):
        """Нормализация даты с обработкой исключений"""
        try:
            return parse(value, fuzzy=True).strftime('%Y-%m-%d')
        except Exception as e:
            logger.warning(f"Date normalization failed: {str(e)}")
            return value

    def _validate_value(self, value, field_config):
        """Улучшенная валидация с кастомными правилами"""
        if not value:
            return False
            
        validators = {
            'required': lambda x: len(x.strip()) > 0,
            'date': lambda x: bool(re.match(r'^\d{4}-\d{2}-\d{2}$', x)),
            'issn': lambda x: bool(re.match(r'^\d{4}-\d{3}[\dXx]$', x)),
            'doi': lambda x: bool(re.match(r'^10\.\d{4,}/', x, re.I)),
            'url': lambda x: bool(re.match(r'^https?://', x)),
        }
        
        validators.update(self.config.get('validators', {}))
        
        for validator in field_config.get('validate', []):
            if not validators.get(validator, lambda x: True)(value):
                return False
        return True

    def extract_metadata(self, html_content):
        """Основной метод извлечения метаданных с новой логикой разметки"""
        try:
            # Шаг 1: Извлечение всех токенов и xpaths
            html_extractor = HTMLExtractor(
                html_content,
                extract_text=True,
                extract_attributes=True,
                target_attrs={'content', 'href', 'src', 'value', "name"},
                exclude_tags=self.remove_tags
            )
            all_tokens, all_xpaths = html_extractor.extract()
            
            # Шаг 2: Извлечение структурированных метаданных с xpaths
            metadata = {}
            field_xpath_map = {}

            parser = etree.HTMLParser(encoding='utf-8')
            tree = etree.fromstring(html_content, parser)

            for field_name, field_config in self.config['fields'].items():
                field_data = {"text": [], "xpaths": []}
                collected_xpaths = set()

                for selector in field_config['selectors']:
                    try:
                        elements = []
                        if selector['type'] == 'css':
                            elements = tree.cssselect(selector['selector'])
                        elif selector['type'] == 'xpath':
                            elements = tree.xpath(selector['selector'])

                        # Поддержка стратегии: по умолчанию 'first'
                        strategy = selector.get('strategy', 'first').lower()
                        if strategy == 'first':
                            elements = elements[:1]
                        # Если указан 'multi', оставляем все найденные элементы
                        
                        for element in elements:
                            # Извлечение сырого значения
                            raw_value = ""
                            attr_name = None
                            if selector.get('extract') == 'attr':
                                attr_name = selector.get('attr_name', '')
                                raw_value = element.get(attr_name, '')
                            else:
                                raw_value = element.text.strip() if hasattr(element, 'text') else ""

                            value = self._process_value(raw_value, selector)
                            if value and self._validate_value(value, field_config):
                                #Поиск соответствующего xpath через HTMLExtractor
                                if attr_name:
                                    xpath = html_extractor.get_xpath(element=element, attr=attr_name)
                                else:
                                    xpath = html_extractor.get_xpath(element=element)
                                
                                if xpath and xpath not in collected_xpaths:
                                    print("value", value)
                                    field_data["text"].append(value)
                                    field_data["xpaths"].append(xpath)
                                    collected_xpaths.add(xpath)

                    except Exception as e:
                        logger.error(f"Error processing selector {selector}: {str(e)}")

                metadata[field_name] = field_data
                field_xpath_map[field_name] = field_data["xpaths"]

            # Шаг 3: Генерация node_labels
            node_labels = [self.label2id.get("O", 0)] * len(all_tokens)
            for field, xpaths in field_xpath_map.items():
                label_id = self.label2id.get(field.upper(), 0)
                print("FIELD UPPER: ",field.upper(), " LABEL ID: ", label_id)
                for xpath in xpaths:
                    for idx, token_xpath in enumerate(all_xpaths):
                        if token_xpath == xpath:
                            node_labels[idx] = label_id

            return {
                "tokens": all_tokens,
                "xpaths": all_xpaths,
                "metadata": metadata,
                "node_labels": node_labels,
                "html": html_content,
                "processing_time": datetime.now().isoformat()
            }

        except Exception as e:
            logger.error(f"Metadata extraction failed: {str(e)}")
            return None


    def _find_element_xpath(self, element_html, tokens, xpaths):
        """Находит xpath элемента через сравнение HTML-контента"""
        try:
            # Поиск точного совпадения HTML
            for idx, token in enumerate(tokens):
                if token in element_html and element_html in xpaths[idx]:
                    return xpaths[idx]
            
            # Фолбэк: поиск по частичному совпадению
            for idx, token in enumerate(tokens):
                if token in element_html:
                    return xpaths[idx]
            
            return None
        except Exception as e:
            logger.warning(f"XPath finding error: {str(e)}")
            return None


def main(args):
    extractor = MetadataExtractor(args.config, args.label_map)
    
    features = Features({
        "id": Value("string"),
        "source_file": Value("string"),
        "resource": Value("string"),
        "doc_type": Value("string"),
        "html": Value("string"),
        "tokens": Sequence(Value("string")),
        "xpaths": Sequence(Value("string")),
        "metadata": Features({
            field: Features({
                "text": Sequence(Value("string")),
                "xpaths": Sequence(Value("string"))
            }) for field in extractor.config['fields']
        }),
        "node_labels": Sequence(Value("string")),
        "processing_time": Value("string")
    })
    
    def generate_examples():
        html_files = list(Path(args.input_dir).glob("*.html"))
        for html_file in tqdm(html_files, desc="Processing files"):
            try:
                with open(html_file, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                
                result = extractor.extract_metadata(content)
                if result:
                    yield {
                        "id": str(uuid.uuid4()),
                        "source_file": html_file.name,
                        "resource": args.resource,
                        "doc_type": args.doc_type,
                        "html": result['html'],
                        "tokens": result['tokens'],
                        "xpaths": result['xpaths'],
                        "metadata": result['metadata'],
                        "node_labels": result['node_labels'],
                        "processing_time": result['processing_time']
                    }
            except Exception as e:
                logger.error(f"File processing error: {str(e)}")
    
    dataset = Dataset.from_generator(
        generate_examples,
        features=features
    )
    
    save_path = Path(args.output_dir) / f"{args.resource}_dataset"
    dataset.to_json('output/test.jsonl')
    dataset.save_to_disk(str(save_path))
    logger.info(f"Dataset saved to {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="HTML Metadata Extraction Pipeline (Model-agnostic)")
    parser.add_argument("--input_dir", required=True, help="Directory with HTML files")
    parser.add_argument("--resource", required=True, help="Source resource name")
    parser.add_argument("--doc_type", required=True, help="Document type")
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    parser.add_argument("--label_map", default="label_map.json", help="Path to JSON label map file")
    parser.add_argument("--output_dir", default="./output", help="Output directory")
    
    args = parser.parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
