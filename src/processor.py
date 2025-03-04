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
from .FeatureHTMLExtractor import HTMLExtractor
# Настройка логирования
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO,
    filename='logs/processor.log'
)
logger = logging.getLogger(__name__)

class MetadataExtractor:
    def __init__(self, config_path, label_map_path):
        # Загрузка конфигурации
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        logger.info(f"Config loaded from: {config_path}")

        # Загрузка маппинга меток
        with open(label_map_path, 'r', encoding='utf-8') as f:
            self.label_map = json.load(f)
        logger.info(f"Label map loaded from: {label_map_path} with {self.label_map['label2id']}")

        self.label2id = self.label_map["label2id"]
        self.bio_labels = {v: k for k, v in self.label2id.items()}
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
                pattern = selector_config['regex']
                #matches = list(re.finditer(pattern, value))
                #value = [match.group() for match in matches] - ??? формат строки нужен
                matches = re.findall(pattern, value)
                value = matches[0] if matches else ""
            
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

    def _generate_bio_labels(self, token_indices, label):
        """Генерация BIO-меток для последовательности токенов"""
        bio_labels = []
        for i, idx in enumerate(token_indices):
            prefix = 'B' if i == 0 else 'I'
            bio_label = f"{prefix}-{label.upper()}"
            bio_labels.append(self.label2id[bio_label])
        return bio_labels

    def extract_metadata(self, html_content):
        """Основной метод извлечения метаданных"""
        try:
            # Шаг 1: Подготовка базовых данных
            html_extractor = HTMLExtractor(
                html_content,
                extract_text=True,
                extract_attributes=True,
                target_attrs={'content', 'href', 'src', 'value', "name"},
                exclude_tags=self.remove_tags,
                split_tokens=True,
                split_token_pattern=r'\S+' #'\w+'
            )
            all_tokens, all_xpaths = html_extractor.extract()
            tree = etree.fromstring(html_content, etree.HTMLParser(encoding='utf-8'))

            # Шаг 2: Извлечение метаданных
            metadata, field_xpath_map = self._extract_fields_metadata(tree, html_extractor)
            
            # Шаг 3: Генерация разметки
            node_labels = self._generate_bio_labels(all_tokens, all_xpaths, field_xpath_map)
            
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

    def _extract_fields_metadata(self, tree, html_extractor):
        """Извлечение метаданных для всех полей конфигурации"""
        metadata = {}
        field_xpath_map = {}

        for field_name, field_config in self.config['fields'].items():
            field_data = {"text": [], "xpaths": []}
            collected_xpaths = set()

            for selector in field_config['selectors']:
                try:
                    # Выбор элементов
                    elements = []
                    if selector['type'] == 'css':
                        elements = tree.cssselect(selector['selector'])
                    elif selector['type'] == 'xpath':
                        elements = tree.xpath(selector['selector'])

                    # Стратегия выбора
                    if selector.get('strategy', 'first').lower() == 'first':
                        elements = elements[:1]

                    # Обработка элементов
                    for element in elements:
                        # Извлечение значения
                        if selector.get('extract') == 'attr':
                            attr_name = selector.get('attr_name', '')
                            raw_value = element.get(attr_name, '')
                            xpath = html_extractor.get_xpath(element=element, attr=attr_name)
                        elif selector.get('extract') == 'innerText':
                            raw_value = ''.join(element.itertext()).strip()
                            xpath = html_extractor.get_xpath(element=element)
                        else:
                            raw_value = element.text if hasattr(element, 'text') else ''
                            xpath = html_extractor.get_xpath(element=element)

                        value = self._process_value(raw_value, selector)
                        
                        # Валидация и сохранение
                        if value and self._validate_value(value, field_config) and xpath not in collected_xpaths:
                            field_data["text"].append(value)
                            field_data["xpaths"].append(xpath)
                            collected_xpaths.add(xpath)

                except Exception as e:
                    logger.error(f"Error processing selector {selector}: {str(e)}")

            metadata[field_name] = field_data
            field_xpath_map[field_name] = field_data["xpaths"]
        
        return metadata, field_xpath_map

    def _generate_bio_labels(self, all_tokens, all_xpaths, field_xpath_map):
        """Генерация BIO-разметки с учётом дочерних элементов"""
        node_labels = [self.label2id.get("O", 0)] * len(all_tokens)
        
        for field_name, xpaths in field_xpath_map.items():
            bio_field = field_name.upper()
            field_config = self.config['fields'][field_name]
            
            for base_xpath in xpaths:
                # Нормализация XPath для сравнения
                base_xpath = base_xpath.rstrip('/')
                
                # Находим все токены, чьи XPaths начинаются с базового
                token_indices = [
                    i for i, xp in enumerate(all_xpaths) 
                    if xp.startswith(base_xpath) and (
                        xp == base_xpath or 
                        xp.startswith(f"{base_xpath}/") or 
                        xp.startswith(f"{base_xpath}[")
                    )
                ]
                
                if not token_indices:
                    continue

                # Обработка regex поверх всего текста
                if 'regex' in field_config["selectors"][0]:
                    combined_text = ' '.join(all_tokens[i] for i in token_indices)
                    matches = list(re.finditer(field_config["selectors"][0]["regex"], combined_text))
                    for match in matches:
                        start, end = match.span()
                        matched_indices = []
                        current_pos = 0
                        
                        for idx in token_indices:
                            token = all_tokens[idx]
                            token_len = len(token)
                            
                            # Проверяем пересечение границ токена с матчем
                            if (current_pos <= start < current_pos + token_len) or \
                            (current_pos < end <= current_pos + token_len) or \
                            (start <= current_pos and current_pos + token_len <= end):
                                matched_indices.append(idx)
                                
                            current_pos += token_len + 1  # +1 для пробела
                        
                        if matched_indices:
                            labels = [f"B-{bio_field}"] + [f"I-{bio_field}"] * (len(matched_indices) - 1)
                            for i, label in zip(matched_indices, labels):
                                node_labels[i] = self.label2id.get(label, 0)
                else:
                    labels = [f"B-{bio_field}"] + [f"I-{bio_field}"] * (len(token_indices) - 1)
                    for i, label in zip(token_indices, labels):
                        node_labels[i] = self.label2id.get(label, 0)
        return node_labels


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
        "node_labels": Sequence(Value("int64")),
        "processing_time": Value("string")
    })
    
    def generate_examples():
        html_files = list(Path(args.input_dir).glob("*.html"))
        total_files = len(html_files)
         
        for i, html_file in enumerate(tqdm(html_files, desc="Processing files"), start=1):
            try:
                logger.info(f"Processing {i}/{total_files}: file {html_file}")
                with open(html_file, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                
                result = extractor.extract_metadata(content)
                if result:
                    names_of_empty_fields = [
                        field_name
                        for field_name, field_data in result["metadata"].items()
                        if not field_data["text"]
                    ]

                    if names_of_empty_fields:
                        logging.warning(f"File {html_file} contains empty fields: {names_of_empty_fields}")

                    logger.info(f"{i}/{total_files}: file {html_file} succeeded")
                     
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
                logger.error(f"Error processing file {html_file}: {str(e)}")
     
    dataset = Dataset.from_generator(
        generate_examples,
        features=features
    )
    
    save_path = Path(args.output_dir) / f"{args.resource}_dataset"
    dataset.to_json('output/test.jsonl', num_proc=args.num_proc)
    dataset.save_to_disk(str(save_path), num_proc=args.num_proc)
    logger.info(f"Dataset saved to {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="HTML Metadata Extraction Pipeline (Model-agnostic)")
    parser.add_argument("--input_dir", required=True, help="Directory with HTML files")
    parser.add_argument("--resource", required=True, help="Source resource name")
    parser.add_argument("--doc_type", required=True, help="Document type")
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    parser.add_argument("--label_map", default="label_map.json", help="Path to JSON label map file")
    parser.add_argument("--output_dir", default="./output", help="Output directory")
    parser.add_argument("--num_proc", default=4, type=int, help="Number of processes for saving data")
    
    args = parser.parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)