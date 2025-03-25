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

        # Определение ожидаемых полей на основе label_map
        self.expected_fields = set()
        for label in self.label_map["label2id"].keys():
            if label == "O":
                continue
            if '-' in label:
                field = label.split('-', 1)[1].lower()
                self.expected_fields.add(field)
        self.expected_fields = sorted(list(self.expected_fields))

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
            metadata, list_of_node_maps_data = self._extract_fields_metadata(tree, html_extractor)

            # Добавление отсутствующих полей с пустыми значениями из label map
            for field in self.expected_fields:
                if field not in metadata:
                    metadata[field] = {"text": [], "xpaths": []}
            
            # Шаг 3: Генерация разметки
            node_labels = self._generate_bio_labels(all_tokens, all_xpaths, list_of_node_maps_data)
            
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
        list_of_node_maps_data = [] #used as a list for node labeling

        for field_name, field_config in self.config['fields'].items():
            node_maps = {"field_name": field_name, "data": []}
            
            field_data = {"text": [], "xpaths": []}
            collected_xpaths = set()

            for selector in field_config['selectors']:
                selectors_xpath_map = {"selector": selector, "xpaths": []}
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

                            selectors_xpath_map["xpaths"].append(xpath)

                    node_maps["data"].append(selectors_xpath_map)

                except Exception as e:
                    logger.error(f"Error processing selector {selector}: {str(e)}")

            metadata[field_name] = field_data
            list_of_node_maps_data.append(node_maps)
        return metadata, list_of_node_maps_data

    def _generate_bio_labels(self, all_tokens, all_xpaths, list_of_node_maps_data):
        """Генерация BIO-меток с улучшенной обработкой regex и фильтрацией токенов."""
        node_labels = [self.label2id.get("O", 0)] * len(all_tokens)

        for record in list_of_node_maps_data:
            field_name = record["field_name"]  # Достаём название поля
            data = record["data"]  # Достаём список селекторов

            bio_field = field_name.upper()

            for selector in data:
                extract_type = selector.get("selector", {}).get("extract", "text")
                is_inner_text = extract_type == "innerText"

                for base_xpath in selector.get("xpaths", []):
                    base_xpath = base_xpath.rstrip("/")
                        
                    # Фильтрация токенов: исключаем атрибуты для innerText
                    token_indices = [
                        i for i, xp in enumerate(all_xpaths)
                        if xp.startswith(base_xpath) and (
                            xp == base_xpath or 
                            xp.startswith(f"{base_xpath}/") or 
                            xp.startswith(f"{base_xpath}[")
                        )
                        and (not (is_inner_text and '/@' in xp))
                    ]
                    
                    if not token_indices:
                        continue

                    # Обработка regex
                    if 'regex' in selector.get('selector'):
                        combined_text = ' '.join(all_tokens[i] for i in token_indices)
                        matches = list(re.finditer(selector.get('selector')['regex'], combined_text))
                        
                        for match in matches:
                            # Определяем группы для обработки
                            groups = [0] if not match.groups() else range(1, len(match.groups()) + 1)
                            
                            for group_idx in groups:
                                start, end = match.span(group_idx)
                                matched_indices = []
                                current_pos = 0
                                
                                # Сопоставляем позиции в combined_text с токенами
                                for idx in token_indices:
                                    token = all_tokens[idx]
                                    token_len = len(token)
                                    token_end = current_pos + token_len
                                    
                                    if (current_pos <= start < token_end) or \
                                    (current_pos < end <= token_end) or \
                                    (start <= current_pos and token_end <= end):
                                        matched_indices.append(idx)
                                    
                                    current_pos += token_len + 1  # Учет пробелов
                                
                                # Назначаем метки
                                if matched_indices:
                                    labels = [f"B-{bio_field}"] + [f"I-{bio_field}"] * (len(matched_indices) - 1)
                                    for i, label in zip(matched_indices, labels):
                                        if node_labels[i] == self.label2id.get("O", 0):
                                            node_labels[i] = self.label2id.get(label, 0)

                    # Стандартная разметка, если regex не сработал
                    else:
                        labels = [f"B-{bio_field}"] + [f"I-{bio_field}"] * (len(token_indices) - 1)
                        for i, label in zip(token_indices, labels):
                            if node_labels[i] == self.label2id.get("O", 0):
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
            }) for field in extractor.expected_fields #for all possible fields in label map
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
    
    save_path = Path(args.output_dir) / f"{args.resource}_{args.doc_type}"
    dataset.to_json(f"output/{args.resource}_{args.doc_type}.jsonl", num_proc=args.num_proc)
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