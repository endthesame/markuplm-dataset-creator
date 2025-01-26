import argparse
import yaml
import logging
import re
import uuid
from pathlib import Path
from datetime import datetime
from dateutil.parser import parse
from bs4 import BeautifulSoup
from tqdm import tqdm
from transformers import MarkupLMTokenizer
from datasets import Dataset, Features, Value, Sequence, ClassLabel

logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO,
    filename='processor.log'
)
logger = logging.getLogger(__name__)

class MetadataExtractor:
    def __init__(self, config_path, resource, doc_type):
        with open(config_path) as f:
            self.config = yaml.safe_load(f)
        
        self.resource = resource
        self.doc_type = doc_type
        self.tokenizer = MarkupLMTokenizer.from_pretrained("microsoft/markuplm-base")
        self.label_classes = self._get_label_classes()
        
    def _get_label_classes(self):
        labels = set()
        for field in self.config['fields']:
            labels.add(f"B-{field.upper()}")
            labels.add(f"I-{field.upper()}")
        return sorted(list(labels) + ["O"])
    
    def clean_html(self, html):
        soup = BeautifulSoup(html, 'lxml')
        for tag in self.config.get('remove_tags', ['script', 'style', 'nav', 'footer']):
            for el in soup.find_all(tag):
                el.decompose()
        return str(soup)[:self.config.get('max_html_length', 10000)]
    
    def _extract_with_selector(self, soup, selector):
        try:
            if selector['type'] == 'css':
                return soup.select(selector['selector'])
            elif selector['type'] == 'xpath':
                return soup.xpath(selector['selector'])
            elif selector['type'] == 'regex':
                return re.findall(selector['selector'], str(soup))
        except Exception as e:
            logger.error(f"Selector error: {str(e)}")
            return []
    
    def _validate_field(self, value, field_config):
        validators = {
            'date': lambda x: parse(x, fuzzy=False),
            'isbn': lambda x: re.match(r"^\d{3}-\d-\d{3}-\d{5}-\d$", x),
            'year': lambda x: 1900 <= int(x) <= datetime.now().year,
            'list': lambda x: len(x) >= field_config.get('min_items', 1)
        }
        
        for val_type in field_config.get('validate', []):
            if not validators[val_type](value):
                return False
        return True
    
    def process_html(self, html_content):
        try:
            cleaned_html = self.clean_html(html_content)
            soup = BeautifulSoup(cleaned_html, 'lxml')
            metadata = {}
            
            for field, config in self.config['fields'].items():
                values = []
                for selector in config['selectors']:
                    elements = self._extract_with_selector(soup, selector)
                    
                    if selector['strategy'] == 'multi-element':
                        for el in elements:
                            value = el.get_text(strip=True) if selector.get('extract') == 'text' else el.get(selector.get('attr', ''), '')
                            if value and self._validate_field(value, config):
                                values.append(value)
                    
                    elif selector['strategy'] == 'first-match':
                        if elements:
                            value = elements[0].get_text(strip=True) if selector.get('extract') == 'text' else elements[0].get(selector.get('attr', ''), '')
                            if self._validate_field(value, config):
                                values = [value]
                                break
                
                metadata[field] = values[:config.get('max_items', 10)]
            
            bio_tags = self._generate_bio_tags(cleaned_html, metadata)
            return {
                "html": cleaned_html,
                "metadata": metadata,
                "bio_tags": bio_tags
            }
        except Exception as e:
            logger.error(f"Processing error: {str(e)}")
            return None
    
    def _generate_bio_tags(self, html, metadata):
        encoding = self.tokenizer(
            html,
            truncation=True,
            max_length=512,
            return_offsets_mapping=True
        )
        bio_tags = ["O"] * len(encoding.input_ids)
        
        for field, values in metadata.items():
            for value in values:
                for match in re.finditer(re.escape(value), html):
                    start, end = match.start(), match.end()
                    tokens = []
                    for idx, (tok_start, tok_end) in enumerate(encoding.offset_mapping):
                        if tok_start <= start < tok_end or tok_start < end <= tok_end:
                            tokens.append(idx)
                    
                    if tokens:
                        bio_tags[tokens[0]] = f"B-{field.upper()}"
                        for idx in tokens[1:]:
                            bio_tags[idx] = f"I-{field.upper()}"
        return bio_tags

def main(args):
    extractor = MetadataExtractor(args.config, args.resource, args.doc_type)
    
    features = Features({
        "id": Value("string"),
        "resource": Value("string"),
        "doc_type": Value("string"),
        "html": Value("string"),
        "metadata": Sequence({
            field: Sequence(Value("string")) for field in extractor.config['fields']
        }),
        "bio_tags": Sequence(ClassLabel(names=extractor.label_classes))
    })
    
    def generate_dataset():
        html_files = list(Path(args.input_dir).glob("*.html"))
        for html_file in tqdm(html_files, desc="Processing files"):
            with open(html_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            result = extractor.process_html(content)
            if result:
                yield {
                    "id": f"{args.resource}_{args.doc_type}_{uuid.uuid4().hex[:8]}",
                    "resource": args.resource,
                    "doc_type": args.doc_type,
                    "html": result['html'],
                    "metadata": result['metadata'],
                    "bio_tags": result['bio_tags']
                }
    
    dataset = Dataset.from_generator(
        generate_dataset,
        features=features
    )
    
    save_path = Path(args.output_dir) / f"{args.resource}_{args.doc_type}_dataset"
    dataset.save_to_disk(str(save_path))
    logger.info(f"Dataset saved to {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process HTML files into MarkupLM training dataset")
    parser.add_argument("--input_dir", type=str, required=True, help="Directory with HTML files")
    parser.add_argument("--resource", type=str, required=True, help="Resource name (e.g. Springer)")
    parser.add_argument("--doc_type", type=str, required=True, help="Document type (e.g. book)")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")
    parser.add_argument("--output_dir", type=str, default="./output", help="Output directory")
    
    args = parser.parse_args()
    
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)