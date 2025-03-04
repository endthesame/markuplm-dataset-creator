import pytest
import os
from pathlib import Path
from src.processor import MetadataExtractor
from lxml import etree

TEST_DATA_DIR = Path(__file__).parent / 'test_data'

@pytest.fixture
def extractor():
    config_path = TEST_DATA_DIR / 'test_config.yaml'
    label_map_path = TEST_DATA_DIR / 'test_label_map.json'
    return MetadataExtractor(config_path, label_map_path)

@pytest.fixture
def sample_html():
    with open(TEST_DATA_DIR / 'sample.html', 'r', encoding='utf-8') as f:
        return f.read()

def test_basic_extraction(extractor, sample_html):
    result = extractor.extract_metadata(sample_html)
    
    assert result is not None
    assert len(result['tokens']) > 0
    assert len(result['tokens']) == len(result['xpaths']) == len(result['node_labels'])
    
    # Проверка структуры метаданных
    metadata = result['metadata']
    assert 'title' in metadata
    assert 'authors' in metadata
    assert 'date' in metadata
    assert 'abstract' in metadata

def test_title_extraction(extractor, sample_html):
    result = extractor.extract_metadata(sample_html)
    title_data = result['metadata']['title']
    
    # Проверка что собраны данные из обоих селекторов
    assert len(title_data['text']) >= 1
    assert 'Test Article' in title_data['text']
    assert 'Main Title' in title_data['text']
    
    # Проверка BIO разметки
    title_xpaths = title_data['xpaths']
    for xpath in title_xpaths:
        token_indices = [i for i, xp in enumerate(result['xpaths']) if xp == xpath]
        assert all(result['node_labels'][i] in [1, 2] for i in token_indices)

def test_authors_extraction(extractor, sample_html):
    result = extractor.extract_metadata(sample_html)
    authors_data = result['metadata']['authors']
    
    assert len(authors_data['text']) == 2
    assert 'John Doe' in authors_data['text']
    assert 'Jane Smith' in authors_data['text']
    
    # Проверка что разметка включает все токены авторов
    author_labels = [result['node_labels'][i] for xpath in authors_data['xpaths'] 
                    for i, xp in enumerate(result['xpaths']) if xp == xpath]
    assert all(label in [3, 4] for label in author_labels)

def test_date_validation(extractor, sample_html):
    result = extractor.extract_metadata(sample_html)
    date_data = result['metadata']['date']
    
    assert len(date_data['text']) == 1
    assert date_data['text'][0] == '2023-03-15'
    
    # Проверка правильной разметки
    date_labels = [result['node_labels'][i] for xpath in date_data['xpaths'] 
                 for i, xp in enumerate(result['xpaths']) if xp == xpath]
    
    assert date_labels.pop(0) == 5 # I-DATE
    assert all(label == 6 for label in date_labels) # B-DATE

def test_regex_processing(extractor, sample_html):
    result = extractor.extract_metadata(sample_html)
    abstract_data = result['metadata']['abstract']
    
    assert len(abstract_data['text']) > 0
    
    # Проверяем все элементы в списках
    found = False
    for text in abstract_data['text']:
        if isinstance(text, list):
            for item in text:
                if 'test abstract' in item.lower():
                    found = True
                    break
        else:
            if 'test abstract' in text.lower():
                found = True
        if found:
            break
    
    assert found, "Expected 'test abstract' not found in processed text"
    
    node_labesls = result["node_labels"]
    assert 7 in node_labesls  # B-ABSTRACT
    assert 8 in node_labesls  # I-ABSTRACT

def test_html_cleaning(extractor):
    dirty_html = "<script>alert()</script><div>Clean<style type='text/css'>style</style><p>text</p></div>"
    cleaned = extractor._clean_html(dirty_html)
    assert 'script' not in cleaned
    assert 'style' not in cleaned
    assert '<p>' in cleaned
    assert "<div>Clean<p>text</p></div>" in cleaned

def test_value_processing(extractor):
    test_cases = [
        ('  Test Value  ', {'post_process': ['strip']}, 'Test Value'),
        ('TEST', {'post_process': ['lower']}, 'test'),
        ('<test>', {'post_process': ['unescape']}, '<test>'),
    ]
    
    for raw, config, expected in test_cases:
        processed = extractor._process_value(raw, config)
        assert processed == expected