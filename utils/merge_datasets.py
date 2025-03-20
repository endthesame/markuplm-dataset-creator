import os
import argparse
from datasets import load_from_disk as load_dataset, Dataset, concatenate_datasets

def load_datasets(input_dir, load_format):
    """Загружает датасеты из указанной папки в указанном формате"""
    datasets = []
    
    if load_format not in ['disk', 'jsonl']:
        raise ValueError("Неверный формат загрузки: выберите disk или jsonl")
    
    for entry in os.listdir(input_dir):
        entry_path = os.path.join(input_dir, entry)
        
        if load_format == 'disk':
            # Загрузка из подпапок (формат save_to_disk)
            if os.path.isdir(entry_path):
                try:
                    ds = load_dataset(entry_path)
                    datasets.append(ds)
                except Exception as e:
                    print(f"Ошибка загрузки {entry_path}: {str(e)}")
        else:
            # Загрузка из JSONL файлов
            if entry.endswith('.jsonl'):
                try:
                    ds = Dataset.from_json(entry_path)
                    datasets.append(ds)
                except Exception as e:
                    print(f"Ошибка загрузки {entry_path}: {str(e)}")
                
    return datasets

def merge_datasets(datasets):
    """Объединяет список датасетов"""
    if not datasets:
        return None
    return concatenate_datasets(datasets)

def save_merged_dataset(dataset, output_dir, save_format='both'):
    """Сохраняет объединенный датасет"""
    if not dataset:
        print("Нет данных для сохранения")
        return
    
    os.makedirs(output_dir, exist_ok=True)
    
    if save_format in ['jsonl', 'both']:
        json_path = os.path.join(output_dir, 'merged_dataset.jsonl')
        dataset.to_json(json_path)
        
    if save_format in ['disk', 'both']:
        disk_path = os.path.join(output_dir, 'merged_dataset')
        dataset.save_to_disk(disk_path)

def main():
    parser = argparse.ArgumentParser(description='Объединение датасетов')
    parser.add_argument('--input_dir', '-i', type=str, required=True, 
                       help='Путь к папке с исходными датасетами')
    parser.add_argument('--output_dir', '-o', type=str, required=True, 
                       help='Путь для сохранения объединенного датасета')
    parser.add_argument('--load_format', '-l', choices=['disk', 'jsonl'], 
                       required=True,
                       help='Формат загрузки: disk (подпапки) или jsonl')
    parser.add_argument('--save_format', '-s', choices=['jsonl', 'disk', 'both'], 
                       default='both', 
                       help='Формат сохранения (по умолчанию: оба)')
    
    args = parser.parse_args()

    # Загрузка датасетов
    datasets = load_datasets(args.input_dir, args.load_format)
    if not datasets:
        print("Не найдены датасеты для объединения")
        return
    
    # Объединение
    merged_dataset = merge_datasets(datasets)
    
    # Сохранение
    save_merged_dataset(merged_dataset, args.output_dir, args.save_format)
    print(f"Объединенный датасет сохранен в {args.output_dir}")

if __name__ == "__main__":
    main()