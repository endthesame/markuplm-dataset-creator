## Создание датасета для модели [MarkupLM](https://huggingface.co/docs/transformers/model_doc/markuplm)

### Запуск формирование коллекции
``` bash
python -m src.processor --input_dir data/<coll_name> --resource <resource_name> --doc_type <article/books/conf/etc> --config <config_path> --num_proc <num of proc>
```

### Запуск тестов
```bash
python -m pytest tests/ -v
```