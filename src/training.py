from transformers import (
    MarkupLMProcessor,
    MarkupLMForTokenClassification,
    TrainingArguments,
    Trainer
)
from datasets import load_from_disk
import torch

# 1. Загрузка датасета
dataset = load_from_disk("")

# 2. Подготовка меток
label_list = dataset["train"].features["node_labels"].feature.names
label2id = {label: i for i, label in enumerate(label_list)}

# 3. Загрузка модели и процессора
processor = MarkupLMProcessor.from_pretrained("microsoft/markuplm-base")
model = MarkupLMForTokenClassification.from_pretrained(
    "microsoft/markuplm-base",
    num_labels=len(label_list),
    id2label={v: k for k, v in label2id.items()},
    label2id=label2id
)

# 4. Функция для обработки примеров
def process_examples(examples):
    processed = processor(
        html_strings=examples["html"],
        nodes=examples["tokens"],
        xpaths=examples["xpaths"],
        node_labels=examples["node_labels"],
        padding="max_length",
        truncation=True,
        max_length=512,
        return_tensors="pt"
    )
    return processed

# 5. Применение обработки к датасету
tokenized_dataset = dataset.map(
    process_examples,
    batched=True,
    remove_columns=dataset["train"].column_names,
    num_proc=4
)

# 6. Настройка параметров обучения
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    learning_rate=5e-5,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=50,
    save_steps=500,
    evaluation_strategy="epoch",
    load_best_model_at_end=True,
    fp16=True if torch.cuda.is_available() else False,
    report_to="none"
)

# 7. Создание Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["validation"] if "validation" in tokenized_dataset else None,
)

# 8. Запуск обучения
trainer.train()