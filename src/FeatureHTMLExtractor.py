from lxml import html
from typing import List, Tuple, Optional, Set, Pattern, Union
import re

class HTMLExtractor:
    def __init__(
        self,
        html_content: str,
        extract_text: bool = True,
        extract_attributes: bool = True,
        target_attrs: Optional[Set[str]] = None,
        exclude_pattern: Pattern = re.compile(r'^\s*$'),
        exclude_tags: Set[str] = {'script', 'style', 'noscript'},
        split_tokens: bool = False,
        split_token_pattern: Union[str, Pattern] = r'\w+',
    ):
        """
        Универсальный HTML-экстрактор текста с соответствующими XPath-путями

        :param html_content: HTML-контент для парсинга
        :param extract_text: Флаг извлечения текстового содержимого элементов
        :param extract_attributes: Флаг извлечения содержимого атрибутов
        :param target_attrs: Множество целевых атрибутов для извлечения (None - все атрибуты)
        :param exclude_pattern: Регулярное выражение для фильтрации нежелательного текста
        :param exclude_tags: Множество тегов, которые следует игнорировать
        :param split_tokens: Флаг включения разделения текста на отдельные токены
        :param split_token_pattern: Паттерн для поиска токенов (регулярное выражение)
        """
        self.tree = html.fromstring(html_content)
        self.extract_text = extract_text
        self.extract_attributes = extract_attributes
        self.target_attrs = target_attrs
        self.exclude_pattern = exclude_pattern
        self.exclude_tags = exclude_tags
        self.split_tokens = split_tokens
        
        # Инициализация паттерна для разделения токенов
        if isinstance(split_token_pattern, str):
            self.split_token_pattern = re.compile(split_token_pattern)
        else:
            self.split_token_pattern = split_token_pattern
        
        self._texts: List[str] = []
        self._xpaths: List[str] = []

    def get_xpath(self, element, attr: Optional[str] = None) -> str:
        """
        Генерирует XPath-путь для элемента с использованием встроенного метода getpath.
        Если указан атрибут, к пути добавляется суффикс /@атрибут.

        :param element: Элемент, для которого генерируется путь
        :param attr: Опциональный атрибут для включения в путь
        :return: Строка с XPath-путём
        """
        xpath = element.getroottree().getpath(element)
        return f"{xpath}/@{attr}" if attr else xpath

    def _should_include(self, text: str) -> bool:
        """
        Проверяет, соответствует ли текст критериям включения.

        :param text: Проверяемый текст
        :return: True, если текст должен быть включен в результат
        """
        return not self.exclude_pattern.match(text.strip())

    def _process_element_text(self, element):
        """
        Обрабатывает текстовое содержимое элемента, добавляя его в результаты,
        если оно соответствует критериям.

        :param element: Обрабатываемый HTML-элемент
        """
        if element.tag in self.exclude_tags:
            return

        # Если элемент !(не имеет дочерних элементов) и содержит текст
        if element.text:
            text = element.text.strip()
            if self._should_include(text):
                self._texts.append(text)
                self._xpaths.append(self.get_xpath(element))

        # Обработка tail-текста (после закрывающего тега элемента)
        self._process_element_tail(element)

    def _process_element_tail(self, element):
        """Обрабатывает tail-текст элемента"""
        if element.tail:
            tail_text = element.tail.strip()
            if self._should_include(tail_text):
                # Tail принадлежит родительскому элементу
                parent = element.getparent()
                if parent is not None and parent.tag not in self.exclude_tags:
                    self._texts.append(tail_text)
                    self._xpaths.append(self.get_xpath(parent))  # XPath родителя!

    def _process_element_attributes(self, element):
        """
        Обрабатывает атрибуты элемента, добавляя их значения в результаты,
        если они соответствуют критериям.

        :param element: Обрабатываемый HTML-элемент
        """
        if element.tag in self.exclude_tags:
            return

        for attr in (self.target_attrs if self.target_attrs is not None else element.attrib.keys()):
            if attr in element.attrib:
                value = element.attrib[attr]
                # Обычно значение атрибута является строкой, но на всякий случай проверяем
                if isinstance(value, list):
                    value = ' '.join(value)
                if value and self._should_include(value):
                    self._texts.append(value.strip())
                    self._xpaths.append(self.get_xpath(element, attr))

    def _process_element(self, element):
        """
        Основной метод обработки элемента, делегирующий задачи
        обработки текста и атрибутов.

        :param element: Обрабатываемый HTML-элемент
        """
        if self.extract_text:
            self._process_element_text(element)
        if self.extract_attributes:
            self._process_element_attributes(element)

    def _separate_elems_tokens(self):
        """Разбивает выделенный текст из элемента по токенам"""
        new_texts = []
        new_xpaths = []
        for text, xpath in zip(self._texts, self._xpaths):
            # Ищем все совпадения с паттерном
            tokens = self.split_token_pattern.findall(text)
            for token in tokens:
                # Применяем фильтрацию к каждому подтокену
                if self._should_include(token):
                    new_texts.append(token)
                    new_xpaths.append(xpath)
        self._texts = new_texts
        self._xpaths = new_xpaths


    def extract(self) -> Tuple[List[str], List[str]]:
        """
        Запускает процесс извлечения данных из HTML.
        """
        self._texts = []
        self._xpaths = []

        for element in self.tree.iter():
            self._process_element(element)

        # Постобработка для разделения токенов
        if self.split_tokens:
            self._separate_elems_tokens()

        return self.texts, self.xpaths

    @property
    def texts(self) -> List[str]:
        """
        Возвращает копию списка извлеченных текстов.

        :return: Список текстовых значений
        """
        return self._texts.copy()

    @property
    def xpaths(self) -> List[str]:
        """
        Возвращает копию списка XPath-путей.

        :return: Список XPath-строк
        """
        return self._xpaths.copy()
