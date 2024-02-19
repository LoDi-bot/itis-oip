import json

import requests
from bs4 import BeautifulSoup
import os

# конфигурационные переменные
# сайт для получения ссылок
MAIN_LINK = 'https://ru.wikipedia.org/wiki/Википедия:Список_хороших_статей/'
BASE_URL = 'https://ru.wikipedia.org'
# количество нужных ссылок
COUNT_PAGES = 5
# файл для сохранения информации о ссылке и файла "выкачки"
INFO_FILE = 'index.txt'

index = {}


def get_links(url):
    page = requests.get(url)
    data = page.text
    soup = BeautifulSoup(data, features="lxml")
    links = []
    for row in soup.find('table', {'class': 'standard'}).find_all('tr'):
        href = row.find_next('a').get('href')
        links.append(f'{BASE_URL}{href}')
    return links


# Функция для удаления html-тегов
def remove_tags(html):
    soup = BeautifulSoup(html, "html.parser")
    for data in soup(['style', 'script', 'noscript', 'link']):
        # Удаление тегов
        data.decompose()
    return str(soup)


def crawl(url):
    page = requests.get(url)
    data = page.text
    return remove_tags(data)


def save_index(text_file_path):
    dump = json.dumps(index,
                      sort_keys=False,
                      indent=4,
                      ensure_ascii=False,
                      separators=(',', ': '))
    with open(text_file_path, "w") as f:
        f.write(dump)


if __name__ == '__main__':
    links_all = []
    info_string = ""
    year = 2006
    while len(links_all) < COUNT_PAGES:
        current_link = f'{MAIN_LINK}{year}'
        links = get_links(current_link)
        links_all += links
        year += 1
    for i, link in enumerate(links_all):
        html_text = crawl(link)
        index[i] = link
        filename = f'00{i}' if i < 10 else f'0{i}'
        info_string += f"{filename}\t{link}\n"
        path_result = f"Выкачка/{filename}.txt"
        os.makedirs(os.path.dirname(path_result), exist_ok=True)
        with open(path_result, "w", encoding="utf-8") as file_result:
            file_result.write(html_text)
    with open(INFO_FILE, "w", encoding="utf-8") as f:
        f.write(info_string)
    save_index('index.json')

    # page = requests.get('https://ru.wikipedia.org/wiki/Википедия:Список_хороших_статей/2006')
    # data = page.text
    # soup = BeautifulSoup(data, features="lxml")
    # links = []
    # for row in soup.find('table', { 'class' : 'standard' }).find_all('tr'):
    #     href = row.find_next('a').get('href')
    #     links.append(f'{BASE_URL}{href}')
    # print(links)
