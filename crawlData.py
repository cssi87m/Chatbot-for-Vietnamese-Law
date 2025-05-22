import requests
from bs4 import BeautifulSoup
from tqdm import tqdm
import json
import time

BASE_URL = "https://thuvienphapluat.vn/hoi-dap-phap-luat/tim-tu-van?searchType=1&q=an+to%C3%A0n+th%C3%B4ng+tin&searchField=0"
START_PAGE = 1
END_PAGE = 1  # Crawl từ page 1 đến 5, có thể mở rộng
KEYWORDS = ["an ninh mạng", "an toàn thông tin", "bảo mật", "công nghệ cao", "an toàn hệ thống", "an ninh thông tin", "phần mềm độc hại"]

def is_relevant(text: str) -> bool:
    text = text.lower()
    return any(kw in text for kw in KEYWORDS)

def crawl_question_list(page_num):
    url = f"{BASE_URL}&page={page_num}"
    response = requests.get(url)
    soup = BeautifulSoup(response.content, "html.parser")
    
    # Giả sử mỗi câu hỏi nằm trong thẻ <a class="question-link" href="...">
    question_links = soup.select("article.news-card tvpl-find")
    
    return [BASE_URL + link["href"] for link in question_links]

def crawl_question_detail(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.content, "html.parser")

    article = soup.select_one("div.col-md-9")

    # question_text = soup.select_one("strong.d-block mt-3 mb-3 sapo").text.strip()
    # answer_text = soup.select_one("div.answer-content").text.strip()

    # return {
    #     "url": url,
    #     "question": question_text,
    #     "answer": answer_text
    # }
    return article

def main():
    results = []

    for page in tqdm(range(START_PAGE, END_PAGE + 1)):
        question_links = crawl_question_list(page)

        for link in question_links:
            try:
                item = crawl_question_detail(link)
                print(item)
                break
                if is_relevant(item["question"]):
                    results.append(item)
                    print(f"✓ Collected: {item['question'][:60]}...")
                else:
                    print(f"- Skipped (irrelevant): {item['question'][:60]}...")
            except Exception as e:
                print(f"Error with {link}: {e}")
            time.sleep(1)  # tránh bị chặn

    # with open("cyberlaw_questions.json", "w", encoding="utf-8") as f:
    #     json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\n✅ Crawl hoàn tất. Tổng số câu hỏi hợp lệ: {len(results)}")

if __name__ == "__main__":
    main()
