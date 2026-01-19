import requests
from typing import List, Dict, Optional
from bs4 import BeautifulSoup
import re


class SearchEngine:
    def __init__(self):
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
    
    def search(self, query: str, num_results: int = 3) -> List[Dict]:
        try:
            search_url = f"https://duckduckgo.com/html/?q={query}"
            response = requests.get(search_url, headers=self.headers, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            results = []
            
            result_divs = soup.find_all('div', class_='result')
            
            for div in result_divs[:num_results]:
                try:
                    title_tag = div.find('a', class_='result__a')
                    snippet_tag = div.find('a', class_='result__snippet')
                    url_tag = div.find('a', class_='result__url')
                    
                    if title_tag and snippet_tag:
                        title = title_tag.get_text(strip=True)
                        url = url_tag.get('href', '') if url_tag else ''
                        snippet = snippet_tag.get_text(strip=True)
                        
                        if title and snippet:
                            results.append({
                                'title': title,
                                'url': url,
                                'snippet': snippet
                            })
                    
                    if len(results) >= num_results:
                        break
                        
                except Exception as e:
                    continue
            
            return results
            
        except Exception as e:
            print(f"Search error: {e}")
            return []
    
    def get_answer_from_results(self, query: str, results: List[Dict]) -> Dict:
        if not results:
            return {
                'answer': '抱歉，我没有找到相关的答案。请尝试换个问题或联系人工客服。',
                'source': None,
                'confidence': 0.0
            }
        
        best_result = results[0]
        answer = best_result['snippet']
        
        if len(answer) > 300:
            answer = answer[:300] + '...'
        
        return {
            'answer': f'根据搜索结果：{answer}',
            'source': best_result['title'],
            'url': best_result['url'],
            'confidence': 0.3
        }
    
    def search_and_answer(self, query: str) -> Dict:
        results = self.search(query, num_results=3)
        return self.get_answer_from_results(query, results)


search_engine = SearchEngine()
