import os
from dotenv import load_dotenv
from langchain_community.llms import Tongyi
from langchain.prompts import PromptTemplate

class ParseLLM:
    def __init__(self,model_name,api_key):
        self.llm=Tongyi(model=model_name,api_key=api_key,streaming=True)
        # 解析引用，对应建树方法2
        self.parse_prompt_template = PromptTemplate(template="""
            <document>{document}</document>
            
            Extract the following information from the document above and return it in JSON format:
            - title:Title of the paper
            - abstract:Abstract of the paper
            - references: List of titles from the "references" section (exact strings as they appear)

            Output example:
            {{
                "title": "Attention Is All You Need",
                "publication_date": "2017-12-01",
                "references": [
                    "Neural Machine Translation by Jointly Learning to Align and Translate",
                    "ImageNet Classification with Deep Convolutional Neural Networks"
                ]
            }}
        """)
    
    def response(self,prompt):
        response=self.llm.invoke(prompt)
        return response
    
    def parse_paper(self,paper):
        prompt=self.parse_prompt_template.format(document=paper)
        response=self.response(prompt)
        return response




# 加载 .env 文件
load_dotenv()
dashscope_api_key = os.getenv("DASHSCOPE_API_KEY")

