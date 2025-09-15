import os
import warnings
from dotenv import load_dotenv
from crewai import Agent, Task, Crew
from crewai.process import Process

from crewai_tools import (SerperDevTool, WebsiteSearchTool)
from langchain_openai import ChatOpenAI


load_dotenv()


llm = ChatOpenAI(model="gpt-4o")

search_tool = SerperDevTool()
web_rang_tool = WebsiteSearchTool()
# scrap_tool = ScrapeWebsiteTool()

researcher = Agent(
    role="React, Next.js,Typescript를 사용하는 프론트엔드 개발자",
    goal="프론트엔드의 최신 기술 트렌드를 한국어로 제공합니다. 지금은 2025년 9월 입니다.",
    backstory="기술 트렌드에 예리한 안목을 지닌 전문가이자 프론트엔드 개발자입니다.",
    tools=[search_tool, web_rang_tool],
    max_iter=5,
    llm=llm
)

writer = Agent(
    role="뉴스레터 작성자",
    goal="최신 프론트엔드(React, Next.js, Typescript) 기술 트렌드에 대한 매력적인 테크 뉴스레터를 한국어로 작성하세요. 지금은 2025년 9월입니다.",
    backstory="기술에 대한 열정을 가진 숙련된 작가입니다.",
    verbose=True,
    allow_delegation=False,
    llm=llm
)

# Define Task
research = Task(
    description="프론트엔드(React, Next.js, Typescript) 최신 기술 동향을 조사하고 요약을 제공하세요.",
    expected_output="프론트엔드 업계에서 주목받는 기술 개발 동향을 요약한 글",
    agent=researcher
)

write = Task(
    description="프론트엔드(React, Next.js, Typescript) 최신 기술에 대해 매력적인 테크 뉴스레터를 작성하세요. 테크 뉴스레터이므로 전문적인 용어를 사용해도 괜찮습니다.",
    expected_output="최신 기술 관련 소식을 재밌는 말투로 소개하는 4문단짜리 마크다운 형식 뉴스레터",
    agent=writer,
    output_file="./md/tech_newsletter.md"
)

crew = Crew(
    agents=[researcher, writer],
    tasks=[research, write],
    verbose=True,
    process=Process.sequential
)

result = crew.kickoff()

result
