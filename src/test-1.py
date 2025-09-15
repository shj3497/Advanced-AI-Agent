from crewai import Agent, Task, Crew
from langchain_openai import ChatOpenAI
import os
from dotenv import load_dotenv

load_dotenv()

# 목차 설정 에이전트
outline_generator = Agent(
    role="Outline Generator",
    goal="Create structured outlines for articles on given topics. answer in Korean",
    llm=ChatOpenAI(model="gpt-4o-mini", max_tokens=1000),
    backstory="You ar an expert at organizing information and creating comprehensive outlines for various subjets."
)

# 본문 작성 에이전트
writer = Agent(
    role="Writer",
    goal="Create engaging content based on research. answer in Korean",
    llm=ChatOpenAI(model="gpt-4o", max_tokens=3000),
    backstory="You ar a skilled writer who can transform complex information into readable content."
)


# Task 정의
outline_task = Task(
    description="Create a detailed outline for an article about AI\'s impact on job markets.",
    agent=outline_generator,
    expected_output="A comprehensive outline covering the main aspects of AI\'s influence on employment."
)

writing_task = Task(
    description="Write an article about the findings from the research.",
    agent=writer,
    expected_output="An engaging article discussing AI\'s influence on job markets."
)

# Crew 정의
ai_impact_crew = Crew(
    agents=[outline_generator, writer],
    tasks=[outline_task, writing_task],
    verbose=True
)

result = ai_impact_crew.kickoff()

result
