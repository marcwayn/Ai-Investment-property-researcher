import os
from crewai import Agent, Task, Crew
from crewai_tools import SerperDevTool
from langchain.chat_models import ChatOpenAI

#Using the OpenAI 4o Mini model
llm = ChatOpenAI(model="gpt-4o-mini")

#Serper API setup to run the search on Google
search_tool =SerperDevTool()

researcher = Agent(
    llm=llm,
    role="Senior Property Researcher",
    goal="Find promising investment properties",
    backstory="You are a veteran property analyst. You're looking for retail properties to invest in.",
    allow_delegation=False,
    tools=[search_tool]
)

task1 = Task(
    description="Search the internet and find 5 promising real estate investment suburbs in Washington D.C. " \
    "For each suburb highlighting the mean, low and max prices as well as the rental yield and any potential " \
    "factors that would be useful to know for that area.",
    expected_output="""A detailed report of each of the suburbs. The results should be formated as shown below:
    Suburb 1: Randomplace
    Mean Price: $400,000
    Rental Vacancy: 4.2%
    Rental Yield: 2.9%
    Background Information: These suburns are typically a major transport hub, employment center, and has highly ranked
    educational institutions. The following list highlights some of the top contenders for investment opprotunities""",
    agent=researcher,
    output_file='task1_output.txt'
)

writer = Agent(
    llm=llm,
    role="Senior Property Analyst",
    goal="Summarise property facts into a report.",
    backstory="You're a real estate agent, your goal is compile property analytics into a report for potential investors.",
    allow_delegation=False,
)

task2 = Task(
    description="Summarise the property information into a bullet point list.",
    expected_output="A summarised dot point list of each of the suburbs, prices and important features of the suburb.",
    agent=writer,
    output_file='task2_output.txt'
)

crew = Crew(agents=[researcher, writer], tasks=[task1, task2], verbose=True)

task_output = crew.kickoff()

print(task_output)