from pydantic import BaseModel
from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers.pydantic import PydanticOutputParser
from langchain_core.outputs import Generation

template = """
    Evaluate the performance of the employee based on the following data on sprint {sprint_id}:
    Tasks Completed on this sprint (2 weeks): {tasks}
    1 on 1 Notes: {meeting_notes}
    Provide a summary of performance_summary, high_lighting, strengths_areas_for_improvement and reason for can_be_laid_off or not.
    in the following JSON format:
    {{
        "sprint_id": "{sprint_id}",
        "performance_summary": "<summary>",  // can be str or list[str]
        "high_lighting": "<highlighting>",  // can be str or list[str]
        "strengths_areas_for_improvement": "<strengths_areas_for_improvement>",  // can be str or list[str]
        "can_be_laid_off": "<reason>"  // can be str or list[str] 
    }}
    """


class Task(BaseModel):
    task_id: str
    title: str
    description: str
    status: str
    created_at: str
    started_at: str | None
    completed_at: str | None
    duration: str
    points: int


class Sprint(BaseModel):
    sprint_id: str
    tasks: list[Task]


class MeetingNote(BaseModel):
    topic: str
    discussion: str


class Meeting(BaseModel):
    employee_id: str
    date: str
    notes: list[MeetingNote]


class PerformanceResult(BaseModel):
    sprint_id: str
    performance_summary: str | list[str]
    high_lighting: str | list[str]
    strengths_areas_for_improvement: str | list[str]
    can_be_laid_off: str | list[str]


def evaluate_performance(sprints_data: Sprint, meeting_data: Meeting) -> PerformanceResult:
    model = OllamaLLM(model="llama3.2")

    parser = PydanticOutputParser(pydantic_object=PerformanceResult)

    prompt = ChatPromptTemplate.from_template(
        template,
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )

    chain = prompt | model

    tasks_summary = "\n".join([f"- {task.title}: {task.status} ({task.duration})" for task in sprints_data.tasks])
    meeting_summary = "\n".join([f"- {note.topic}: {note.discussion}" for note in meeting_data.notes])

    input_data = {
        "sprint_id": sprints_data.sprint_id,
        "tasks": tasks_summary,
        "meeting_notes": meeting_summary
    }

    result_text = chain.invoke(input_data)
    result = [Generation(text=result_text)]
    parsed_result = parser.parse_result(result)

    return parsed_result
