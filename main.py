import os
import signal
from typing import Annotated

import uvicorn
from fastapi.encoders import jsonable_encoder

from performance_review import evaluate_performance, Meeting, Sprint, Task
from read_json import read_json_file
from fastapi import Body, FastAPI

app = FastAPI()


@app.post("/")
async def performance(
    sprint_data: Annotated[Sprint, Body()],
    meeting_data: Annotated[Meeting, Body()]
):
    task_data = read_json_file('task.json')
    meeting_data = read_json_file('meeting.json')

    tasks = Sprint.model_validate(task_data)
    meeting = Meeting.model_validate(meeting_data)

    performance_summary = evaluate_performance(tasks, meeting)
    return jsonable_encoder(performance_summary)


if __name__ == "__main__":
    try:
        uvicorn.run(app="main:app", host="localhost", port=8000, reload=True)
    except KeyboardInterrupt:
        pass
    finally:
        os.kill(os.getpid(), signal.SIGINT)
