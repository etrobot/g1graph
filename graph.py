from pydantic import BaseModel, Field
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
import json
import os
import time
from dotenv import load_dotenv
from langchain_groq import ChatGroq


load_dotenv()


class State(TypedDict):
    message: str
    steps: list
    step_count: int
    total_thinking_time: float
    is_final_answer: bool


class ResponseFormat(BaseModel):
    title: str = Field(..., description="Title of the reasoning step")
    content: str = Field(..., description="Content of the reasoning step")
    next_action: str = Field(..., description="Next action to take after this step")


llm = ChatGroq(
    model=os.getenv("MODEL"),
    api_key=os.getenv("GROQ_API_KEY"),
)

def make_api_call(message, max_tokens, is_final_answer=False):
    messages = [
        {"role": "user", "content": message},
    ]

    if not is_final_answer:
        messages.insert(0,{
            "role": "system",
            "content": """
You are an expert AI assistant that performs step by step deconstructive reasoning.
For each step, provide a title that describes what you're doing in that step, along with the content.
Decide if you need another step or if you're ready to give the final answer.
Respond in schema format with 'title', 'content', and 'next_action' (either 'continue' or 'final_answer') keys.
USE AS MANY REASONING STEPS AS POSSIBLE. AT LEAST 3.
BE AWARE OF YOUR LIMITATIONS AS AN LLM AND WHAT YOU CAN AND CANNOT DO.
IN YOUR REASONING, INCLUDE EXPLORATION OF ALTERNATIVE ANSWERS.
CONSIDER YOU MAY BE WRONG, AND IF YOU ARE WRONG IN YOUR REASONING, WHERE IT WOULD BE.
FULLY TEST ALL OTHER POSSIBILITIES.
YOU CAN BE WRONG.
WHEN YOU SAY YOU ARE RE-EXAMINING, ACTUALLY RE-EXAMINE, AND USE ANOTHER APPROACH TO DO SO.
DO NOT JUST SAY YOU ARE RE-EXAMINING.
USE AT LEAST 3 METHODS TO DERIVE THE ANSWER.
USE BEST PRACTICES.
Example of a valid JSON response:
```json
{
    "title": "Identifying Key Information",
    "content": "To begin solving this problem, we need to carefully examine the given information and identify the crucial elements that will guide our solution process. This involves...",
    "next_action": "continue"
}```
""",
        })

    for attempt in range(3):
        try:
            if is_final_answer:
                response = llm.invoke(
                    input=messages,
                    temperature=0.4,
                )
                return response.content
            else:
                response = llm.with_structured_output(ResponseFormat).invoke(
                    input=messages,
                    max_tokens=max_tokens,
                    temperature=0.8,
                )
                return response.model_dump()
        except Exception as e:
            if attempt == 2:
                if is_final_answer:
                    return {
                        "title": "Error",
                        "content": f"Failed to generate final answer after 3 attempts. Error: {str(e)}",
                    }
                else:
                    return {
                        "title": "Error",
                        "content": f"Failed to generate step after 3 attempts. Error: {str(e)}",
                        "next_action": "final_answer",
                    }
            time.sleep(1)  # Wait for 1 second before retrying


def generate_response_graph():
    graph = StateGraph(State)

    def initialize_state(state: State):
        return {
            "message": state["message"],
            "steps": [],
            "step_count": 1,
            "total_thinking_time": 0,
            "is_final_answer": False,
        }

    def process_step(state: State):
        start_time = time.time()
        step_data = make_api_call(state["message"], 300)
        end_time = time.time()
        thinking_time = end_time - start_time

        if type(step_data) is dict:
            # 检查新步骤是否与最后一个步骤相同
            if not state["steps"] or state["steps"][-1][1] != step_data["content"]:
                new_step = (
                    f"Step {state['step_count']}: {step_data['title']}",
                    step_data["content"],
                    thinking_time,
                )
                state["steps"] = state["steps"] + [new_step]
                message = state["message"] + json.dumps(new_step)

        return {
            "message": message,
            "steps": state["steps"],
            "step_count": state["step_count"] + 1,
            "total_thinking_time": state["total_thinking_time"] + thinking_time,
            "is_final_answer": step_data["next_action"] == "final_answer"
            or state["step_count"] >= os.getenv("MAX_STEPS", 10),
        }

    def generate_final_answer(state: State):
        message = (
            state["message"]
            + "Please provide the final answer based solely on your reasoning above. Do not use JSON formatting. Only provide the text response without any titles or preambles."
        )

        start_time = time.time()
        final_data = make_api_call(message, 1200, is_final_answer=True)
        end_time = time.time()
        thinking_time = end_time - start_time

        final_step = ("Final Answer", final_data, thinking_time)

        return {
            "steps": state["steps"] + [final_step],
            "total_thinking_time": state["total_thinking_time"] + thinking_time,
        }

    def should_continue(state: State):
        return (
            "process_step" if not state["is_final_answer"] else "generate_final_answer"
        )

    graph.add_node("initialize", initialize_state)
    graph.add_node("process_step", process_step)
    graph.add_node("generate_final_answer", generate_final_answer)

    graph.add_edge(START, "initialize")
    graph.add_edge("initialize", "process_step")
    graph.add_conditional_edges("process_step", should_continue)
    graph.add_edge("generate_final_answer", END)

    return graph.compile()

if __name__ == "__main__":
    app = generate_response_graph()
    print(app.get_graph().draw_mermaid())
