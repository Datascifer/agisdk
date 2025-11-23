import asyncio

from agi_agents.photo_agent import PhotoWorkflowAgentArgs
from arena import RunHarness


async def main():
    """
    Benchmark runner for the PhotoWorkflowAgent using the Arena harness.

    - Loads the custom agent designed for photographer workflow automation.
    - Runs all tasks located under `src/benchmarks/hackathon/tasks/*`.
    - Executes tasks in parallel.
    - Limits each task to 60 steps.
    - Uses headless mode for performance.
    """

    # Instantiate agent via its Args class (similar to DemoAgentArgs / ManualAgentArgs)
    agent_args = PhotoWorkflowAgentArgs(
        model_name="gpt-4o",        # you may switch to another model here
        use_html=False,
        use_axtree=True,
        use_screenshot=True,
        temperature=0.1,
    )

    agent = agent_args.make_agent()

    harness = RunHarness(
        agent=agent,
        tasks=[
            "src/benchmarks/hackathon/tasks/*"
        ],
        parallel=60,
        sample_count=1,
        max_steps=60,
        headless=True,
    )

    results = await harness.run()
    print(results)


if __name__ == "__main__":
    asyncio.run(main())