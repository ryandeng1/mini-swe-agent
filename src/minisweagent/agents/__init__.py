"""Agent implementations for mini-SWE-agent."""


def get_agent_class(class_name: str) -> type:
    if class_name.lower() == "double_checking":
        from minisweagent.agents.extra.double_checking import DoubleCheckingAgent

        return DoubleCheckingAgent
    elif class_name.lower() == "interactive":
        from minisweagent.agents.interactive import InteractiveAgent

        return InteractiveAgent
    from minisweagent.agents.default import DefaultAgent

    return DefaultAgent
