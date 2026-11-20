"""Der LLM-Agent: ein LangGraph-Graph, der auf einem eingefrorenen Snapshot entscheidet.

Die Gemma-Engine kann keine nativen Tool-Aufrufe. Daten sammeln deshalb feste Python-Schritte, das Modell
entscheidet nur, und zwar als JSON nach festem Schema. Für Gemma und Opus ist der Ablauf identisch.
"""

from trading_agent.agent.config import AgentConfig, load_agent_config
from trading_agent.agent.llm_agent import LLMAgent

__all__ = ["AgentConfig", "LLMAgent", "load_agent_config"]
