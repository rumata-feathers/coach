"""Provider-agnostic web search adapter.

All web search calls go through :class:`WebSearchClient`. Concrete
implementations: :class:`~career_coach.web.tavily.TavilyClient` (default).
Use :func:`~career_coach.web.factory.get_client` to obtain a configured
client for a given agent.

See SPEC_v1.md §4 for the design contract.
"""
