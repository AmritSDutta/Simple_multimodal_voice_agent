from src.flow_agent.config import settings


def configure_arize():
    """
    Configure arize.

    os.environ["ARIZE_SPACE_ID"] = "YOUR_ARIZE_SPACE_ID"
    os.environ["ARIZE_API_KEY"] = "YOUR_ARIZE_API_KEY"

    `register` has set this TracerProvider as the global OpenTelemetry default.
    To disable this behavior, call `register` with `set_global_tracer_provider=False`
    """
    import os
    import logging
    from arize.otel import register
    from openinference.instrumentation.langchain import LangChainInstrumentor

    if not settings.ARIZE_TRACING_ENABLED:
        # application way of disabling Arize if required
        return

    # Setup OTel via Arize AX's convenience function
    tracer_provider = register(
        space_id=os.getenv("ARIZE_SPACE_ID"),
        api_key=os.getenv("ARIZE_API_KEY"),
        project_name=settings.TRACING_PROJECT_NAME,
        set_global_tracer_provider=False,
        log_to_console=False
    )

    # Instrument LangChain (which includes LangGraph)
    LangChainInstrumentor().instrument(tracer_provider=tracer_provider)

    logging.info("LangGraph (via LangChain instrumentor) instrumented for Arize AX.")
