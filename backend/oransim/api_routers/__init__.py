"""FastAPI routers extracted from the original ``oransim.api`` god-file.

Each submodule defines an ``APIRouter`` that is mounted by
``oransim.api`` at startup. Routers pull shared runtime state from
``oransim.api_state`` at call time (not import time) so they see the
populated singletons after the lifespan bootstrap.
"""
