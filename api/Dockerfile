# Sandbox image for the "docker" executor.
# Runs `python -m api.cli compile ...` with no network, read-only FS.

FROM python:3.14-slim

WORKDIR /app

COPY ast_nodes.py lexer.py parser.py ./
COPY backends/ ./backends/
COPY api/ ./api/

# No pip installs needed: only stdlib is used for the python backend.
# For the native backend, extend this image with the Rust toolchain.

CMD ["python", "-m", "api.cli", "compile", "--backend", "python", "--stdin"]
