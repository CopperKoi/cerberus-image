# syntax=docker/dockerfile:1
FROM python:3.12.12-slim-bookworm@sha256:593bd06efe90efa80dc4eee3948be7c0fde4134606dd40d8dd8dbcade98e669c AS builder
WORKDIR /build
COPY pyproject.toml uv.lock ./
COPY src ./src
COPY docker/tool-requirements.lock ./tool-requirements.lock
RUN python -m pip install --no-cache-dir uv==0.9.14 && \
    uv sync --frozen --no-dev --python /usr/local/bin/python --no-install-project && \
    uv sync --frozen --no-dev --no-editable --python /usr/local/bin/python && \
    uv pip install --python /build/.venv/bin/python --require-hashes -r tool-requirements.lock

FROM python:3.12.12-slim-bookworm@sha256:593bd06efe90efa80dc4eee3948be7c0fde4134606dd40d8dd8dbcade98e669c
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PATH=/opt/cerberus/.venv/bin:$PATH
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    binutils file curl ca-certificates procps netcat-openbsd \
    gdb gcc libc6-dev make nmap p7zip-full libimage-exiftool-perl tshark \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --uid 10001 --create-home --shell /usr/sbin/nologin cerberus \
    && mkdir -p /opt/cerberus /work && chown -R cerberus:cerberus /work
ENV TMPDIR=/work XDG_CACHE_HOME=/work/.cache
WORKDIR /opt/cerberus
COPY --from=builder /build/.venv ./.venv
COPY configs ./configs
COPY docker/capabilities.json docker/tool-requirements.lock ./
COPY docker/check_capabilities.py ./check_capabilities.py
COPY docker/entrypoint.sh ./entrypoint.sh
RUN rm -rf /opt/cerberus/.venv/lib/python3.12/site-packages/cerberus/evaluation \
    /opt/cerberus/.venv/lib/python3.12/site-packages/cerberus/authoring && \
    chmod 0555 /opt/cerberus/entrypoint.sh && \
    dpkg-query -W > /opt/cerberus/system-packages.txt
USER 10001:10001
ENTRYPOINT ["/opt/cerberus/entrypoint.sh"]
