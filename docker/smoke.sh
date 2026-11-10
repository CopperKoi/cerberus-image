#!/bin/sh
set -eu

image=${1:-cerberus:dev}
base="docker run --rm --platform linux/amd64 --network none --read-only --cap-drop ALL --security-opt no-new-privileges --pids-limit 128 --memory 1g --cpus 1 --tmpfs /work:rw,mode=1777"

$base -e CERBERUS_MODE=mock "$image"
$base "$image" doctor --config /opt/cerberus/configs/ctfy-hosted.yaml
$base --entrypoint /opt/cerberus/.venv/bin/python "$image" /opt/cerberus/check_capabilities.py
docker run --rm --platform linux/amd64 --entrypoint /bin/sh "$image" -c \
    'test ! -d /opt/cerberus/.venv/lib/python3.12/site-packages/cerberus/evaluation && test ! -d /opt/cerberus/.venv/lib/python3.12/site-packages/cerberus/authoring && test ! -e /opt/cerberus/theseus && test ! -e /opt/cerberus/evals'
