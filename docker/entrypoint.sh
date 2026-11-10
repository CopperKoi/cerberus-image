#!/bin/sh
set -eu

if [ "$#" -gt 0 ]; then
    exec /opt/cerberus/.venv/bin/python -m cerberus.cli "$@"
fi

if [ -z "${CERBERUS_MODE:-}" ] && {
    [ -n "${BENCHMARK_TOKEN:-}" ] || [ -n "${BENCHMARK_BASE_URL:-}" ];
}; then
    echo "TSecBench requires CERBERUS_MODE=tsecbench-v1 or tsecbench-v2" >&2
    exit 2
fi

case "${CERBERUS_MODE:-ctfy-hosted}" in
    ctfy-hosted)
        config=/opt/cerberus/configs/ctfy-hosted.yaml
        work_dir=${CTFY_WORK_DIR:-/work}
        ;;
    tsecbench-v1)
        config=/opt/cerberus/configs/tsecbench-v1.yaml
        work_dir=${CERBERUS_WORK_DIR:-/work}
        ;;
    tsecbench-v2)
        config=/opt/cerberus/configs/tsecbench-v2.yaml
        work_dir=${CERBERUS_WORK_DIR:-/work}
        ;;
    ctfy-competition)
        config=/opt/cerberus/configs/ctfy-competition.yaml
        work_dir=${CERBERUS_WORK_DIR:-/work}
        ;;
    mock)
        config=/opt/cerberus/configs/mock.yaml
        work_dir=${CERBERUS_WORK_DIR:-/work}
        ;;
    *)
        echo "unsupported CERBERUS_MODE: ${CERBERUS_MODE}" >&2
        exit 2
        ;;
esac

export TMPDIR="$work_dir"
export XDG_CACHE_HOME="$work_dir/.cache"
exec /opt/cerberus/.venv/bin/python -m cerberus.cli run --config "$config" --work-dir "$work_dir"
