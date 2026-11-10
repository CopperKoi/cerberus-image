# cerberus image

Public build source for the cerberus CTF agent container. This is the packaged runtime source; development plans, reference archives, benchmark material, credentials and evaluation traces are excluded.

The image uses the CTFy hosted contract by default. For TSecBench hosted runs, set `CERBERUS_MODE=tsecbench-v1` or `tsecbench-v2` and configure the DeepSeek `LLM_*` variables in the platform. The platform injects `BENCHMARK_*` variables.

Current image: `ghcr.io/copperkoi/cerberus-image:instance-continuity-20261111-r9`. Its benchmark score is not yet known. The TSecBench v1 run request is in the private development workspace; submit only the image tar or this image reference as supported by the platform.

Sapper v2 by N1nEmAn (MIT license) informed interface and scheduling choices. This repository does not include Sapper source files, benchmark answers, task solutions or success traces.
