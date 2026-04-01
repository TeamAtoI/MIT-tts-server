#!/usr/bin/env sh
set -eu

if [ "${TTS_USE_GPU:-true}" = "true" ]; then
  GPU_LIB_PATHS="$(
    python - <<'PY'
import os
paths = []
for module_name in ("nvidia.cudnn", "nvidia.cublas", "nvidia.cuda_runtime", "nvidia.curand", "nvidia.cufft"):
    try:
        mod = __import__(module_name, fromlist=["*"])
        paths.append(os.path.join(mod.__path__[0], "lib"))
    except Exception:
        continue

# Preserve order while removing duplicates.
seen = set()
ordered = []
for p in paths:
    if p and p not in seen:
        ordered.append(p)
        seen.add(p)
print(":".join(ordered), end="")
PY
  )"

  if [ -n "${GPU_LIB_PATHS}" ]; then
    export LD_LIBRARY_PATH="${GPU_LIB_PATHS}:${LD_LIBRARY_PATH:-}"
  fi
fi

exec uvicorn app.main:app --host 0.0.0.0 --port 8000
