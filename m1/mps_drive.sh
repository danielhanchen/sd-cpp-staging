#!/bin/bash
# interleaved arms, 3 rounds, rotated order; fresh process per run
set -u
PY=$1; DD=$2; OUT=$3; shift 3
mkdir -p "$OUT"
PROBE=$(env -u PYTORCH_MPS_LOW_WATERMARK_RATIO $PY "$DD/../mps_wm.py" --dd "$DD" --arm probe --out /dev/null --probe | tail -1)
echo "probe $PROBE" | tee "$OUT/probe.txt"
CAP=$(python3 -c "import json,sys;print(json.loads(sys.argv[1])['ratio'])" "$PROBE")
low() { case $1 in unset) echo "";; cap) echo "$CAP";; half) echo "0.5";; esac; }
ORDERS=("unset cap half" "cap half unset" "half unset cap")
for r in 0 1 2; do for arm in ${ORDERS[$r]}; do
  L=$(low $arm)
  echo "=== round $r arm $arm low=$L"
  if [ -z "$L" ]; then env -u PYTORCH_MPS_LOW_WATERMARK_RATIO $PY -u "$DD/../mps_wm.py" --dd "$DD" --arm $arm --out "$OUT/${arm}_r$r.json" "$@" 2>&1 | tail -3
  else PYTORCH_MPS_LOW_WATERMARK_RATIO=$L $PY -u "$DD/../mps_wm.py" --dd "$DD" --arm $arm --out "$OUT/${arm}_r$r.json" "$@" 2>&1 | tail -3; fi
done; done
