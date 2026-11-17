"""Print the variant-pick block of the head workflow's Native arm64 gate as a standalone PowerShell script."""
import sys
lines = open(sys.argv[1], encoding="utf-8").read().splitlines()
start = next(i for i, l in enumerate(lines) if l.strip().startswith("# Variant pick:"))
end = next(i for i in range(start, len(lines)) if "without i8mm the pick is" in lines[i])
print("$ErrorActionPreference = 'Continue'")
print("\n".join(l[10:] for l in lines[start:end + 1]))
print("Write-Host 'variant gate: PASS'")
