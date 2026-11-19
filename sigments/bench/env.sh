# source this from the repository root or from bench/: finds the interpreter and prepares its DLL path.
# On Windows a conda interpreter imports numpy only when the environment's Library/bin is on PATH;
# without it the import ends silently with exit code 127.
# Set SIGMENTS_ENV to the environment folder to override the search.
for E in "${SIGMENTS_ENV:-}" "$HOME/.conda/envs/ml" /d/miniforge3/envs/ml /e/miniforge3/envs/ml; do
  if [ -n "$E" ] && [ -x "$E/python.exe" ]; then break; fi
done
export PATH="$E:$E/Library/bin:$PATH"
export GDAL_DATA="$E/Library/share/gdal" PROJ_DATA="$E/Library/share/proj" PYTHONIOENCODING=utf-8
PY="$E/python.exe"
