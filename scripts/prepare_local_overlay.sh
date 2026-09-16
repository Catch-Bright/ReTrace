#!/usr/bin/env bash
# Prepare the existing /root overlay and point its editable install at this repo.
set -euo pipefail
repo_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)
personal_root=$(cd -- "$repo_root/../.." && pwd -P)
case "${1:-}" in
  webshop)
    archive="$personal_root/envs/webshop-local-overlay-py31020.tar.gz"
    env_dir=/root/envs/webshop-uv-py31020
    ;;
  alfworld)
    archive="$personal_root/envs/alfworld-local-overlay-py312.tar.gz"
    env_dir=/root/envs/alfworld-uv-py312
    ;;
  *) echo "Usage: bash $0 {webshop|alfworld}" >&2; exit 2 ;;
esac
if [[ ! -x "$env_dir/bin/python" ]] || \
   [[ "${1}" == webshop && ! -x /root/software/jdk11/bin/java ]] || \
   [[ "${1}" == alfworld && ! -d /root/.cache/alfworld ]]; then
  tar -xzf "$archive" -C /
fi
"$env_dir/bin/python" - "$env_dir" "$personal_root" "$repo_root" <<'PYCODE'
import pathlib, sys
venv, personal, repo = map(pathlib.Path, sys.argv[1:])
old = '/var/gfs/wangyuhao'
for site in venv.glob('lib/python*/site-packages'):
    files = list(site.glob('*.pth')) + list(site.glob('*.egg-link')) + list(site.glob('__editable__*.py'))
    for path in files:
        text = path.read_text()
        if old in text:
            path.write_text(text.replace(old, str(personal)))
            print('Updated', path)
    # Make this checkout importable even when the old overlay had no editable install.
    (site / 'verl_migrated_checkout.pth').write_text(str(repo) + '\n')
PYCODE
cd -- "$repo_root"
"$env_dir/bin/python" - <<'PYCODE'
import importlib.util, pathlib
repo = pathlib.Path.cwd()
for name in ('verl', 'gigpo', 'agent_system'):
    spec = importlib.util.find_spec(name)
    if spec is None:
        raise SystemExit(f'Missing module: {name}')
    locations = list(spec.submodule_search_locations or [])
    if spec.origin and spec.origin not in ('built-in', 'frozen'):
        locations.append(spec.origin)
    if not any(pathlib.Path(p).resolve().is_relative_to(repo) for p in locations):
        raise SystemExit(f'{name} resolves outside this checkout: {locations}')
    print(name, locations)
PYCODE
printf 'Ready. Activate with: source %s/bin/activate\n' "$env_dir"
