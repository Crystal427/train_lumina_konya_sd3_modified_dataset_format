import difflib
from pathlib import Path
local = Path('library/lumina_train_util.py').read_text(encoding='utf-8').splitlines()
remote = Path('lumina_train_util_remote.py').read_text(encoding='utf-8').splitlines()
for line in difflib.unified_diff(local, remote, fromfile='local', tofile='remote'):
    print(line)
