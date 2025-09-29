from pathlib import Path
text = Path("library/lumina_train_util.py").read_text(encoding="utf-8").splitlines()
for idx in range(820, 880):
    print(f"{idx+1:04d}: {text[idx]}")
