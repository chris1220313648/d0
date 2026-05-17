from pathlib import Path

base = Path("/data/user/wsong890/user68/cjy/Motus/data/robotwin_dataset")

rename_map = {
    "aloha-agilex_clean_50": "clean",
    "radom": "randomized",
}

# 先收集，再重命名（按深度从深到浅，避免遍历冲突）
targets = [p for p in base.rglob("*") if p.is_dir() and p.name in rename_map]
targets.sort(key=lambda p: len(p.parts), reverse=True)

ok, skip = 0, 0
for old_path in targets:
    new_path = old_path.with_name(rename_map[old_path.name])

    if new_path.exists():
        print(f"[SKIP] target exists: {new_path}")
        skip += 1
        continue

    old_path.rename(new_path)
    print(f"[OK] {old_path} -> {new_path}")
    ok += 1

print(f"Done. renamed={ok}, skipped={skip}")
