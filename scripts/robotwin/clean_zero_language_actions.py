"""Remove zero-motion clauses from existing RobotWin labels, with backups."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import shutil
import tempfile


ZERO = re.compile(
    r"(?:move (?:forward|back|backward|up|down|left|right) [+-]?0(?:\.0+)? cm"
    r"|(?:tilt (?:left|right|back|forward)|rotate (?:counterclockwise|clockwise))"
    r" [+-]?0(?:\.0+)? degrees?)"
)


def clean(text):
    removed = Counter()
    output = []
    for line in text.splitlines(keepends=True):
        if not ZERO.search(line):
            output.append(line)
            continue
        match = re.fullmatch(r"Left arm: (.*?)\. Right arm: (.*?)(\r?\n)?", line)
        if not match:
            raise ValueError(f"Unexpected label format: {line!r}")
        arms = []
        for arm in match.group(1, 2):
            kept = []
            for clause in arm.split(", "):
                if ZERO.fullmatch(clause):
                    removed[clause] += 1
                else:
                    kept.append(clause)
            if not kept or not any(kept):
                raise ValueError(f"Cleaning would leave an empty arm: {line!r}")
            arms.append(", ".join(kept))
        output.append(f"Left arm: {arms[0]}. Right arm: {arms[1]}{match.group(3) or ''}")
    return "".join(output), removed


def self_check():
    before = "Left arm: move down 0 cm, move left 10 cm, open gripper. Right arm: tilt left 0 degrees, close gripper\n"
    after, removed = clean(before)
    assert after == "Left arm: move left 10 cm, open gripper. Right arm: close gripper\n"
    assert sum(removed.values()) == 2
    assert clean(after) == (after, Counter())
    decimals = "Left arm: move up 0.1 cm, move down 0.0 cm, open gripper. Right arm: close gripper"
    assert clean(decimals)[0] == decimals.replace("move down 0.0 cm, ", "")
    assert clean(before.replace("\n", "\r\n"))[0] == after.replace("\n", "\r\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    self_check()
    root = args.root.resolve()
    backup = root.parent / ("zero_motion_backup_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ"))
    if args.apply:
        backup.mkdir()
        shutil.copy2(__file__, backup / "clean_zero_language_actions.py")
        print(f"Backup: {backup}", flush=True)
    counts = Counter()
    clauses = Counter()
    for i, path in enumerate(sorted(root.glob("*/*/language_action/*.txt")), 1):
        raw = path.read_bytes()
        text = raw.decode("utf-8")
        cleaned, removed = clean(text)
        counts["scanned_files"] += 1
        if removed:
            assert len(text.splitlines()) == len(cleaned.splitlines())
            assert clean(cleaned)[0] == cleaned
            relative = path.relative_to(root)
            counts["changed_files"] += 1
            counts[f"{relative.parts[0]}_changed_files"] += 1
            counts["changed_lines"] += sum(a != b for a, b in zip(text.splitlines(), cleaned.splitlines()))
            clauses.update(removed)
            if args.apply:
                if path.is_symlink():
                    raise ValueError(f"Refusing to replace symlink: {path}")
                saved = backup / relative
                saved.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, saved)
                assert saved.read_bytes() == raw
                payload = cleaned.encode("utf-8")
                with tempfile.NamedTemporaryFile(dir=path.parent, prefix=".zero-clean-", delete=False) as tmp:
                    tmp.write(payload)
                    tmp.flush()
                    os.fsync(tmp.fileno())
                shutil.copymode(path, tmp.name)
                assert path.read_bytes() == raw, f"File changed concurrently: {path}"
                os.replace(tmp.name, path)
                assert path.read_bytes() == payload
        if i % 2000 == 0:
            print(json.dumps(dict(counts)), flush=True)
    report = {"root": str(root), "applied": args.apply, **counts,
              "removed_clauses": sum(clauses.values()), "clauses": dict(clauses)}
    if args.apply:
        (backup / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
