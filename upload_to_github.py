#!/usr/bin/env python3

import json
import subprocess
import sys
from pathlib import Path

storage_dir = Path(".github_issues")

with open(storage_dir / "labels.json") as f:
    labels = json.load(f)

with open(storage_dir / "milestones.json") as f:
    milestones = json.load(f)

with open(storage_dir / "issues.json") as f:
    issues = json.load(f)

print("Creating labels...")
for label, color in labels.items():
    cmd = ["gh", "label", "create", label, "--color", color, "--force"]
    subprocess.run(cmd, capture_output=True)

print("Creating milestones...")
for ms in milestones:
    cmd = ["gh", "milestone", "create", ms["title"], "--description", ms["description"]]
    subprocess.run(cmd, capture_output=True)

print("Creating issues...")
for issue in issues:
    cmd = [
        "gh", "issue", "create",
        "--title", issue["title"],
        "--body", issue["body"]
    ]
    for label in issue["labels"]:
        cmd.extend(["--label", label])
    
    phase = issue["phase"]
    cmd.extend(["--milestone", phase])
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode == 0:
        print(f"  [OK] {issue['title']}")
    else:
        print(f"  [SKIP] {issue['title']}")

print(f"\nTotal issues configured: {len(issues)}")
print("Run this script when you have internet connection:")
print("  python upload_to_github.py")
