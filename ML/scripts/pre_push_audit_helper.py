"""
Pre-push security, hygiene, and integrity audit helper.
"""

import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

def scan_working_tree_secrets():
    print("=== [1] Working Tree Secret & Credential Scan ===")
    secret_patterns = [
        ("Generic Secret/Token/Key Assignment", re.compile(r'(?i)(api[_-]?key|secret[_-]?key|password|auth[_-]?token|bearer[_-]?token|private[_-]?key|access[_-]?key|aws_secret)\s*[:=]\s*[\'"][A-Za-z0-9_\-\.]{10,}[\'"]')),
        ("Private Key Header", re.compile(r'-----BEGIN\s+(RSA|EC|DSA|OPENSSH|PRIVATE)\s+KEY-----')),
        ("GitHub Personal Access Token", re.compile(r'gh[pousr]_[A-Za-z0-9_]{36,}')),
        ("AWS Access Key ID", re.compile(r'(?:A3T[A-Z0-9]|AKIA|AGPA|AIDA|AROA|AIPA|ANPA|ANVA|ASIA)[A-Z0-9]{16}')),
        ("Google API Key", re.compile(r'AIza[0-9A-Za-z\\-_]{35}')),
        ("Slack Webhook / Token", re.compile(r'xox[baprs]-[0-9]{10,}-[0-9]{10,}-[a-zA-Z0-9]{24,}')),
        ("Generic JWT", re.compile(r'eyJ[A-Za-z0-9-_=]+\.eyJ[A-Za-z0-9-_=]+\.?[A-Za-z0-9-_.+/=]*')),
    ]
    
    ignore_dirs = {'.git', '.venv', 'node_modules', '__pycache__', '.pytest_cache'}
    
    findings = []
    suspicious_files = []
    for p in PROJECT_ROOT.rglob('*'):
        if p.is_file() and not any(part in ignore_dirs for part in p.parts):
            # Check for suspicious file names
            if p.name.startswith('.env') and p.name != '.env.example':
                suspicious_files.append(str(p.relative_to(PROJECT_ROOT)))
            if p.suffix in ['.pem', '.key', '.p12', '.pfx', '.pkcs12']:
                suspicious_files.append(str(p.relative_to(PROJECT_ROOT)))
                
            try:
                content = p.read_text(encoding='utf-8', errors='ignore')
                for i, line in enumerate(content.splitlines(), 1):
                    # Skip markdown code block quotes, test synthetic data, and hash entries
                    if "hash" in line.lower() or "sha256" in line.lower() or "synthetic" in line.lower():
                        continue
                    if len(line) > 500: # skip minified files
                        continue
                    for pat_name, pat in secret_patterns:
                        if pat.search(line):
                            findings.append({
                                "file": str(p.relative_to(PROJECT_ROOT)),
                                "line": i,
                                "pattern": pat_name,
                                "snippet": line.strip()[:60]
                            })
            except Exception:
                pass
                
    print(f"Suspicious files found: {len(suspicious_files)}")
    for sf in suspicious_files:
        print(f"  [!] {sf}")
    print(f"Potential secret pattern matches: {len(findings)}")
    for f in findings:
        print(f"  [!] {f['file']}:{f['line']} ({f['pattern']}) -> {f['snippet']}")
    return len(findings) == 0 and len(suspicious_files) == 0


def scan_git_history_secrets():
    print("\n=== [2] Git History Secret & Key Scan ===")
    try:
        # Search git log for secret-related words or deleted .env files
        res = subprocess.run(
            ["git", "log", "--all", "-p", "--", "*.env*", "*.pem", "*.key"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            errors="ignore"
        )
        diff_output = res.stdout
        print(f"Git history .env/*.pem search output lines: {len(diff_output.splitlines())}")
        
        # Check commits for keyword additions
        keywords = ["AKIA", "AIza", "BEGIN PRIVATE KEY", "ghp_"]
        history_findings = []
        for kw in keywords:
            res_kw = subprocess.run(
                ["git", "log", "-S", kw, "--oneline"],
                cwd=PROJECT_ROOT,
                capture_output=True,
                text=True
            )
            if res_kw.stdout.strip():
                history_findings.append((kw, res_kw.stdout.strip()))
                
        print(f"Historical credential additions found: {len(history_findings)}")
        for kw, commits in history_findings:
            print(f"  [!] Found '{kw}' in commits:\n{commits}")
        return len(history_findings) == 0
    except Exception as e:
        print(f"Git history search failed: {e}")
        return False


def scan_large_files():
    print("\n=== [3] Large File Audit (>500KB) ===")
    ignore_dirs = {'.git', '.venv', 'node_modules', '__pycache__', '.pytest_cache'}
    large_files = []
    for p in PROJECT_ROOT.rglob('*'):
        if p.is_file() and not any(part in ignore_dirs for part in p.parts):
            size_mb = p.stat().st_size / (1024 * 1024)
            if size_mb > 0.5:
                large_files.append((str(p.relative_to(PROJECT_ROOT)), size_mb))
                
    large_files.sort(key=lambda x: x[1], reverse=True)
    for path, size in large_files:
        print(f"  {size:6.2f} MB  {path}")
    return large_files


def scan_absolute_paths():
    print("\n=== [4] Absolute Path Audit ===")
    ignore_dirs = {'.git', '.venv', 'node_modules', '__pycache__', '.pytest_cache', 'figures'}
    abs_path_patterns = [
        re.compile(r'[a-zA-Z]:\\[Uu]sers\\[A-Za-z0-9_.\-]+'),
        re.compile(r'/home/[a-zA-Z0-9_.\-]+'),
        re.compile(r'/Users/[a-zA-Z0-9_.\-]+')
    ]
    
    findings = []
    for p in PROJECT_ROOT.rglob('*'):
        if p.is_file() and not any(part in ignore_dirs for part in p.parts):
            if p.suffix in ['.py', '.yaml', '.yml', '.json', '.toml']:
                try:
                    content = p.read_text(encoding='utf-8', errors='ignore')
                    for i, line in enumerate(content.splitlines(), 1):
                        for pat in abs_path_patterns:
                            if pat.search(line):
                                findings.append((str(p.relative_to(PROJECT_ROOT)), i, line.strip()[:70]))
                except Exception:
                    pass
    print(f"Hardcoded user absolute paths in source/configs: {len(findings)}")
    for f in findings[:20]:
        print(f"  [!] {f[0]}:{f[1]} -> {f[2]}")
    return findings


def verify_reproducibility_hashes():
    print("\n=== [5] Manifest & Synthetic Baseline Verification ===")
    synth_p = PROJECT_ROOT / "ML" / "reports" / "final_model_comparison.csv"
    h = hashlib.sha256()
    with open(synth_p, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    synth_hash = h.hexdigest()
    expected_synth = "54cb659a7d8e86f0014114e616d2ecbf834d574b877b8db69326e6440c8223fd"
    print(f"Synthetic baseline hash: {synth_hash}")
    assert synth_hash == expected_synth, "SYNTHETIC BASELINE HASH MISMATCH!"
    print("  -> Synthetic Baseline Hash: PASS (UNCHANGED)")
    
    manifest_p = PROJECT_ROOT / "ML" / "reports" / "final_research_manifest.json"
    with open(manifest_p, "r", encoding="utf-8") as f:
        manifest = json.load(f)
        
    mismatches = []
    for fname, exp_h in manifest["report_hashes"].items():
        fp = PROJECT_ROOT / "ML" / "reports" / fname
        if not fp.exists():
            mismatches.append((fname, "FILE_NOT_FOUND"))
            continue
        h = hashlib.sha256()
        with open(fp, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                h.update(chunk)
        act_h = h.hexdigest()
        if act_h != exp_h:
            mismatches.append((fname, f"MISMATCH: exp {exp_h[:8]} vs act {act_h[:8]}"))
            
    print(f"Manifest report hash checks: {len(manifest['report_hashes'])} files, {len(mismatches)} mismatches")
    for m in mismatches:
        print(f"  [!] {m[0]}: {m[1]}")
    return len(mismatches) == 0


if __name__ == "__main__":
    scan_working_tree_secrets()
    scan_git_history_secrets()
    scan_large_files()
    scan_absolute_paths()
    verify_reproducibility_hashes()
