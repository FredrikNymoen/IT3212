"""Verify word counts, LaTeX cross-references, figures, and untouched source hashes."""
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "notebooks/assignment4"))
from q_learning import verify_sources


def strip_tables_and_captions(tex: str) -> str:
    tex = re.sub(r"(?<!\\)%.*", "", tex)
    tex = re.sub(r"\\begin\{table\}.*?\\end\{table\}", " ", tex, flags=re.S)
    tex = re.sub(r"\\begin\{figure\}.*?\\end\{figure\}", " ", tex, flags=re.S)

    def remove_cmd(s, cmd):
        out = []
        i = 0
        while i < len(s):
            m = re.search(r"\\" + cmd + r"(?:\{[^}]*\})?\{", s[i:])
            if not m:
                out.append(s[i:])
                break
            out.append(s[i : i + m.start()])
            j = i + m.end()
            depth = 1
            while j < len(s) and depth > 0:
                if s[j] == "{":
                    depth += 1
                elif s[j] == "}":
                    depth -= 1
                j += 1
            i = j
        return "".join(out)

    tex = remove_cmd(tex, "captionof")
    tex = remove_cmd(tex, "caption")
    tex = re.sub(
        r"\\(?:includegraphics|label|ref|setcounter|usepackage|documentclass|input|clearpage|"
        r"maketitle|centering|noindent|par|smallskip|medskip|bigskip|quad|addlinespace)"
        r"(?:\[[^\]]*\])?(?:\{[^}]*\})*",
        " ",
        tex,
    )
    tex = re.sub(r"\\(?:begin|end)\{[^}]*\}(?:\{[^}]*\})?", " ", tex)
    tex = re.sub(r"\\[a-zA-Z]+", " ", tex)
    tex = re.sub(r"[{}$~\\]", " ", tex)
    return tex


def main():
    sec_dir = ROOT / "reports/assignment4/Sections"
    files = [
        "01_predictions.tex",
        "01_prediction_annotations.tex",
        "02_learning_settings.tex",
        "03_mechanisms.tex",
        "04_episode_inspection.tex",
        "05_human_review.tex",
        "06_recommendations.tex",
    ]
    counts = {}
    for name in files:
        txt = (sec_dir / name).read_text(encoding="utf-8")
        w = len(re.findall(r"\b[\w'-]+\b", strip_tables_and_captions(txt)))
        counts[name] = w
        print(f"{name:32s}: {w:4d} prose words")

    brief = sum(counts[k] for k in files[2:5] + [files[6]])
    review = counts["05_human_review.tex"]
    print(f"\nEVIDENCE BRIEF (Parts 2,3,4,6): {brief} / 2000 words -> {'PASS' if brief <= 2000 else 'OVER LIMIT'}")
    print(f"REVIEW RECORD  (Part 5)       : {review} / 1000 words -> {'PASS' if review <= 1000 else 'OVER LIMIT'}")

    all_tex = "\n".join(p.read_text(encoding="utf-8") for p in (ROOT / "reports/assignment4").rglob("*.tex"))
    labels = set(re.findall(r"\\label\{([^}]+)\}", all_tex))
    refs = set(re.findall(r"\\ref\{([^}]+)\}", all_tex))
    missing_refs = sorted(refs - labels)
    imgs = re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", all_tex)
    missing_imgs = [im for im in imgs if not (ROOT / "reports/assignment4" / im).is_file()]
    print(f"Labels: {len(labels)}, Refs: {len(refs)}, Missing refs: {missing_refs}")
    print(f"Images referenced: {len(imgs)}, Missing images: {missing_imgs}")

    sel_path = ROOT / "results/assignment4/part3/study/e8c0c6a8f4071325/PPO_3b1630567f4b85f5/seed_17/selected.json"
    verify_sources(json.loads(sel_path.read_text(encoding="utf-8"))["source_hashes"])
    print("Source SHA-256 verification   : PASS")

    assert brief <= 2000, f"Evidence brief exceeds 2,000 words ({brief})"
    assert review <= 1000, f"Review record exceeds 1,000 words ({review})"
    assert not missing_refs, f"Missing LaTeX refs: {missing_refs}"
    assert not missing_imgs, f"Missing images: {missing_imgs}"
    print("\nALL CHECKS PASSED!")


if __name__ == "__main__":
    main()
