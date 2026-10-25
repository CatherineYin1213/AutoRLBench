"""Cross-grade one fixed set of MATH answers with every grader, so grading changes are separated from generation changes.

Graders on the same answers:
  buggy_thread   pre-fix verl math_verify.compute_score via loop.run_in_executor (the reward loop's real call pattern)
  buggy_main     the same pre-fix function called from the main thread (its logic, without the thread)
  fixed_thread   verl @6f2aa24 math_verify.compute_score via loop.run_in_executor
  indep_mv       math-verify parse()+verify() called directly (reference)
  indep_verl     verl math_reward (last \\boxed{} + is_equiv), a second, unrelated implementation
Each grader is reported as a confusion matrix against indep_mv, so a grader that returns 1 (or 0) for everything fails.

Optionally, `--real <control dump> <defect dump>`: answers that both arms generated identically (greedy step 0) carry the
score each arm's *real* reward path logged; those are compared on the same answers too.
Usage: python crossgrade.py <val dump .jsonl> [--real <control .jsonl> <defect .jsonl>] [--out per_answer.jsonl]
"""
import asyncio
import importlib.util
import json
import sys

sys.path.insert(0, "/workspace/verl")  # fixed module must be importable by name: it spawns a process pool
from math_verify import parse, verify  # noqa: E402
from math_verify.parser import ExprExtractionConfig, LatexExtractionConfig  # noqa: E402
from verl.utils.reward_score import math_reward  # noqa: E402
from verl.utils.reward_score import math_verify as fixed_mv  # noqa: E402

spec = importlib.util.spec_from_file_location("buggy_math_verify", "/workspace/verl/verl/utils/reward_score/math_verify.py")
buggy_mv = importlib.util.module_from_spec(spec)
spec.loader.exec_module(buggy_mv)

GOLD, PRED = (LatexExtractionConfig(),), (ExprExtractionConfig(), LatexExtractionConfig())


def indep_mv(out, gt):
    g, p = parse("\\boxed{" + gt + "}", GOLD), parse(out, PRED)
    return float(bool(g and p and verify(g, p)))


def load_math(path):
    rows = [json.loads(line) for line in open(path)]
    return [r for r in rows if '"####"' not in r["input"]]


async def threaded(fn, rows):
    loop = asyncio.get_running_loop()
    return [float(await loop.run_in_executor(None, lambda r=r: fn(r["output"], str(r["gts"])))) for r in rows]


def confusion(name, pred, ref):
    tp = sum(p == 1 and r == 1 for p, r in zip(pred, ref))
    tn = sum(p == 0 and r == 0 for p, r in zip(pred, ref))
    fp = sum(p == 1 and r == 0 for p, r in zip(pred, ref))
    fn = sum(p == 0 and r == 1 for p, r in zip(pred, ref))
    agree = (tp + tn) / len(ref)
    print(f"  {name:<14} mean={sum(pred) / len(pred):.4f}  TP={tp:4d} TN={tn:4d} FP={fp:4d} FN={fn:4d}  agree={agree:.4f}")


def main():
    args = sys.argv[1:]
    out_path = args[args.index("--out") + 1] if "--out" in args else None
    rows = load_math(args[0])
    ref = [indep_mv(r["output"], str(r["gts"])) for r in rows]
    grades = {
        "buggy_thread": asyncio.run(threaded(buggy_mv.compute_score, rows)),
        "buggy_main": [float(buggy_mv.compute_score(r["output"], str(r["gts"]))) for r in rows],
        "fixed_thread": asyncio.run(threaded(fixed_mv.compute_score, rows)),
        "indep_verl": [float(math_reward.compute_score(r["output"], str(r["gts"]))) for r in rows],
    }
    print(f"{args[0]}: {len(rows)} MATH answers; reference indep_mv: {int(sum(ref))} correct, {len(ref) - int(sum(ref))} wrong")
    for name, pred in grades.items():
        confusion(name, pred, ref)
    if out_path:
        with open(out_path, "w") as f:
            for i, r in enumerate(rows):
                f.write(json.dumps({"i": i, "gts": r["gts"], "logged": r["score"], "indep_mv": ref[i],
                                    **{k: v[i] for k, v in grades.items()}}) + "\n")
    if "--real" in args:
        c_path, d_path = args[args.index("--real") + 1: args.index("--real") + 3]
        ctrl, dfct = load_math(c_path), {r["input"]: r for r in load_math(d_path)}
        same = [(c, dfct[c["input"]]) for c in ctrl if c["input"] in dfct and dfct[c["input"]]["output"] == c["output"]]
        print(f"real reward paths on identical answers: {len(same)} / {len(ctrl)} step-0 answers identical in both arms")
        ref_same = [indep_mv(c["output"], str(c["gts"])) for c, _ in same]
        print(f"  reference on that subset: {int(sum(ref_same))} correct, {len(same) - int(sum(ref_same))} wrong")
        confusion("control_logged", [float(c["score"]) for c, _ in same], ref_same)
        confusion("defect_logged", [float(d["score"]) for _, d in same], ref_same)


if __name__ == "__main__":
    main()
