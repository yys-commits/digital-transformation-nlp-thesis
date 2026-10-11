# -*- coding: utf-8 -*-
"""
一键复现：按顺序执行 01 -> 08，产出全部表格与插图。
    python3 code/run_all.py
如需使用真实年报数据，只需替换 01 步（把年报 MD&A 文本写成
data/raw/mda_corpus.jsonl，财务面板写成 data/raw/panel_raw.csv），
然后从 02 开始执行即可：python3 code/run_all.py --from 02
"""
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
STEPS = [
    ("01_simulate_corpus.py", "语料与面板构造"),
    ("02_preprocess.py", "清洗与分词"),
    ("03_train_word2vec.py", "词向量训练"),
    ("04_expand_lexicon.py", "种子词扩展与词典定稿"),
    ("05_construct_measure.py", "测度构造"),
    ("06_validity.py", "效度检验"),
    ("07_regression.py", "实证回归"),
    ("08_figures.py", "结果可视化"),
    ("09_robustness_measure.py", "测度参数敏感性"),
]


def main():
    start = int(sys.argv[sys.argv.index("--from") + 1]) if "--from" in sys.argv else 1
    for fn, desc in STEPS:
        if int(fn[:2]) < start:
            continue
        t0 = time.time()
        print(f"\n===== [{fn}] {desc} =====")
        r = subprocess.run([sys.executable, str(HERE / fn)],
                           capture_output=True, text=True, cwd=HERE)
        print(r.stdout[-4000:] if r.stdout else "")
        if r.returncode != 0:
            print(r.stderr[-3000:])
            print(f"!! {fn} 执行失败，已中止")
            sys.exit(1)
        print(f"----- 用时 {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
