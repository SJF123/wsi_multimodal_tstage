
from pathlib import Path as _ToolPath
import sys as _tool_sys

_tool_sys.path.insert(
    0, str(_ToolPath(__file__).resolve().parents[1])
)

"""Train TF-IDF + LogisticRegression only on training texts."""

import argparse
import json
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from src.data import STAGES, clinical_text, CLINICAL_COLUMNS


def main():
    # 一、读取参数
    a = argparse.ArgumentParser()
    a.add_argument('--manifest', required=True)
    a.add_argument('--output', default='outputs/tfidf')
    a.add_argument(
        '--seeds',
        nargs='+',
        type=int,
        default=[11, 22, 33, 44, 55]
    )
    args = a.parse_args()

    # 二、读取临床数据
    df = pd.read_csv(args.manifest)

    missing = set(CLINICAL_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(
            f'Missing approved clinical fields: {sorted(missing)}'
        )

    # 严格使用论文规定的10项临床变量
    df['clinical_text'] = df.apply(clinical_text, axis=1)

    df['y_true'] = (
        df.stage.astype(str).str.upper().map(STAGES)
    )

    if df.y_true.isna().any():
        raise ValueError('Expected stage T1-T4')

    root = Path(args.output)

    print(
        'Using SAGA solver with five random seeds. '
        'Results may still be identical after convergence.'
    )

    # 三、使用五个随机种子进行实验
    for seed in args.seeds:

        tr = df[df.split == 'train']
        te = df[df.split == 'test']
        va = df[df.split == 'val']

        # TF-IDF：只用训练集拟合词汇表
        vect = TfidfVectorizer(
            ngram_range=(1, 2),
            min_df=1
        )

        xtr = vect.fit_transform(
            tr.clinical_text.fillna('unknown')
        )

        # 四、训练 Logistic Regression
        # ===== 本次修改的核心位置 =====
        model = LogisticRegression(
            solver='saga',
            max_iter=5000,
            tol=1e-4,
            random_state=seed,
            class_weight='balanced',
            n_jobs=1
        )

        model.fit(xtr, tr.y_true)

        # 五、建立输出文件夹
        folder = root / str(seed)
        folder.mkdir(
            parents=True,
            exist_ok=True
        )

        # 保存实际实验参数
        config = {
            'seed': seed,
            'solver': 'saga',
            'max_iter': 5000,
            'tol': 1e-4,
            'vectorizer_fit': 'train_only',
            'fixed_patient_split': True
        }

        (folder / 'run_config.json').write_text(
            json.dumps(config, indent=2),
            encoding='utf-8'
        )

        # 六、在验证集和测试集上预测
        for split, subset in [
            ('test', te),
            ('val', va)
        ]:

            x = vect.transform(
                subset.clinical_text.fillna('unknown')
            )

            probs = model.predict_proba(x)

            predictions = model.classes_[
                probs.argmax(axis=1)
            ]

            out = pd.DataFrame({
                'patient_id': subset.patient_id,
                'y_true': subset.y_true.astype(int),
                'y_pred': predictions
            })

            # 输出四个T分期对应的预测概率
            for k in range(4):
                out[f'prob_T{k+1}'] = probs[
                    :, list(model.classes_).index(k)
                ]

            out.to_csv(
                folder / f'{split}_predictions.csv',
                index=False
            )

        print(f'Seed {seed}: completed')


if __name__ == '__main__':
    main()
