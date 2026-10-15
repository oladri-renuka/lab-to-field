import pandas as pd

from src.data.splits import few_shot, stratified_val_split


def frame():
    labels = [0] * 50 + [1] * 20 + [2] * 10 + [3] * 1
    return pd.DataFrame({"path": [f"img{i}.jpg" for i in range(len(labels))], "label": labels})


def test_val_split_is_stratified_disjoint_and_deterministic():
    df = frame()
    tr, va = stratified_val_split(df, 0.2, 0)
    assert set(tr.path).isdisjoint(va.path)
    assert len(tr) + len(va) == len(df)
    assert va.label.value_counts().to_dict() == {0: 10, 1: 4, 2: 2}
    assert 3 in set(tr.label)  # singleton class stays in train
    tr2, va2 = stratified_val_split(df, 0.2, 0)
    assert list(va.path) == list(va2.path)


def test_few_shot_caps_and_varies_with_seed():
    df = frame()
    k = few_shot(df, 5, 0)
    assert k.label.value_counts().to_dict() == {0: 5, 1: 5, 2: 5, 3: 1}
    assert set(few_shot(df, 5, 0).path) == set(k.path)
    assert set(few_shot(df, 5, 1).path) != set(k.path)
