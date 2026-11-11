import os
import pickle
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split

# CSVファイルを読み込むための環境変数設定
# TODO: ここを適切なパスに変更してね

# CSVファイルのパス
CSV_PATH = "./data_to_path"

# PKLファイルの保存先ディレクトリ
SAVE_DIR = "./saved_data_to_path"

# CSVファイルにファイル名しかないので
# 画像パスの先頭に付与するフォルダパス
IMG_ROOT = "img_root"

# CSVファイルの画像列のカラム名
IMG_COL = "filename"

# クラス列のカラム名
CLASS_COL = "class"

# CSVファイルの分割列のカラム名
# K-Foldをやるなら，この分割は使わないかもね
SPLIT_COL = "dataset"

# 交差検証の分割数
N_SPLITS = 5

# 学習・検証データのうちの検証データの割合
VAL_RATIO = 0.15

# シード値（ここの乱数を固定すると，結果が再現される）
SEED = 42


# 保存先フォルダを作成
os.makedirs(SAVE_DIR, exist_ok = True)

# CSVファイルを読み込む
df = pd.read_csv(CSV_PATH)

# コンセプト列を取得
exclude_cols = [IMG_COL, CLASS_COL, SPLIT_COL]
# columnsはカラム名のリストを返すので，len()で数を数えられる
concept_cols = [col for col in df.columns if col not in exclude_cols]

# コンセプト数確認
# print(f"sample size: {len(concept_cols)}\nconcept num: {len(concept_cols.columns)}")
print(f"sample size: {len(df)}\nconcept num: {len(concept_cols)}")


#　DataFrameをLogicCBM用の辞書（PKL形式）に変換する
def convert_df_to_pkl(target_df):
    dataset_list = []

    # 画像1枚ずつpkl形式に変換する
    # reset_indexは元のインデックスを消して数字のインデックスに変換するためのもの
    # そうすればidxに数字を入れられるから便利
    # iterrowはDFのインデックスと行データをタプルで取り出す関数
    for idx, row in target_df.reset_index(drop = True).iterrows():
        # 画像パスを作成
        # os.path.abspathを使えば，実行環境に依存しない絶対パスを勝手に作ってくれるので便利
        img_path = os.path.abspath(os.path.join(IMG_ROOT, str(row[IMG_COL])))

        # 各コンセプトごとにラベルを作成する（attrはattribute（属性）の略）
        # TODO：とりあえず01でやってみるけど，5グレード評価なので，それを適用するときは変えてね
        attr_labels = []

        # 確信度付きラベルを使う時のやつ
        # TODO: data_processing.pyでラベルの計算方法があるけど，計算がいらない系のラベルなので，そっち側を変える必要がある
        uncertain_attribute_label = []
        for col in concept_cols:
            # uncertain_attribute_labelは属性ラベルをそのまま入れる（1~5）
            # お試し，0~1の範囲に正規化するために4で割る
            uncertain_attribute_label.append((float(row[col]) - 1.0) / 4.0)

            # 0,1のラベルに変換する
            if row[col] == 1:
                attr_labels.append(int(0))
            elif row[col] == 2 or row[col] == 3 or row[col] == 4 or row[col] == 5:
                attr_labels.append(int(1))
            else:
                raise ValueError(f"Unexpected value {row[col]} in column {col}")

        # pkl形式の辞書を作成する
        dataset_list.append({
            "id": int(idx),
            "img_path": img_path,
            "class_label": int(row[CLASS_COL]),
            "attribute_label": attr_labels,
            "uncertain_attribute_label": uncertain_attribute_label
        })

    return dataset_list

# クラス比率を保ったままK-Fold分割する
# StratifiedKFoldを使うと，クラス比率を保ったまま自動で分割できる
skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)

# enumerateはインデックスと要素をとりだず関数
# splitはデータとクラスを入れると，学習用とテスト用に分けた元データ（df）のインデックスを返してくれる
for fold, (train_val_idx, test_idx) in enumerate(skf.split(df, df[CLASS_COL])):
    # 保存先作成
    fold_dir = os.path.join(SAVE_DIR, f"fold_{fold}")
    os.makedirs(fold_dir, exist_ok = True)

    # 学習用とテスト用のデータを取り出す
    # ilocは行番号を入れると，その行番号のデータを返してくれる関数
    # 2要素目のilocは列番号を入れると，その列番号のデータを返してくれる関数
    train_val_df = df.iloc[train_val_idx]
    test_df = df.iloc[test_idx]

    # train_val_dfをさらに学習用と検証用に分割する
    train_df, val_df = train_test_split(
        train_val_df,
        test_size = VAL_RATIO,
        stratify = train_val_df[CLASS_COL],
        random_state = SEED
    )

    # 次のfor文で使うための辞書
    splits = {
        "train": train_df,
        "val": val_df,
        "test": test_df
    }

    # 各分割をpkl形式に変換して保存する
    # itemsは辞書のキーと要素をタプルで返す関数
    for split_name, split_df in splits.items():
        # pkl形式に変換
        dataset_list = convert_df_to_pkl(split_df)

        # 保存先パスを作成
        out_path = os.path.join(fold_dir, f"{split_name}.pkl")

        # out_pathにバイナリ形式でpickleで保存
        with open(out_path, "wb") as f:
            pickle.dump(dataset_list, f)

    print(
        "pkl形式への変換"
        f"Fold {fold}の変換完了 -> {fold_dir}"
        f"train: {len(train_df)}, val: {len(val_df)}, test: {len(test_df)}"
    )

