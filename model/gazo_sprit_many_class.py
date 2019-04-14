import os
import shutil
import random


def gazo_sprit_many_class(original_dir, base_dir, train_size):
    '''
    画像データをトレインデータとテストデータに分割します。フォルダもなければ作成します。

    parameter
    ------------
    original_dir: strのリスト
      オリジナルデータフォルダのパスのリスト その下に各クラスのフォルダがある
    base_dir: str
      分けたデータを格納するフォルダのパス　そ子にフォルダが作られます
    train_size: float
      トレインデータの割合
    '''
    try:
        os.mkdir(base_dir)
    except FileExistsError:
        print(base_dir + "は作成済み")

    #クラス分のフォルダ名の取得
    dir_lists = os.listdir(original_dir)
    dir_lists = [f for f in dir_lists if os.path.isdir(os.path.join(original_dir, f))]
    original_dir_path = [os.path.join(original_dir, p) for p in dir_lists]

    num_class = len(dir_lists)

    # フォルダの作成(トレインとバリデーション)
    try:
        train_dir = os.path.join(base_dir, 'train')
        os.mkdir(train_dir)
    except FileExistsError:
        print(train_dir + "は作成済み")

    try:
        validation_dir = os.path.join(base_dir, 'validation')
        os.mkdir(validation_dir)
    except FileExistsError:
        print(validation_dir + "は作成済み")

    #クラスフォルダ
    train_dir_path_lists = []
    val_dir_path_lists = []
    for D in dir_lists:
        train_class_dir_path = os.path.join(train_dir, D)
        try:
            os.mkdir(train_class_dir_path)
        except FileExistsError:
            print(train_class_dir_path + "は作成済み")
        train_dir_path_lists += [train_class_dir_path]
        val_class_dir_path = os.path.join(validation_dir, D)
        try:
            os.mkdir(val_class_dir_path)
        except FileExistsError:
            print(val_class_dir_path + "は作成済み")
        val_dir_path_lists += [val_class_dir_path]


    """
    元データをシャッフルしたものを上で作ったフォルダにコピーします。
    """
    # ファイル名を取得してシャッフル
    for i,path in enumerate(original_dir_path):
        files_class = os.listdir(path)
        random.shuffle(files_class)
        # 分割地点のインデックスを取得
        num_bunkatu = int(len(files_class) * train_size)
        #トレインへファイルをコピー
        for fname in files_class[:num_bunkatu]:
            src = os.path.join(path, fname)
            dst = os.path.join(train_dir_path_lists[i], fname)
            shutil.copyfile(src, dst)
        #valへファイルをコピー
        for fname in files_class[num_bunkatu:]:
            src = os.path.join(path, fname)
            dst = os.path.join(val_dir_path_lists[i], fname)
            shutil.copyfile(src, dst)
        print(path + "コピー完了")

    print("分割終了")


def main():
    original_dir = "images"
    base_dir = "zidolegi_data"
    train_size = 0.8
    gazo_sprit_many_class(original_dir,base_dir,train_size)

if __name__ == "__main__":
    main()