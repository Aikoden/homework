データセットの準備
事前にデータセットを準備し、適切なフォルダに配置してください。dataset.yaml ファイルを含むデータセットのディレクトリ構造を確認してください。

訓練コード
以下のコードを使用して、YOLOv8モデルを訓練します。

python
from ultralytics import YOLO

# YOLOv8モデルの初期化（軽量モデル yolov8n.pt を使用）
model = YOLO('yolov8n.pt')

# モデルの訓練
model.train(
    data="path/to/dataset.yaml",  # データセット設定ファイルのパスを指定
    epochs=50,                     # 訓練エポック数
    imgsz=640                      # 画像サイズ
)
訓練パラメータの説明
data: データセット設定ファイルへのパス
epochs: 訓練エポック数（必要に応じて変更可）
imgsz: 画像サイズ（デフォルトは640）
テストコード
訓練済みモデルを使用して、新しい画像に対して推論を行います。

python
# 推論の実行
results = model.predict(
    source="path/to/image.jpg",  # 推論する画像のパスを指定
    save=True                     # 推論結果を保存する場合はTrue
)

# 推論結果の表示
for result in results:
    result.show()
推論パラメータの説明
source: 推論対象となる画像ファイルのパス
save: 推論結果を保存するかどうかを指定（Trueにすると保存）
訓練結果の保存場所
訓練結果は以下のディレクトリに保存されます。

bash
复制代码
runs/detect
このディレクトリには、以下のファイル・情報が含まれます。

学習曲線や損失関数のグラフ
各エポックの検出結果
訓練済みモデルの重みファイル（.pt）

