# Vietnamese Noise-Aware Hate Speech Detection
[![Open In Colab](https://colab.research.google.com/drive/1oxHe8X5zSrvLrWSBgyzS-7Y6L1cVw7WN?usp=sharing)]

Mô hình multi-task học đồng thời 2 nhiệm vụ trên văn bản tiếng Việt có khả năng bị
làm nhiễu (teencode, mất dấu, lặp ký tự, che ký tự...):

1. **Hate Speech Classification** — phân loại `CLEAN` / `OFFENSIVE` / `HATE`.
2. **Noise Type Detection** — nhận diện kiểu nhiễu đã áp dụng lên câu gốc (nếu có),
   dùng như tín hiệu phụ trợ (auxiliary signal) giúp nhánh phân loại chính bền
   vững hơn trước dữ liệu nhiễu trong thực tế.

Backbone: `xlm-roberta-base`, kiến trúc multi-sample dropout + noise-aware feature
fusion, huấn luyện theo ensemble 3-seed (soft-voting).

## Problem Statement

- **Input:** một câu văn bản tiếng Việt (có thể chứa teencode, viết tắt, ký tự lặp,
  hoặc bị che một phần).
- **Output:** nhãn mức độ thù ghét (`CLEAN`/`OFFENSIVE`/`HATE`) và loại nhiễu được
  phát hiện trong câu.
- **Thử thách chính:** phân bố lớp mất cân bằng (lớp `HATE` hiếm hơn nhiều so với
  `CLEAN`), và sự đa dạng của các kiểu nhiễu trong văn bản mạng xã hội tiếng Việt.

## Kiến trúc mô hình

```
XLM-RoBERTa (backbone)
        │
        ├──► Noise Head (multi-sample dropout) ──► noise_logits (7 lớp)
        │                                              │
        │                                        softmax + projection
        │                                              │
        └──► concat(CLS, noise_features) ──► Hate Head ──► hate_logits (3 lớp)
```

Loss tổng hợp bằng uncertainty weighting (learnable log-variance) giữa Focal Loss
(cho nhánh hate, xử lý mất cân bằng lớp) và Cross-Entropy có label smoothing (cho
nhánh noise).

## Evaluation & Error Analysis

Điểm số được tính theo công thức `0.85 × F1_hate(macro) + 0.15 × F1_noise(macro)`.

| Giai đoạn | Score |
|---|---|
| Baseline ban đầu (bug tiền xử lý train/val/test không đồng nhất) | **~55.8 / 100** |
| Sau khi thống nhất pipeline tiền xử lý + ensemble 3-seed | **~120 / 200** (đang cải thiện) |

**3 nguyên nhân chính khiến mô hình chưa đạt điểm tối đa, và hướng khắc phục:**

1. **Lệch phân bố tiền xử lý giữa train/val/test (đã khắc phục).** Ban đầu, dữ liệu
   train được tokenize + augment không đồng nhất (50% mẫu qua `preprocess()`, 50%
   dùng text thô), trong khi tập test lại dùng một hàm làm sạch hoàn toàn khác. Hệ
   quả: mô hình học một phân bố input nhưng bị đánh giá trên phân bố khác hẳn.
   → Đã chuẩn hoá lại: mọi tập dữ liệu (train/val/test) đều đi qua đúng một hàm
   `preprocess()`, chỉ tập train được cộng thêm nhiễu ngẫu nhiên có kiểm soát.

2. **Mất cân bằng lớp (`HATE` chiếm tỉ lệ nhỏ trong dữ liệu).** Dù đã dùng Focal
   Loss với trọng số alpha ưu tiên lớp thiểu số, F1 của lớp `HATE` vẫn thấp hơn hẳn
   so với `CLEAN`/`OFFENSIVE` trong ma trận nhầm lẫn.
   → Hướng khắc phục tiếp theo: oversampling có kiểm soát cho lớp `HATE`, hoặc thử
   nghiệm thêm data augmentation theo hướng ngữ nghĩa (back-translation) thay vì
   chỉ nhiễu ở mức ký tự/từ.

3. **Độ đa dạng giữa các model trong ensemble còn hạn chế.** Vì cả 3 seed dùng
   chung kiến trúc và chung tập train/val, các model có xu hướng hội tụ về vùng
   hiệu suất khá giống nhau (chênh lệch F1 giữa các seed ở cùng epoch thường dưới
   1%), khiến lợi ích của soft-voting bị giới hạn.
   → Hướng khắc phục: thử ensemble đa kiến trúc (kết hợp thêm một backbone khác,
   ví dụ PhoBERT) thay vì chỉ đổi seed khởi tạo.

## Cấu trúc dự án

```
my-ai-project/
├── data/
│   └── sample_data/       # vài mẫu nhỏ để test nhanh, KHÔNG chứa full dataset
├── models/                # nơi đặt file trọng số đã tải về (xem hướng dẫn bên dưới)
├── notebooks/             # notebook thử nghiệm (đã dọn log thừa)
├── scripts/
│   ├── train.py           # huấn luyện ensemble 3-seed
│   └── inference.py       # suy luận từ checkpoint có sẵn, không train lại
├── .gitignore
├── requirements.txt
└── README.md
```

## How to Run

### 1. Cài đặt thư viện

```bash
pip install -r requirements.txt
```

### 2. Tải trọng số mô hình đã huấn luyện

Vì file trọng số (`.pth`) khá nặng (~500MB-1GB mỗi seed) nên **không được lưu trực
tiếp trong repository này** (xem `.gitignore`). Tải 3 checkpoint đã huấn luyện sẵn
tại đây:

**Model weights:** [Google Drive — best_model_seed{2026,2027,2028}.pth](https://drive.google.com/drive/folders/DUONG_DAN_DRIVE_CUA_BAN)

> Thay đường link trên bằng link thư mục Google Drive thật của bạn (nhớ đặt chế độ
> chia sẻ "Anyone with the link" nếu muốn người khác tải được).

Sau khi tải về, đặt cả 3 file vào thư mục `models/`:

```
models/
├── best_model_seed2026.pth
├── best_model_seed2027.pth
└── best_model_seed2028.pth
```

### 3. Chạy suy luận (inference) trên dữ liệu của bạn

```bash
python scripts/inference.py --input data/sample_data/sample.csv --output predictions.csv
```

### 4. (Tuỳ chọn) Huấn luyện lại từ đầu

```bash
python scripts/train.py --train data/training_set.csv --val data/validation_set.csv
```

## Disclaimer

This repository contains personal implementation and experimental code developed
for the AI competition challenge. All problem descriptions have been paraphrased
for educational purposes.
