# ai-image-detector-thesis
## Kết quả baseline (EfficientNet-B0, CIFAKE)

| Chỉ số | Giá trị |
|---|---:|
| Accuracy (test) | 98.25% |
| Precision (FAKE) | 97.92% |
| Recall (FAKE) | 98.59% |
| F1 (FAKE) | 98.26% |
| AUC | 99.86% |
| Tỉ lệ ảnh thật bị báo nhầm là AI | 2.09% |

Cấu hình: 5 epoch, batch size 128, lr 1e-3, AdamW, cosine annealing, AMP, seed 42.

Baseline EfficientNet-B0 đạt 98.25% accuracy và 99.86% AUC trên
20.000 ảnh test của CIFAKE. Mô hình phát hiện đúng 9.859/10.000
ảnh FAKE, với recall của lớp FAKE đạt 98.59%.

Có 209/10.000 ảnh REAL bị dự đoán nhầm thành FAKE, tương ứng
false positive rate 2.09%. Kết quả này được sử dụng làm mốc
baseline để so sánh với các thử nghiệm tiếp theo.