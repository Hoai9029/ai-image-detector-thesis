# CIFAKE EDA Notes

## Dataset

- Dataset: CIFAKE
- Train: 100,000 images
- Test: 20,000 images
- Total: 120,000 images

## Class distribution

- REAL train: 50,000
- FAKE train: 50,000
- Tỷ lệ REAL/FAKE: 1:1

## Image size

- Kích thước ảnh: 32x32 pixels

## Initial observations

- Dataset có số lượng ảnh REAL và FAKE cân bằng.
- Ảnh có kích thước nhỏ 32x32 pixels.
- Một số ảnh AI có thể khó phân biệt bằng mắt thường.
- Khi resize lên kích thước lớn hơn để đưa vào backbone CNN, ảnh có thể bị mờ.

# Ngày 3 — Đọc tài liệu nền tảng

## 1. CIFAKE

Tên bài báo:

CIFAKE: Image Classification and Explainable Identification of AI-Generated Synthetic Images

Năm: 2023

Mục tiêu:
- Phân biệt ảnh thật và ảnh được tạo bởi AI.
- Bài toán được đưa về binary classification: REAL và FAKE.

Dataset:
- CIFAKE được xây dựng dựa trên CIFAR-10.
- Ảnh AI được tạo bằng latent diffusion.
- Dataset gồm ảnh thật và ảnh tổng hợp để thực hiện phân loại REAL/FAKE.

Model:
- Tác giả sử dụng CNN cho bài toán phân loại.
- Thử nghiệm 36 network topologies khác nhau.
- Kiến trúc tốt nhất đạt accuracy khoảng 92.98%.

Explainable AI:
- Bài báo sử dụng Grad-CAM để giải thích quyết định của CNN.
- Grad-CAM cho biết những vùng nào trong ảnh có ảnh hưởng đến quyết định của model.
- Kết quả cho thấy model có thể tập trung vào các sai khác nhỏ ở background thay vì chỉ tập trung vào đối tượng chính.

Ý nghĩa đối với đề tài:
- CIFAKE là dataset phù hợp để bắt đầu xây dựng AI-generated image detector.
- Accuracy 92.98% là một kết quả tham khảo.
- Grad-CAM có thể được sử dụng sau này để giải thích model đang dựa vào đặc trưng nào.

## 2. GenImage

GenImage là benchmark quy mô lớn cho bài toán phát hiện ảnh được tạo bởi AI.

Đặc điểm:
- Hơn 1 triệu cặp ảnh REAL và AI-generated.
- Có nhiều loại nội dung.
- Có nhiều AI generator khác nhau.
- Dùng để đánh giá khả năng generalization của detector.

Các generator tiêu biểu:
- Stable Diffusion
- Midjourney
- GLIDE
- ADM
- Wukong
- VQDM
- BigGAN

Các detector/backbone được benchmark:
- ResNet-50
- DeiT-S
- Swin-T
- CNNSpot
- Spec
- F3Net
- GramNet

Một kết quả đáng chú ý:
- Khi train và test trên cùng generator, ResNet-50 có thể đạt trên 98.5% accuracy.
- Nhưng khi train trên Stable Diffusion V1.4 và test trên Midjourney, accuracy giảm xuống khoảng 54.9%.
- Điều này cho thấy khả năng generalization giữa các generator là một vấn đề quan trọng.

GenImage có hai bài toán đáng chú ý:
1. Cross-generator image classification.
2. Degraded image classification.

## 3. Các kiến trúc/model tham khảo

Một số kiến trúc thường được sử dụng trong bài toán AI-generated image detection:

| Model | Loại | Accuracy tham khảo |
|---|---|---|
| ResNet-50 | CNN | CIFAKE best model khoảng 92.98%; GenImage có thể >98.5% khi train/test cùng generator |
| DeiT-S | Vision Transformer | GenImage average khoảng 71.6% trong cross-generator benchmark |
| Swin-T | Vision Transformer | GenImage average khoảng 74.8% trong cross-generator benchmark |
| CNNSpot | CNN-based detector | Được sử dụng làm baseline trong GenImage |
| F3Net | CNN-based detector | Được sử dụng trong GenImage |
| GramNet | CNN-based detector | Được sử dụng trong GenImage |

Lưu ý:
Accuracy giữa các nghiên cứu không nên so sánh trực tiếp nếu dataset, cách chia dữ liệu và điều kiện test khác nhau.

## W2D1 — EfficientNet-B0

- Model: EfficientNet-B0
- Framework: PyTorch + timm
- Pretrained: ImageNet
- Input: 3 × 224 × 224
- Output: 1 logit
- Parameters: [điền kết quả thực tế] M

### Lý do chọn

EfficientNet-B0 được chọn làm mô hình baseline vì có kiến trúc tương đối nhẹ,
số lượng tham số không quá lớn và tốc độ huấn luyện/inference phù hợp với
điều kiện tài nguyên của đề tài. Mô hình pretrained trên ImageNet được sử dụng
theo hướng transfer learning, sau đó thay lớp phân loại cuối bằng một đầu ra
phục vụ bài toán phân loại REAL/FAKE.

## W2D3 — Sanity Check

Mục tiêu của sanity check là kiểm tra pipeline training trước khi chạy toàn bộ
dataset. Hai thử nghiệm được thực hiện: overfit trên tập rất nhỏ và training
trên subset 2.000 ảnh.

| Thử nghiệm | Cấu hình | Train Acc | Val Acc | Time/Epoch | Nhận xét |
|---|---|---:|---:|---:|---|
| A - Overfit | 256 ảnh, 15 epochs, batch 32, lr 3e-4 | 1.0 | 0.905 | 1.3 s | ... |
| B - Debug | 2000 ảnh, 3 epochs, batch 64, lr 3e-4 | 0.98| 0.93 | 6.7 s | ... |

### Kết luận

Mô hình có/không có khả năng học trên tập nhỏ. Pipeline Dataset, DataLoader,
loss, optimizer, AMP và GPU được kiểm tra thông qua các thử nghiệm sanity check.

## W2D4 — Debug và chọn Learning Rate

### Kiểm tra nhãn

- REAL: 50.000 ảnh
- FAKE: 50.000 ảnh
- REAL được gán label 0
- FAKE được gán label 1

### So sánh Learning Rate

| Learning Rate | Val Loss | Val Acc | Val AUC | Time/Epoch |
|---|---:|---:|---:|---:|
| 1e-3 | 0.1848 | 94.45% | 98.49% | 28.5 s |
| 3e-4 | 0.3129 | 93.05% | 97.91% | 28.8 s |
| 1e-4 | 0.4554 | 90.25% | 95.96% | 29.4 s |

### Lựa chọn Learning Rate

Learning rate `1e-3` được chọn làm cấu hình mặc định cho baseline
vì đạt kết quả validation tốt nhất trong thử nghiệm:

- Val Loss: 0.1848
- Val Accuracy: 94.45%
- Val AUC: 98.49%

Đồng thời, thời gian huấn luyện cũng tương đương với các learning rate
còn lại, khoảng 28.5 giây/epoch.

### Tốc độ

Với subset 10.000 ảnh, thời gian trung bình khoảng 28–29 giây/epoch
trên GPU Tesla T4. Do tốc độ hiện tại khá tốt nên chưa cần thay đổi
chiến lược preprocessing.

## W2D7 — Tổng kết baseline

### Baseline chính thức

- Model: EfficientNet-B0
- Dataset: CIFAKE
- Input: 3 × 224 × 224
- Pretrained: ImageNet
- Epochs: 5
- Batch size: 128
- Learning rate: 1e-3
- Optimizer: AdamW
- Weight decay: 1e-4
- Scheduler: CosineAnnealingLR
- AMP: Có
- Seed: 42
- Device: NVIDIA Tesla T4

### Kết quả trên tập test

| Chỉ số | Kết quả |
|---|---:|
| Accuracy | 98.25% |
| Precision (FAKE) | 97.92% |
| Recall (FAKE) | 98.59% |
| F1 (FAKE) | 98.26% |
| AUC | 99.86% |
| False Positive Rate | 2.09% |
| False Negative Rate | 1.41% |

### Confusion Matrix

```text
                 Predicted
                 REAL    FAKE
Actual REAL      9791     209
Actual FAKE       141    9859

## W3D1 — Degradation và quan sát tác động

### Định dạng dữ liệu

Ảnh CIFAKE có định dạng JPEG (`.jpg`) và kích thước gốc là `32 × 32` pixel.

Do ảnh gốc đã là JPEG, phép nén JPEG trong thí nghiệm có thể được xem là quá trình nén lại ảnh. Điều này vẫn phù hợp với mục tiêu mô phỏng ảnh đã được xử lý hoặc nén thêm khi chia sẻ trên Internet.

### Các phép suy giảm

Hai phép suy giảm được xây dựng trong `src/degrade.py`:

* JPEG compression với các mức quality: 90, 70, 50, 30 và 10.
* Resize down-up với scale 0.75× và 0.5×.

Cùng một hàm suy giảm sẽ được sử dụng cho cả augmentation khi huấn luyện và đánh giá robustness để đảm bảo điều kiện thí nghiệm nhất quán.

### Đo mức thay đổi bằng MAE

| Điều kiện    |   MAE |
| ------------ | ----: |
| JPEG q=90    |  1.52 |
| JPEG q=70    |  2.32 |
| JPEG q=50    |  7.57 |
| JPEG q=30    |  8.69 |
| JPEG q=10    | 13.66 |
| Resize 0.75× |  9.05 |
| Resize 0.5×  | 13.81 |

Kết quả cho thấy mức độ thay đổi của ảnh tăng khi chất lượng JPEG giảm. JPEG q=90 có MAE chỉ 1.52, trong khi JPEG q=10 có MAE 13.66.

Tương tự, resize xuống 0.5× rồi phóng lại tạo ra mức thay đổi lớn hơn resize 0.75×, với MAE lần lượt là 13.81 và 9.05.

Do CIFAKE chỉ có ảnh kích thước 32 × 32 pixel, các phép resize và nén có thể gây ảnh hưởng tương đối mạnh đến nội dung ảnh. Đây là một hạn chế cần lưu ý khi đánh giá khả năng robustness trên CIFAKE.

### Nhận xét

Các kết quả quan sát và MAE cho thấy JPEG compression và resize down-up thực sự làm thay đổi dữ liệu đầu vào. Những thay đổi này có thể làm mất hoặc biến dạng các đặc trưng hình ảnh mà mô hình sử dụng để phân biệt REAL và FAKE.

Vì vậy, Tuần 3 sẽ thử nghiệm augmentation với các phép suy giảm này trong quá trình huấn luyện, sau đó đánh giá xem mô hình có giữ được độ chính xác tốt hơn khi ảnh bị nén hoặc thu nhỏ hay không.


## W3D2 — Robust Augmentation

Đã xây dựng `RobustAugment` trong `src/augment.py` để mô phỏng các thay đổi thường gặp khi ảnh được chia sẻ trên Internet.

Augmentation gồm hai phép biến đổi:

* JPEG compression với quality được lấy ngẫu nhiên trong khoảng 30–70.
* Resize down-up với scale ngẫu nhiên trong khoảng 0.5–0.75.

Xác suất mặc định của mỗi phép biến đổi là 0.5. Hai phép biến đổi có thể được áp dụng độc lập, vì vậy một ảnh có thể không bị biến đổi, bị áp dụng một phép hoặc được áp dụng cả hai phép.

Các phép biến đổi sử dụng lại hai hàm trong `src/degrade.py` để đảm bảo quá trình augmentation và quá trình đánh giá robustness sử dụng cùng cách xử lý ảnh.

Sau augmentation, kích thước ảnh vẫn được giữ nguyên ở mức `32 × 32`. Việc resize lên `224 × 224` tiếp tục được thực hiện bởi pipeline transform hiện tại.

Mục đích của augmentation là giúp mô hình học các đặc trưng ít phụ thuộc hơn vào chất lượng ảnh và tăng khả năng nhận diện ảnh AI khi ảnh đã trải qua quá trình nén hoặc thay đổi kích thước.

