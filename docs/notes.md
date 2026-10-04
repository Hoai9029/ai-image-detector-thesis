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