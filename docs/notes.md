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