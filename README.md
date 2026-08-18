# STT-MRAM: hướng dẫn chạy nhanh

## Chuẩn bị

Chạy trong thư mục dự án bằng môi trường ảo:

```powershell
.\.venv\Scripts\python.exe run_all_curves.py --help
```

Nếu chưa cài thư viện:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## 1. Quét theo `sigma_mu`

Mặc định cố định `P1 = 2e-4`, quét `sigma_mu = 10..15` và chạy đủ 5 đường:

```powershell
.\.venv\Scripts\python.exe run_all_curves.py
```

Chọn các sigma riêng:

```powershell
.\.venv\Scripts\python.exe run_all_curves.py --sigmas 9 10 11
```

Kết quả chính: `results/all_curves_sigma10_15.csv`.

## 2. Quét theo `P1`

Cố định `sigma_mu = 10%`:

```powershell
.\.venv\Scripts\python.exe run_all_curves.py --p1-sweep-only --p1-sigma 10
```

Chạy nhanh hơn bằng cách bỏ đường `only-BCH`:

```powershell
.\.venv\Scripts\python.exe run_all_curves.py --p1-sweep-only --p1-sigma 10 --skip-only-bch
```

Ví dụ chạy gần điều kiện Figure 6 với `sigma_mu = 9%`:

```powershell
.\.venv\Scripts\python.exe run_all_curves.py --p1-sweep-only --p1-sigma 9 --skip-only-bch
```

Chọn các mốc `P1` riêng:

```powershell
.\.venv\Scripts\python.exe run_all_curves.py --p1-sweep-only --p1-sigma 10 --p1-values 1e-8 1e-7 1e-6 1e-5 1e-4 1e-3 --skip-only-bch
```

Kết quả được đặt tên theo sigma, ví dụ:

- `results/all_curves_p1_sigma9.csv`
- `results/all_curves_p1_sigma10.csv`
- `results/run_manifest_p1_sigma9.json`
- `results/run_manifest_p1_sigma10.json`

CSV P1 sweep chứa cả BER và FER cho từng đường được chạy.

## 3. Checkpoint FFNN

Mặc định chương trình tải checkpoint hiện có tại
`models/deep_ffnn_model.pt`.

Huấn luyện lại từ đầu:

```powershell
.\.venv\Scripts\python.exe run_all_curves.py --retrain
```

Fine-tune checkpoint hiện tại:

```powershell
.\.venv\Scripts\python.exe run_all_curves.py --finetune
```

`--retrain` và `--finetune` không được dùng cùng lúc.

.
- `--skip-only-bch` hiện chỉ áp dụng cho P1 sweep.
- Các mô phỏng dùng seed cố định `42` và ghi CSV tăng dần sau mỗi điểm.
