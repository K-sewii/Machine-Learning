<div align="center">

# 🏦 Bank Marketing — Dự đoán khách hàng gửi tiết kiệm có kỳ hạn

**Đồ án cá nhân môn Máy học (DCT124C) · Năm 3**

Dự đoán **trước khi gọi** khách hàng nào có khả năng đồng ý mở tài khoản tiết kiệm có kỳ hạn,
để ngân hàng ưu tiên gọi đúng người.

![Python](https://img.shields.io/badge/Python-3.x-3776AB?logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?logo=scikitlearn&logoColor=white)
![pandas](https://img.shields.io/badge/pandas-150458?logo=pandas&logoColor=white)
![Colab](https://img.shields.io/badge/Google%20Colab-F9AB00?logo=googlecolab&logoColor=white)
![Dataset](https://img.shields.io/badge/Dataset-UCI%20Bank%20Marketing-blue)
![Status](https://img.shields.io/badge/Tiến%20độ-Tuần%203%2F4-yellow)

</div>

---

## 📑 Mục lục

- [Tổng quan](#-tổng-quan)
- [Dữ liệu](#-dữ-liệu)
- [Phương pháp](#-phương-pháp)
- [Kết quả (trên validation)](#-kết-quả-trên-validation)
- [Cấu trúc thư mục](#-cấu-trúc-thư-mục)
- [Cách chạy](#-cách-chạy)
- [Tiến độ](#-tiến-độ)
- [Nguyên tắc đánh giá trung thực](#-nguyên-tắc-đánh-giá-trung-thực)
- [Tài liệu tham khảo](#-tài-liệu-tham-khảo)

---

## 🎯 Tổng quan

Ngân hàng gọi điện mời khách mở tài khoản tiết kiệm có kỳ hạn, nhưng phần lớn cuộc gọi thất bại
(chỉ khoảng **11,7%** khách đồng ý). Mô hình cần cân bằng hai loại lỗi:

| Loại lỗi | Ý nghĩa | Hậu quả |
|---|---|---|
| **Bỏ lỡ** (FN) | khách sẽ đồng ý nhưng mô hình dự đoán `no` | mất khách tiềm năng |
| **Gọi nhầm** (FP) | khách sẽ từ chối nhưng mô hình dự đoán `yes` | tốn một cuộc gọi |

➡️ **Thước đo chính: F1 của lớp `yes` (F1-yes).** Accuracy không dùng được vì một mô hình luôn đoán `no`
vẫn đạt Accuracy ≈ 0,88.

**Mục tiêu học tập:**

- **M1:** Khám phá dữ liệu (EDA), đưa ra nhận xét có số liệu, phát hiện rò rỉ dữ liệu.
- **M2:** Tiền xử lý đúng quy trình (one-hot, chuẩn hóa, `Pipeline`), xây và so sánh các mô hình bằng scikit-learn.
- **M3:** Chọn siêu tham số trên validation, nhận ra quá khớp, kiểm định giả thuyết, chọn ngưỡng quyết định.
- **M4:** Đánh giá trên test **một lần duy nhất**, diễn giải theo nghiệp vụ và phân tích lỗi.

---

## 📊 Dữ liệu

[**Bank Marketing**](https://archive.ics.uci.edu/dataset/222/bank+marketing) (UCI ML Repository; Moro, Cortez & Rita, 2014),
tệp `bank-full.csv`: **45.211** khách của một ngân hàng Bồ Đào Nha (5/2008 – 11/2010),
16 thuộc tính (7 số, 9 phân loại) và nhãn `y`. Giấy phép **CC BY 4.0**.

Notebook tự tải dữ liệu, kiểm tra mã SHA-256 rồi chia **phân tầng** theo nhãn (`random_state=42`):

| Tập | Số khách | Số `yes` | Tỉ lệ `yes` | Vai trò |
|---|---:|---:|---:|---|
| train | 27.126 | 3.173 | 11,7% | mô hình học |
| validation | 9.042 | 1.058 | 11,7% | so sánh, chọn mô hình / siêu tham số / ngưỡng |
| test | 9.043 | 1.058 | 11,7% | đánh giá cuối, **chỉ dùng một lần** (tuần 4) |

> [!WARNING]
> **`duration` bị loại khỏi đầu vào.** Thời lượng cuộc gọi chỉ biết *sau khi* gọi xong, nên dùng nó là
> rò rỉ dữ liệu. Thử nghiệm ở tuần 2: thêm `duration` làm F1-yes của E1 tăng từ **0,375 → 0,551**,
> một con số đẹp nhưng không dùng được trong thực tế.

**Một số phát hiện từ EDA (tập train):**

- Kết quả chiến dịch trước `poutcome = success` → tỉ lệ đồng ý **63,8%**, gấp ~5,5 lần mức chung 11,7%.
- Tỉ lệ đồng ý theo tuổi có **dạng chữ U**: dưới 25 tuổi 24,2%, 40–50 tuổi chỉ 9,4%, trên 60 tuổi 41,9%.
- Tháng 3 đạt 49,3% nhưng chỉ có 296 cuộc gọi; tháng 5 có 8.259 cuộc gọi nhưng chỉ đạt 6,8%. Đây là tương quan, chưa phải nhân quả.

---

## 🧠 Phương pháp

```text
bank-full.csv ─► chia 60/20/20 (phân tầng)
                    │
                    ▼
   ColumnTransformer ── OneHotEncoder(handle_unknown="ignore")  → 9 thuộc tính phân loại (44 cột)
                    └── StandardScaler                         → 6 thuộc tính số
                    │                                             (= 50 đặc trưng, chỉ fit trên train)
                    ▼
   Mô hình ─► predict_proba ─► ngưỡng τ tối ưu trên validation ─► khóa QUYET_DINH ─► test (1 lần)
```

| Mã | Mô hình | Ghi chú |
|---|---|---|
| **E0** | `DummyClassifier(most_frequent)` | mốc so sánh, luôn đoán `no` |
| **E1** | `LogisticRegression(class_weight="balanced")` | mô hình tuyến tính cơ sở |
| **E2** | `KNeighborsClassifier(n_neighbors=15)` | cần chuẩn hóa thuộc tính số |
| **E3** | `MLPClassifier`, 1 lớp ẩn, `early_stopping=True` | chọn H ∈ {4, 16, 64, 256} qua 5 seed |
| **E4** | E1 + thuộc tính `nhom_tuoi` | kiểm định giả thuyết về dạng chữ U của tuổi |

**So sánh công bằng:** mỗi mô hình dùng **ngưỡng tốt nhất của chính nó** (τ ∈ {0,1; …; 0,9}) trên validation.
Lý do: các mô hình không có `class_weight` (kNN, MLP) cho xác suất `yes` thấp nên bị thiệt nếu cùng dùng ngưỡng 0,5.

---

## 📈 Kết quả (trên validation)

<details open>
<summary><b>Tuần 2: E0, E1, E2 ở ngưỡng 0,5</b></summary>

| Mô hình | Accuracy | Precision-yes | Recall-yes | F1-yes |
|---|---:|---:|---:|---:|
| E0 lớp đa số | 0,883 | 0,000 | 0,000 | 0,000 |
| E1 hồi quy logistic | 0,755 | 0,267 | 0,629 | **0,375** |
| E2 kNN (k = 15) | 0,891 | 0,625 | 0,170 | 0,267 |

- Bỏ `class_weight` ở E1: F1-yes giảm 0,375 → 0,294 (Recall chỉ còn 0,188).
- Bỏ chuẩn hóa ở E2: F1-yes giảm 0,267 → 0,095.

</details>

<details open>
<summary><b>Tuần 3: MLP, số nơ-ron ẩn qua 5 seed</b></summary>

| H | F1-yes TB | Độ lệch chuẩn | Min | Max | Số tham số |
|---:|---:|---:|---:|---:|---:|
| 4 | 0,434 | 0,032 | 0,379 | 0,461 | 209 |
| 16 | 0,460 | 0,013 | 0,439 | 0,471 | 833 |
| 64 | 0,460 | 0,013 | 0,446 | 0,475 | 3.329 |
| **256** | **0,483** | 0,009 | 0,470 | 0,491 | 13.313 |

Đường cong loss (H = 256, không dừng sớm): loss validation thấp nhất ở **epoch 11** rồi tăng dần,
trong khi loss train vẫn giảm. Đây là dấu hiệu **quá khớp**.

</details>

<details open>
<summary><b>Tuần 3: Chọn mô hình (Bảng 13)</b></summary>

| Mô hình | F1-yes @ τ = 0,5 | F1-yes @ τ tốt nhất | τ tốt nhất |
|---|---:|---:|---:|
| E1 | 0,375 | 0,446 | 0,6 |
| E2 | 0,267 | 0,412 | 0,2 |
| **E3 (MLP, H = 256)** | 0,352 | **0,476** | **0,2** |
| E4 (E1 + nhóm tuổi) | 0,385 | 0,458 | 0,7 |

Nếu so ở ngưỡng 0,5 thì E4 đứng đầu. Khi mỗi mô hình dùng ngưỡng của chính nó thì **E3** đứng đầu.

</details>

**🔒 Quyết định đã khóa (trước khi chạy test):**

```json
{"mo_hinh": "E3", "so_no_ron": 256, "nguong": 0.2, "f1_yes_val": 0.4761, "thoi_diem": "2026-11-17 14:01"}
```

Tại τ\* = 0,2, E3 chọn 1.328 khách để gọi trên validation, với Precision-yes **0,428** (so với ~0,117 nếu gọi
ngẫu nhiên, tức gấp khoảng 3,7 lần) và Recall-yes **0,537**.

> [!NOTE]
> Các con số trên được đo trên **validation**, là tập đã dùng để chọn mô hình nên mang tính lạc quan.
> Hiệu năng thực tế sẽ được đo trên **test** ở tuần 4.

---

## 📁 Cấu trúc thư mục

```text
Machine-Learning/
├── 3124411148_tuan1.ipynb     # Tuần 1: bài toán, EDA, rò rỉ dữ liệu, mốc E0
├── 3124411148_tuan2.ipynb     # Tuần 2: tiền xử lý, E1 (logistic), E2 (kNN), hệ số
├── 3124411148_tuan3.ipynb     # Tuần 3: MLP, quá khớp, chọn H, E4, chọn ngưỡng, khóa quyết định
├── 3124411148_tuan4.ipynb     # Tuần 4: đánh giá test, phân tích lỗi 
├── docs/
│   ├── ML_Project_DCT124C_6-9.pdf        # Đề bài đồ án
│   ├── ML_Report-Template_DCT124_6-9.docx # Mẫu báo cáo
│   └── So_tay_thu_vien.pdf                # Sổ tay thư viện (pandas, scikit-learn)
└── README.md
```

---

## 🚀 Cách chạy

**Cách 1: Google Colab (khuyến nghị, chỉ cần CPU)**

1. Mở [Google Colab](https://colab.research.google.com/), chọn **File → Upload notebook** và tải lên notebook của tuần cần chạy.
2. Chọn **Runtime → Run all**. Ô thiết lập sẽ tự tải dữ liệu từ UCI và kiểm tra SHA-256.

**Cách 2: Chạy trên máy**

```bash
git clone https://github.com/K-sewii/Machine-Learning.git
cd Machine-Learning
pip install numpy pandas matplotlib scikit-learn jupyter
jupyter notebook
```

> [!TIP]
> Nếu không tải được dữ liệu, đặt tệp `bank-full.csv` vào thư mục `du_lieu/` rồi chạy lại ô thiết lập.
> Notebook sẽ kiểm tra xem tệp có đúng không.

---

## ✅ Tiến độ

- [x] **Tuần 1:** Bài toán, chia dữ liệu, EDA, rò rỉ qua `duration`, mốc E0
- [x] **Tuần 2:** Tiền xử lý, `Pipeline`, E1 / E2, ảnh hưởng của trọng số lớp, chuẩn hóa và rò rỉ, đọc hệ số
- [x] **Tuần 3:** Tính tay MLP, đường cong loss, chọn H qua 5 seed, giả thuyết E4, so sánh công bằng, khóa `QUYET_DINH`
- [x] **Tuần 4:** Đánh giá cuối trên test, phân tích lỗi, hoàn thiện báo cáo
  - [x] Xác nhận `QUYET_DINH` khớp với tuần 3
  - [x] Đánh giá trên test **một lần** tại τ\* = 0,2 (đối chiếu thêm tại τ = 0,5)
  - [x] So sánh với phương án gọi ngẫu nhiên cùng số khách
  - [x] So sánh kết quả validation và test
  - [x] Phân tích lỗi theo nhóm khách và 6 lỗi cụ thể
  - [ ] *(Nâng cao)* Khoảng tin cậy bootstrap cho F1-yes trên test
- [x] **Báo cáo** `3124411148_ML_baocao.pdf` (docs/3124411148_ML_baocao.pdf)

---

## ⚖️ Nguyên tắc đánh giá trung thực

1. **Chỉ học trên train.** Tiền xử lý và mô hình chỉ `fit` trên train; validation và test chỉ `transform` / `predict`. Không dùng `duration`.
2. **Chọn trên validation, đánh giá test một lần.** Mô hình, số nơ-ron và ngưỡng được khóa bằng `QUYET_DINH` trước khi chạy test, và không thay đổi sau đó.
3. **Viết giả thuyết trước.** Giả thuyết cho E4 được viết, có ghi ngày, trước khi chạy thực nghiệm.

**Giới hạn đã biết:** dữ liệu sắp xếp theo thời gian (2008–2010) nhưng được chia ngẫu nhiên, nên kết quả có thể
lạc quan hơn so với việc dự đoán các chiến dịch trong tương lai.

---

## 📚 Tài liệu tham khảo

- Moro, S., Cortez, P., & Rita, P. (2014). *A data-driven approach to predict the success of bank telemarketing.* Decision Support Systems, 62, 22–31.
- [UCI Machine Learning Repository: Bank Marketing](https://archive.ics.uci.edu/dataset/222/bank+marketing)
- [scikit-learn User Guide](https://scikit-learn.org/stable/user_guide.html)
- Đề bài, mẫu báo cáo và sổ tay thư viện của học phần trong [docs/](docs/)

---

<div align="center">

Đồ án học tập · Môn Máy học DCT124C · Học kỳ 1, 2026–2027 · Tác giả: [K-sewii](https://github.com/K-sewii)

</div>
