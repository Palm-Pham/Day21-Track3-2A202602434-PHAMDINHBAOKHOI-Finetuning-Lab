# Báo cáo Dữ liệu Miền Riêng (Custom Dataset Specification)

> **Bonus Challenge B2** — Lab 21 (Track 3) · Đi kèm Deck 2026 §17 & §3.3
> **Tác giả**: Phạm Đình Bảo Khôi (MSSV: `2A202602434`)
> **Tập tin dữ liệu**: `data/custom_train.jsonl` (250 mẫu) · `data/custom_eval.jsonl` (50 mẫu)
> **Script sinh & kiểm định**: `scripts/make_custom_dataset.py`

---

## 1. Bối cảnh Miền & Bài toán (Domain & Task Definition)

* **Miền chuyên sâu**: Sàng lọc & Điều phối Bệnh nhân Khám từ xa Tiếng Việt (**Vietnamese Telemedicine Clinical Triage**).
* **Bài toán**: Tiếp nhận phản ánh/kêu cứu của bệnh nhân hoặc người nhà bằng ngôn ngữ tự nhiên đời thường (chứa khẩu ngữ dân gian, mô tả triệu chứng tản mạn, trạng thái lo âu, thông tin nhân khẩu) và phân loại tự động thành cấu trúc **JSON 4 khóa khách quan**:

```json
{
  "chuyen_khoa": "tim_mach | ho_hap | tieu_hoa | da_lieu | than_kinh",
  "muc_do_khan_cap": "cap_cuu | khan_cap | tieu_chuan | theo_doi",
  "nhom_doi_tuong": "tre_em | nguoi_lon | nguoi_cao_tuoi | phu_nu_mang_thai",
  "huong_xu_ly": "goi_115 | den_phong_kham | hen_kham_online | tu_cham_soc"
}
```

* **Ý nghĩa thực tiễn**: Trong hệ thống y tế từ xa (telehealth), việc phân luồng tức thì giữa ca cần cấp cứu ngoại viện (`goi_115`), ca cần khám trực tiếp (`den_phong_kham`), và ca nhẹ có thể tư vấn từ xa (`hen_kham_online` / `tu_cham_soc`) có ý nghĩa sống còn đối với an toàn người bệnh.

---

## 2. Nguồn & Phương pháp Xây dựng (Source & Methodology)

### 2.1. Nguồn xây dựng
* Dựa trên các ca bệnh điển hình và nguyên tắc phân tầng cấp cứu theo thang phân loại lâm sàng (tương tự Manchester Triage System) được bản địa hóa theo mô hình phân khoa và mạng lưới cấp cứu tại Việt Nam (Trung tâm cấp cứu 115, phòng khám đa khoa, trạm y tế).
* Thu thập và chuẩn hóa bộ thuật ngữ triệu chứng thông qua kịch bản sinh có kiểm soát (`scripts/make_custom_dataset.py`, deterministic random seed `20261007`) kết hợp từ vựng lâm sàng thực tế và các sắc thái nhân khẩu học (trẻ sơ sinh, thai phụ, người cao tuổi có bệnh nền).

### 2.2. Quy mô dữ liệu
* **Tổng số lượng**: **300 mẫu** chất lượng cao (đạt yêu cầu $\ge 200$ mẫu của đề bài).
  * **Tập huấn luyện (`custom_train.jsonl`)**: **250 mẫu**.
  * **Tập đánh giá (`custom_eval.jsonl`)**: **50 mẫu**.
* **Phân bổ nhãn cân bằng**:
  * *Chuyên khoa*: `tim_mach` (69), `tieu_hoa` (65), `da_lieu` (61), `ho_hap` (53), `than_kinh` (52).
  * *Mức độ khẩn cấp*: `cap_cuu` (84), `khan_cap` (81), `theo_doi` (80), `tieu_chuan` (55).
  * *Nhóm đối tượng*: Phân bổ đều giữa `tre_em`, `nguoi_lon`, `nguoi_cao_tuoi`, `phu_nu_mang_thai`.
  * *Quy tắc lâm sàng*: Đối tượng nguy cơ cao (phụ nữ mang thai, trẻ nhỏ, người già bệnh nền) khi có triệu chứng vừa được nâng mức độ ưu tiên xử lý.

---

## 3. Quy trình Khử nhiễm Triệt để (Decontamination Protocol — Deck §17)

Một sai lầm nghiêm trọng trong fine-tuning là rò rỉ dữ liệu (data contamination): tập eval bị trùng lặp câu hỏi hoặc ngữ cảnh với tập train khiến điểm số cao ảo mà không phản ánh năng lực khái quát hóa.

Để đảm bảo liêm chính khoa học theo Deck §17:
1. **Phân tách kho triệu chứng độc lập (Disjoint Symptom Pools)**:
   * Tập huấn luyện sử dụng 8 nhóm triệu chứng kinh điển cho mỗi chuyên khoa (ví dụ: *"đau thắt ngực trái lan lên cằm và cánh tay"*, *"ho ra máu tươi lẫn đờm nhớt kéo dài 3 ngày"*).
   * Tập đánh giá sử dụng **5 cụm triệu chứng hoàn toàn độc lập** chưa từng xuất hiện trong tập train (ví dụ: *"đau đè nặng sau xương ức vã mồ hôi hột tay chân lạnh ngắt"*, *"thở rít co kéo cơ hõm ức thiếu oxy ngạt thở nghiêm trọng"*).
2. **Phân tách câu mở đầu & kết thúc**:
   * Tập train và eval sử dụng các tập danh từ xưng hô, ngữ cảnh thời gian và câu hỏi hướng xử lý hoàn toàn khác biệt.
3. **Kiểm tra tự động bằng code**:
   * **Trùng lặp câu nguyên văn (`exact_leak_count`)**: **0 / 50 (0%)**.
   * **Trùng lặp cụm triệu chứng eval trong train (`eval_symptom_leak`)**: **0 / 50 (0%)**.
   * Thuật toán kiểm định trong `scripts/make_custom_dataset.py` khẳng định `is_clean: True`.

---

## 4. Vì sao dữ liệu này MỚI VỀ PHÂN PHỐI so với Base Model? (Deck §3.3)

> **Cảnh báo từ Deck §3.3**: *Các base model 2026 (như Qwen3.5, Llama-3) đã được pre-train trên hàng chục nghìn tỷ token văn bản web phổ thông (Common Crawl, Wikipedia, mạng xã hội, tin tức). Fine-tune trên các câu hỏi-đáp phổ thông không mang lại giá trị gia tăng vì base model đã bão hòa miền dữ liệu đó.*

Bộ dữ liệu này thể hiện tính **mới về phân phối (distributionally new)** ở 3 khía cạnh:

1. **Khẩu ngữ triệu chứng y tế tiếng Việt không chuẩn tắc**:
   * Dữ liệu web đại trà chứa nhiều bài báo y khoa hàn lâm bằng tiếng Việt chuẩn (dạng *"Bệnh viêm phổi là gì"*). Tuy nhiên, bệnh nhân thực tế mô tả bằng khẩu ngữ cảm tính: *"mệt xỉu lên xỉu xuống"*, *"thở dốc tím tái"*, *"ợ hơi ậm ạch"*, *"mụn mủ sưng đau rải rác"*, *"rúng mình ho từng cơn"*. Base model chưa có ánh xạ xác suất có điều kiện chính xác giữa các cụm từ này với các bệnh lý đích.
2. **Quy tắc điều phối phân tầng y tế bản địa**:
   * Cấu trúc điều phối tại Việt Nam (liên hệ số khẩn cấp 115, chuyển tuyến phòng khám đa khoa, hẹn khám tư vấn online) tuân theo quy định phân cấp khám chữa bệnh Việt Nam mà dữ liệu web toàn cầu không thể suy diễn zero-shot.
3. **Cấu trúc JSON đầu ra nghiêm ngặt với 4 khóa logic ràng buộc**:
   * Base model khi nhận prompt dạng này thường trả lời bằng văn xuôi dài dòng, giải thích triệu chứng chung chung hoặc bịa đặt chẩn đoán. Việc ép trọng số adapter nội suy trực tiếp vào cấu trúc phân loại 4 trường đòi hỏi cập nhật ma trận biểu diễn LoRA, chứ không thể giải quyết bằng naive prompt.

---

## 5. Thống kê Phân tích Token & Loss Mask (NB1 Verification — Deck §17.2)

Thực hiện đo đạc bằng tokenizer chính thức của `unsloth/Qwen3.5-4B` trên môi trường hiện tại:

```json
{
  "n": 300,
  "mean": 115.0,
  "p50": 115,
  "p95": 124,
  "p99": 126,
  "max": 127,
  "suggested_max_length": 256
}
```

* **Xác định `max_length`**: Phân vị $p95 = 124$ tokens, độ dài lớn nhất $\max = 127$ tokens. Thiết lập `max_length = 256` bao phủ 100% dữ liệu mà không bị cắt cụt (truncation), đồng thời tiết kiệm đáng kể VRAM so với mức 1024 mặc định.
* **Chứng minh Loss Mask (Mask Proof)**:
  * `supervised_fraction = 0.4167` (41.7% số token nằm trong loss).
  * **Phần nằm trong loss (Supervised span)**:
    ```json
    </think>

    {"chuyen_khoa": "ho_hap", "muc_do_khan_cap": "cap_cuu", "nhom_doi_tuong": "nguoi_cao_tuoi", "huong_xu_ly": "goi_115"}<|im_end|>
    ```
  * **Phần bị che chắn (Masked span, `IGNORE_INDEX = -100`)**: Toàn bộ `system prompt`, câu hỏi mô tả triệu chứng của bệnh nhân và khối khởi tạo suy luận `<think>\n\n`.
  * Khẳng định mô hình học đúng bài toán: **chỉ tối ưu hóa xác suất sinh ra chuỗi JSON nhãn, không học vẹt lại prompt của hệ thống**.
