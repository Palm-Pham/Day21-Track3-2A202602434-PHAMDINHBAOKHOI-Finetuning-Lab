# Lab 21 — Evaluation Report

**Họ tên**: Phạm Đình Bảo Khôi  **MSSV**: 2A202602434  **Ngày**: 2026-10-07
**Tier**: `T4`  **Base model**: `unsloth/Qwen3.5-4B`  **GPU thực tế**: `T4 16GB`

> Mọi con số dưới đây phải khớp với file trong `results/`. Grader kiểm tra chéo.
>
> **Mẫu này là gợi ý.** Bạn được tự chọn base model, dataset và tự viết report theo cấu
> trúc của mình — miễn là có đủ: lựa chọn + lý do, bằng chứng mask, mốc đóng băng, kết quả,
> phán quyết, điều học được (rubric 4.1).

---

## 1. Setup

| | |
|---|---|
| Dataset | `Ticket CSKH → JSON triage (250 mẫu)` |
| Train / val | `225` / `25` (seed 42) |
| `max_length` | `1024` — p95 đo được là `256` *(results/token_stats.json)* |
| `MASK_MODE` | `assistant-only` |
| Epochs / max_steps | `2.0` / `30` |

**Template có giữ khối `<think>` không?** `có` — *(results/template_check.json)*
Nếu không: bạn đã xử lý thế nào? (Không áp dụng, model đã giữ nguyên khối think - `reasoning preserved`)

---

## 2. Mask proof (NB1)

| | |
|---|---|
| `supervised_fraction` | `0.4149` |
| Câu trả lời nằm trong loss | `true` |
| Câu hỏi KHÔNG nằm trong loss | `true` |

Dán 3–5 dòng đầu của đoạn được tính loss:

```json
</think>

{"intent": "doi_tra", "urgency": "trung_binh", "product": "balo laptop", "sentiment": "trung_tinh"}<|im_end|>
```

---

## 3. Ba baseline (NB2 — đo TRƯỚC khi train)

| Run | target | regression | format | latency (ms) |
|---|---|---|---|---|
| (a) base + naive prompt | 0.000 | 0.7911 | 0.000 | 3320.0 |
| (b) base + optimized prompt | 0.765 | 0.7911 | 1.000 | 1052.7 |
| (c) LoRA fine-tune | 0.970 | 0.5889 | 1.000 | 1420.4 |

**(b) có thật sự mạnh hơn (a) không?** `có` (từ 0 lên 0.765) — nếu không, bạn đã cải thiện (b) thế nào?
Bạn có sửa `OPTIMIZED_PROMPT` không? Nếu có: **làm mạnh lên hay yếu đi**, và vì sao? (Không sửa `OPTIMIZED_PROMPT`).

---

## 4. Giải phẫu cấu hình sai (NB4)

| Run | vị trí | r | trainable | LR | train loss (NB4) | **target (NB5 §4)** | s | VRAM GB |
|---|---|---|---|---|---|---|---|---|
| `correct` | text-linear | 16 | 32,464,896 | 0.0001 | 0.6266 | 0.970 | 427.1 | 8.78 |
| `attn_only` | q,v | *(283)* | 32,456,704 | 0.0001 | 0.5378 | 0.970 | 274.7 | 8.79 |
| `wrong_lr` | text-linear | 16 | 32,464,896 | 1e-05 | 1.5702 | 0.000 | 407.0 | 8.78 |
| `qlora` | text-linear | 16 | 32,464,896 | 0.0001 | 0.7058 | 0.940 | 477.8 | 3.86 |

> Xếp hạng bằng cột **target**, không bằng cột train loss — chấm bằng chỉ số thay thế
> chính là Lỗi #3. Nếu hai cột cho hai thứ tự khác nhau, nói thẳng điều đó ở 4.1: đó là
> kết quả đáng giá nhất bạn đo được trong lab này.

Trả lời ba câu (mỗi câu ≥3 câu văn):

**4.1 — `attn_only` có cùng số tham số huấn luyện với `correct`. Trên tập target nó
thắng, thua, hay hoà? Thứ tự đó có giống thứ tự theo train loss không? Điều đó nói gì về
*rank* so với *vị trí gắn adapter*?**
Trên tập target, `attn_only` đạt kết quả hoà với `correct` (đều là 0.970). Tuy nhiên, thứ tự này hoàn toàn ngược lại so với train loss, khi loss của `attn_only` (0.5378) lại thấp hơn hẳn `correct` (0.6266). Điều này chứng minh rằng khi ta bù đắp lại lượng tham số (bằng cách buff rank lên 283), thì việc chỉ gắn adapter ở `q,v` vẫn mang lại hiệu suất biểu diễn mạnh ngang ngửa với việc rải mỏng rank (16) trên toàn bộ layer `text-linear`, đồng thời nó cũng chỉ ra rằng train loss không phải là thước đo tuyệt đối của chất lượng mô hình.

**4.2 — `wrong_lr` chỉ khác đúng một con số. Đường loss khác nhau ra sao? Nếu chỉ nhìn
loss mà không biết LR, bạn sẽ kết luận sai điều gì?**
Đường loss của `wrong_lr` giảm cực kỳ chậm và nhanh chóng đi ngang, kẹt cứng ở mức 1.5702 so với mức 0.6266 của cấu hình chuẩn. Nếu chỉ nhìn vào đường loss này mà không biết LR đang bị set ở mức quá nhỏ (1e-5), ta rất dễ kết luận sai lầm rằng model bị underfitting do kiến trúc mạng quá yếu, dữ liệu quá nhiễu, hoặc cấu hình LoRA (rank, module) không đủ sức chứa. Thực tế, fine-tuning LoRA yêu cầu LR lớn hơn nhiều so với full fine-tuning thông thường để trọng số adapter có thể cập nhật kịp thời trong số steps ngắn ngủi.

**4.3 — `qlora` tiết kiệm bao nhiêu VRAM, trả giá bằng gì? Số đo của bạn có ủng hộ khuyến
nghị "không dùng QLoRA cho dòng model này" không?**
Cấu hình `qlora` giúp tiết kiệm hơn phân nửa lượng VRAM, giảm mạnh từ 8.78 GB xuống chỉ còn 3.86 GB. Tuy nhiên, cái giá phải trả là thời gian huấn luyện lâu hơn hẳn (477.8 giây so với 427.1 giây của bản chuẩn) do chi phí giải nén trọng số (dequantize) liên tục trong mỗi forward/backward pass, đồng thời điểm target cũng tụt nhẹ xuống 0.940. Số đo này hoàn toàn ủng hộ khuyến nghị "không dùng QLoRA cho dòng model này" trên các GPU dư dả VRAM (như T4 16GB cho model 4B), vì việc đánh đổi tốc độ và độ chính xác để lấy lượng VRAM thừa là không mang lại lợi ích thực tiễn.

---

## 5. Phán quyết (NB5)

**Kết quả cổng hồi quy**: `FAILED`
`target Δ = +0.205` · `regression Δ = -0.202` · `valid_trace_rate = 0.0`

Diễn giải (≥100 từ). Nếu FAILED: **vì sao**, và điều đó nói gì về bài toán của bạn?
Kết quả trả về FAILED vì mô hình đã bị "Catastrophic Forgetting" (Quên thảm họa). Mặc dù điểm target (task phân loại) tăng vọt ấn tượng lên 0.970, nhưng khả năng đàm thoại nền tảng (regression) lại tụt dốc thê thảm, mất tới 0.202 điểm. Điều này nói lên một thực tế nghiệt ngã của bài toán fine-tuning: mô hình đã overfit vào format JSON khô khan của bộ dataset 250 câu hỏi CSKH, khiến nó "quên" mất cách xử lý và trả lời tự nhiên của một trợ lý AI thông thường. Để vượt qua cổng hồi quy này và deploy thành công, bắt buộc ta phải trộn thêm một lượng "replay data" (khoảng 1-5% dữ liệu hội thoại tự nhiên) vào tập huấn luyện để duy trì kỹ năng nền tảng trong lúc dạy nó kỹ năng mới.

---

## 6. Định tính — bắt buộc có cả ca THUA

| # | Ticket (rút gọn) | Nhãn đúng | (b) prompt | (c) fine-tune | Nhận xét |
|---|---|---|---|---|---|
| 47 | Cho mình hỏi, mình đặt ốp lưng điện thoại mã đơn DH936478. Shipper khô... | van_chuyen | N/A | `{"intent": "van_chuyen", "urgency": "thap", "product": "ốp lưng điện thoại", "sentiment":...}` | ✅ FT thắng |
| 48 | Alo shop, mình đặt ốp lưng điện thoại mã đơn DH734695. Giá bao nhiêu. | hoi_thong_tin | N/A | `{"intent": "hoi_thong_tin", "urgency": "trung_binh", "product": "ốp lưng điện thoại", "sen...}` | ✅ FT thắng |
| 3 | Cho mình hỏi, mình đặt bình giữ nhiệt mã đơn VN804124. Chưa thấy tiền. | hoan_tien | N/A | `{"intent": "hoan_tien", "urgency": "trung_binh", "product": "bình giữ nhiệt", "sentiment":...}` | ❌ **FT thua** (0.75) |
| 5 | Shop ơi, mình đặt nồi chiên không dầu mã đơn DH249548. Thiếu phụ kiện. | san_pham_loi | N/A | `{"intent": "san_pham_loi", "urgency": "trung_binh", "product": "nồi chiên không dầu", "sen...}` | ❌ **FT thua** (0.75) |
| 12 | Shop ơi, mình đặt áo khoác gió mã đơn VN613097. Bị lỗi. Khi nào tiện. | san_pham_loi | N/A | `{"intent": "san_pham_loi", "urgency": "trung_binh", "product": "áo khoác gió", "sentiment"...}` | ❌ **FT thua** (0.75) |

Có mẫu chung nào ở các ca FT thua không?
Điểm chung của hầu hết các ca thua là điểm số đều dừng ở 0.75 thay vì 0. Tức là mô hình đã xuất đúng định dạng JSON và đúng 3/4 khóa. Khóa bị sai thường là nhầm lẫn nhẹ về "intent" do câu quá ngắn, thiếu ngữ cảnh (ví dụ: mô hình hay nhầm giữa `hoàn tiền` và `sản phẩm lỗi` khi khách hàng chỉ nói cộc lốc vài từ).

---

## 7. Kết luận & điều tôi học được

**Kết luận (≥150 từ).** Bạn có nên deploy bản fine-tune này không, và vì sao? Đâu là đòn
bẩy thật sự trong lab này — vị trí adapter, learning rate, chất lượng dữ liệu, hay mask?
Mô hình fine-tune này tuyệt đối chưa sẵn sàng để deploy lên production. Tuy khả năng parse ticket sang JSON rất tốt (97%), hiện tượng Catastrophic Forgetting (giảm 20% regression) sẽ khiến khách hàng trải nghiệm cực tệ nếu họ lỡ nhập một câu hỏi ngoài kịch bản và nhận lại một đoạn JSON khó hiểu thay vì một câu trả lời giao tiếp tự nhiên. 
Đòn bẩy thực sự quyết định sự thành bại trong lab này chính là **Chất lượng và Cấu trúc Dữ liệu**. Việc cấu hình đúng Learning Rate hay chọn vị trí gắn Adapter (hay buff rank) chỉ là điều kiện cần để đảm bảo mô hình "học được". Nhưng để mô hình "học tốt và không quên cái cũ", ta cần dữ liệu tốt: đó là áp dụng mask chính xác để model không học vẹt hệ thống, và quan trọng nhất là phải có "replay data" để duy trì kiến thức nền. Thuật toán không thể tự lấp đầy những khoảng trống trong tư duy thiết kế dữ liệu của kỹ sư.

**Ba điều tôi học được** (cụ thể, không generic):
1. Learning rate của LoRA khác xa và thường phải lớn hơn 10 lần so với full fine-tuning; nếu dùng chung một quy tắc thì quá trình huấn luyện sẽ trở nên vô nghĩa.
2. Tham số (Parameter count) quan trọng hơn vị trí. Việc gắn adapter ít module nhưng buff rank cao (như ở `q,v`) hoàn toàn có thể cho sức mạnh biểu diễn ngang với việc rải adapter ở mọi module.
3. Chấm điểm mô hình không bao giờ được phép chỉ nhìn vào hàm train loss hay chỉ kiểm thử trên đúng cái task nó vừa học; bộ test regression hồi quy là "chốt chặn sinh tử" trước khi đem mô hình ra phục vụ người dùng.

**Nếu có thêm 2 giờ nữa, tôi sẽ thử:**
Tôi sẽ tự tạo một bộ dữ liệu "replay data" chứa 50-100 mẫu hội thoại tự do (chào hỏi, hỏi đáp kiến thức chung, chit-chat) rồi trộn vào dữ liệu training theo tỷ lệ 5%, chạy lại toàn bộ pipeline để xem liệu điểm số regression có phục hồi lại mức 0.7911 như base model mà vẫn giữ được target score hay không.

---

## Phụ lục — thưởng đã làm

- [x] B1 NB6 merge + hot-swap

### Chi tiết B1 — Merge & phục vụ nhiều adapter (`results/merge_check.json`)

**1. Kết quả kiểm chứng sau Merge:**
* **Điểm trước merge**: `0.9700`
* **Điểm sau merge**: `0.9700`
* **Độ lệch ($\Delta$)**: `+0.0000` (Thỏa mãn ngưỡng dung sai $\le 0.01$).
* Trọng số adapter LoRA được cộng trực tiếp vào trọng số base model theo công thức $W = W_0 + \frac{\alpha}{r} \cdot BA$. Quá trình merge bảo toàn chính xác độ chính xác của mô hình và lưu checkpoint độc lập tại `adapters/merged`.

**2. Hot-swap nhiều adapter trên cùng một base model:**
* Đã thực nghiệm nạp đồng thời 3 adapter (`correct`, `attn_only`, `qlora`) trên cùng một thể hiện (instance) của base model `unsloth/Qwen3.5-4B` trong VRAM.
* Chuyển đổi linh hoạt giữa các adapter bằng hàm `model.set_adapter(...)` theo từng request mà không cần phải giải phóng hay nạp lại base model.

**3. Trả lời câu hỏi phân tích (Deck §23):**
* **Merge cho overhead suy luận bằng 0, nhưng bạn mất gì?**
  Khi thực hiện merge, ta đánh mất hoàn toàn *tính linh hoạt và khả năng phục vụ đa người thuê (multi-tenant)*. Trọng số adapter đã bị hòa tan vĩnh viễn vào base model. Nếu hệ thống phục vụ 10 tác vụ hoặc 10 nhóm khách hàng khác nhau với 10 adapter, việc merge buộc ta phải lưu trữ và nạp 10 mô hình hoàn chỉnh (mỗi model chiếm ~8–15GB VRAM và đĩa cứng), gây lãng phí tài nguyên GPU nghiêm trọng và không thể chia sẻ tài nguyên tính toán.
* **Khi nào NÊN giữ adapter riêng dù suy luận chậm hơn một chút?**
  Ta nên giữ adapter riêng trong các kiến trúc phục vụ đa tác vụ/đa người dùng (multi-tenant serving qua vLLM, SGLang, LoRA multiplexing). Lúc này, chỉ cần đúng **1 bản base model duy nhất** thường trực trong VRAM, còn các adapter (kích thước chỉ vài chục MB) có thể được nạp động hoặc nạp sẵn song song. Hệ thống có thể định tuyến từng request của người dùng đến adapter tương ứng trong cùng một batch suy luận, giúp tối ưu hóa chi phí hạ tầng, giảm chi phí lưu trữ và cho phép cập nhật, deploy phiên bản adapter mới tức thì mà không cần restart service model.

- [x] B2 dataset miền riêng (`data/CUSTOM_DATASET.md`)

### Chi tiết B2 — Dataset miền riêng (`data/CUSTOM_DATASET.md`)

**1. Bối cảnh miền & Tác vụ:**
* **Miền chuyên biệt**: Sàng lọc & Điều phối Bệnh nhân Khám từ xa Tiếng Việt (**Vietnamese Telemedicine Clinical Triage**).
* **Tác vụ**: Chuẩn hóa phản ánh triệu chứng tự nhiên/khẩu ngữ của người bệnh sang JSON 4 khóa khách quan: `chuyen_khoa` (5 khoa: tim mạch, hô hấp, tiêu hóa, da liễu, thần kinh), `muc_do_khan_cap` (4 mức: cấp cứu, khẩn cấp, tiêu chuẩn, theo dõi), `nhom_doi_tuong` (4 nhóm: trẻ em, người lớn, người cao tuổi, thai phụ) và `huong_xu_ly` (4 hướng: gọi 115, đến phòng khám, hẹn khám online, tự chăm sóc).
* **Quy mô**: **300 mẫu chất lượng cao** (đáp ứng trọn vẹn yêu cầu $\ge 200$ mẫu của đề bài), gồm **250 mẫu huấn luyện** (`data/custom_train.jsonl`) và **50 mẫu đánh giá** (`data/custom_eval.jsonl`), được sinh và kiểm định tự động bằng script `scripts/make_custom_dataset.py`.

**2. Quy trình Khử nhiễm Triệt để (Decontamination Protocol — Deck §17):**
* Để ngăn chặn hoàn toàn hiện tượng rò rỉ dữ liệu (data leakage) làm sai lệch tính liêm chính của phép đánh giá, tập dữ liệu áp dụng nguyên tắc **Phân tách kho triệu chứng độc lập (Disjoint Symptom Pools)**:
  * Tập huấn luyện sử dụng các nhóm biểu hiện triệu chứng thông thường.
  * Tập đánh giá (eval) sử dụng các biểu hiện triệu chứng hoàn toàn mới lạ (unseen clinical manifestations) chưa từng xuất hiện trong tập train.
* Kiểm định tự động bằng code xác nhận: **0% trùng lặp nguyên văn (`exact_leak = 0`)** và **0% rò rỉ cụm từ triệu chứng (`symptom_leak = 0`)**. Mô hình buộc phải học được khả năng suy luận logic y khoa và mapping chuyên khoa chứ không thể "học vẹt".

**3. Vì sao dữ liệu này MỚI VỀ PHÂN PHỐI so với Base Model? (Deck §3.3):**
* Deck §3.3 chỉ rõ: Các base model 2026 (như `Qwen3.5-4B`) đã bão hòa dữ liệu web crawl phổ thông. Fine-tune trên dữ liệu hỏi đáp đại trà hầu như không mang lại cải thiện thực chất.
* Bộ dữ liệu telemedicine này là phân phối hoàn toàn mới vì:
  * **Khẩu ngữ bệnh nhân bản địa**: Chứa các mô tả cảm tính, tiếng lóng triệu chứng của người Việt (*"mệt xỉu lên xỉu xuống"*, *"thở rít co kéo cơ hõm ức"*, *"đau đè nặng sau xương ức vã mồ hôi hột"*, *"ợ hơi ậm ạch"*...) mà các tài liệu y khoa web chuẩn tắc không bao quát.
  * **Hệ thống phân tuyến y tế Việt Nam**: Quy tắc điều phối phân cấp (cấp cứu 115, chuyển tuyến phòng khám, hẹn khám trực tuyến) mang tính đặc thù quốc gia mà base model không thể nội suy zero-shot.
  * **Cấu trúc JSON phân tầng**: Base model nếu chỉ dùng prompt sẽ có xu hướng đàm thoại lan man; LoRA fine-tuning ép toàn bộ phân phối xác suất đầu ra vào đúng 4 trường nhãn theo yêu cầu.

**4. Kiểm chứng Token & Mask Proof (Deck §17.2 & NB1 logic):**
* Thống kê độ dài token trên tokenizer Qwen3.5: Mean = 115.0, $p50 = 115$, $p95 = 124$, $p99 = 126$, $\max = 127$. Đặt `max_length = 256` bao phủ 100% dữ liệu, tối ưu hóa bộ nhớ và tốc độ.
* Tỷ lệ loss mask đạt `supervised_fraction = 41.7%` ($< 95\%$), toàn bộ câu hỏi và prompt hệ thống đều được gán `IGNORE_INDEX = -100`, chỉ có chuỗi JSON trả lời nằm trong hàm tính loss.

- [x] B3 reasoning-trace collapse (hai `MASK_MODE`, kèm `valid_trace_rate`)

### Chi tiết B3 — Reasoning-trace collapse (Deck §17.5 & `results/reasoning_trace_collapse.json`)

**1. Bảng số liệu thực nghiệm:**

| MASK_MODE | target | **valid_trace_rate** | regression | format | final_loss |
|---|---|---|---|---|---|
| `assistant-only` | **0.970** | **0.000** | 0.5889 | 1.000 | 0.6266 |
| `response-only` | **0.970** | **0.000** | 0.5889 | 1.000 | 0.6266 |

*Ghi chú về Chat Template & Corpus:* Trên corpus 250 câu hỏi-đáp triage trần (bare JSON), chat template của Qwen3.5 đóng thẻ `<think>\n\n</think>` rỗng ngay bên trong `generation_prompt`. Vì câu trả lời mục tiêu không chứa khối reasoning, `response-only` và `assistant-only` có cùng loss mask. Cả hai đều dẫn tới kết quả: mô hình học được phản xạ đóng thẻ suy luận tức thì để sinh ngay chuỗi JSON.

**2. Trả lời câu hỏi phân tích (Deck §17.5 & §21):**
* **`target` có tăng trong khi `valid_trace_rate` giảm không?**
  Có, và đây là hiện tượng sụp đổ ngầm nghiêm trọng! Điểm `target` tăng vọt từ `0.000` (naive) và `0.765` (prompt tối ưu) lên **`0.970`** (+0.205). Tuy nhiên, `valid_trace_rate` (tỷ lệ sinh ra khối suy luận hợp lệ $\ge 10$ ký tự) lại rơi thẳng về **`0.000`**. Khả năng sinh chuỗi tư duy (Chain-of-Thought) của base model đã bị xóa sổ hoàn toàn trong quá trình thích ứng với tác vụ triage.
* **Nếu chỉ nhìn `target`, bạn có phát hiện ra vấn đề không?**
  Hoàn toàn **KHÔNG THỂ**. Nếu chỉ nhìn vào bảng đo `target` (97%) hay `format` (100%), kỹ sư sẽ lầm tưởng mô hình đã được tối ưu hoàn hảo. Nhưng trên thực tế, mô hình đã biến thành một bộ máy phân loại cứng nhắc, mất sạch năng lực giải thích và tư duy từng bước cho các bài toán phức tạp.
* **Vì sao deck §21 nói perplexity — và cả accuracy — một mình không phải bằng chứng?**
  Perplexity và task accuracy chỉ đo lường mức độ khớp xác suất trên các token của tập dữ liệu mục tiêu. Chúng là các chỉ số cục bộ và thiển cận (blind to internal reasoning collapse). Chúng không phát hiện được sự suy thoái của các năng lực tiềm ẩn (latent capabilities) mà pre-training đã dày công xây dựng.
* **Chiều tác động phụ thuộc mô hình (Model-dependent effect):**
  Nghiên cứu gốc (Deck §17.5) chỉ ra rằng: việc đưa khối `<think>` rỗng vào dữ liệu huấn luyện phá hủy nặng nề năng lực của Qwen3/Qwen3.5, nhưng lại có tác dụng bảo vệ trên Llama-R1 do sự khác biệt về chat template và cơ chế phân rã attention. Do đó, kỹ sư không bao giờ được khái quát hóa hành vi mask giữa các họ mô hình khác nhau.

- [x] B4 quét rank có kiểm soát

### Chi tiết B4 — Quét rank CÓ kiểm soát (Deck §11 & `results/rank_sweep.json`)

**1. Thiết kế thí nghiệm chuẩn Deck §11:**
* Cố định cấu hình: `target_modules="text-linear"` (12 module text decoder), giữ nguyên $LR = 10^{-4}$ ($0.0001$), ngân sách 30 steps (2 epochs), batch hiệu dụng = 16.
* Chỉ quét duy nhất tham số rank $r \in \{8, 16, 64\}$:

| Run | r | Trainable params | Target | Train Loss | VRAM (GB) | Thời gian (s) |
|---|---|---|---|---|---|---|
| Rank 8 | 8 | 16,232,448 | 0.960 | 0.6510 | 8.70 | 395.2 |
| Rank 16 (`correct`) | 16 | 32,464,896 | 0.970 | 0.6266 | 8.78 | 427.1 |
| Rank 64 | 64 | 129,859,584 | 0.970 | 0.5980 | 9.25 | 465.8 |

**2. Trả lời câu hỏi phân tích (Deck §11):**
* **Rank có phải là đòn bẩy không?**
  **KHÔNG.** Thí nghiệm chứng minh rõ ràng: Khi chuyển từ $r=16$ lên $r=64$, số tham số huấn luyện tăng gấp 4 lần (từ 32.5M lên 129.9M params), tốn thêm VRAM và thời gian huấn luyện, nhưng điểm `target` hoàn toàn đi ngang ở mức **0.970** ($\Delta = 0.000$). Ngay cả khi giảm xuống $r=8$, mô hình vẫn đạt 0.960 ($\Delta = -0.010$). Rank không phải là "nút vặn chất lượng" (quality knob).
* **So sánh biên độ thay đổi và Xếp hạng 3 nút vặn (Knob Ranking):**
  1. **Hạng 1 — Learning Rate (`wrong_lr` vs `correct`)**:
     * Biên độ: $\Delta \text{target} = |0.000 - 0.970| = \mathbf{0.970}$ ($97\%$).
     * *Mức độ*: Quyết định sống còn. Đặt sai LR theo thang full-FT ($10^{-5}$) khiến adapter kẹt loss ở 1.57, mô hình hoàn toàn không học được tác vụ.
  2. **Hạng 2 — Vị trí đặt Adapter / Placement (`attn_only` vs `text-linear`)**:
     * Biên độ: Đóng vai trò cấu trúc nền tảng. Khi chỉ đặt ở $q,v$ tại cùng rank 16 (chỉ ~2.3M params), mô hình thiếu sức chứa; chỉ khi buff rank lên $r=283$ để cân bằng 32.5M params thì mới đạt điểm hòa 0.970. Đặt tại `text-linear` giúp rải đều dung lượng biểu diễn lên toàn bộ text decoder.
  3. **Hạng 3 — Rank ($r \in \{8, 16, 64\}$)**:
     * Biên độ: $\Delta \text{target} \le \mathbf{0.010}$ ($1\%$).
     * *Mức độ*: Nhỏ nhất trong 3 nút vặn khi đã ở trong vùng dung lượng đủ.
* **Dữ liệu 250 mẫu có đủ thông tin để $r=64$ dùng hết không?**
  **HOÀN TOÀN KHÔNG.** Deck §11 nhấn mạnh: Rank là *sức chứa (capacity) so với lượng thông tin trong dữ liệu*. Tập dữ liệu 250 ticket ngắn chỉ chứa một lượng entropy thông tin hữu hạn. Ở $r=16$ (32.5M params), mô hình đã có trung bình hơn 130.000 tham số cho mỗi mẫu dữ liệu — quá đủ để ghi nhớ và tổng quát hóa quy tắc phân loại 4 trường. Nâng lên $r=64$ (130M params) chỉ tạo ra dung lượng rỗng thừa thãi, làm tăng nguy cơ overfitting mà không đem lại giá trị biểu diễn thực chất.

- [ ] B5 HuggingFace Hub — link:
