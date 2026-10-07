#!/usr/bin/env python3
"""Generate the custom domain dataset for Bonus Challenge B2 (Lab 21).

Domain: **Vietnamese Telemedicine Clinical Triage (Sàng lọc Y tế Từ xa Tiếng Việt)**.
Task: Patient colloquial symptom complaint -> Structured 4-field clinical triage JSON:
  * chuyen_khoa     -> tim_mach | ho_hap | tieu_hoa | da_lieu | than_kinh
  * muc_do_khan_cap -> cap_cuu | khan_cap | tieu_chuan | theo_doi
  * nhom_doi_tuong  -> tre_em | nguoi_lon | nguoi_cao_tuoi | phu_nu_mang_thai
  * huong_xu_ly     -> goi_115 | den_phong_kham | hen_kham_online | tu_cham_soc

Meets all Deck §17 (Data Curation & Decontamination) and Deck §3.3 (Distributional Novelty) requirements.
Run: python scripts/make_custom_dataset.py
"""
from __future__ import annotations

import json
import pathlib
import random
from typing import Dict, List, Set, Tuple

ROOT = pathlib.Path(__file__).resolve().parents[1]
DATA = ROOT / "data"

SEED = 20261007

# 1. Specialized clinical taxonomy partitioned strictly into TRAIN and EVAL pools
# to guarantee absolute decontamination and evaluate real generalization (Deck §17).

SPECIALTY_DATA = {
    "tim_mach": {
        "train_symptoms": [
            ("đau thắt ngực trái lan lên cằm và cánh tay", "cap_cuu"),
            ("nhói buốt vùng ngực trái kèm toát mồ hôi lạnh", "cap_cuu"),
            ("huyết áp tăng vọt 180/100 kèm đau tức đỉnh đầu", "cap_cuu"),
            ("choáng váng tụt huyết áp ngất lịm thoáng qua", "khan_cap"),
            ("khó thở khi nằm phẳng phải kê gối cao để thở", "khan_cap"),
            ("tim đập nhanh dồn dập hồi hộp trống ngực liên hồi", "tieu_chuan"),
            ("phù hai bàn chân nặng nề kèm mệt mỏi kiệt sức", "tieu_chuan"),
            ("đánh trống ngực kèm cảm giác hụt hơi lồng ngực", "theo_doi"),
        ],
        "eval_symptoms": [
            ("đau đè nặng sau xương ức vã mồ hôi hột tay chân lạnh ngắt", "cap_cuu"),
            ("huyết áp tụt kẹp 80/50 người lả đi mạch đập yếu ớt", "cap_cuu"),
            ("thở dốc tím tái khi leo cầu thang kèm sưng to mắt cá chân", "khan_cap"),
            ("nhịp tim không đều lúc nhanh lúc chậm hồi hộp ngực", "tieu_chuan"),
            ("thỉnh thoảng nhói nhẹ ngực trái khi hít sâu rồi tự hết", "theo_doi"),
        ],
    },
    "ho_hap": {
        "train_symptoms": [
            ("khó thở rít từng cơn tím tái môi đầu chi", "cap_cuu"),
            ("ho ra máu tươi lẫn đờm nhớt kéo dài 3 ngày", "cap_cuu"),
            ("sốt cao 39 độ khó thở đau tức hai bên sườn khi hít sâu", "khan_cap"),
            ("thở khò khè nhiều đờm đặc màu vàng xanh", "khan_cap"),
            ("khó thở khi gắng sức leo cầu thang thở dốc", "tieu_chuan"),
            ("ho khan kéo dài ngứa rát cổ họng khàn tiếng", "tieu_chuan"),
            ("ho có đờm trắng đục sau đợt cảm lạnh nhẹ", "theo_doi"),
            ("ngạt mũi chảy nước mũi hắt hơi liên tục do thời tiết", "theo_doi"),
        ],
        "eval_symptoms": [
            ("thở rít co kéo cơ hõm ức thiếu oxy ngạt thở nghiêm trọng", "cap_cuu"),
            ("ho khạc ra dịch lẫn tia máu kèm sụt cân sốt về chiều", "khan_cap"),
            ("tiếng thở rít ran rít nhiều đờm đục thở nông gấp gáp", "khan_cap"),
            ("rát họng rúng mình ho từng cơn kéo dài trên 2 tuần", "tieu_chuan"),
            ("chảy nước mũi trong nghẹt mũi nhẹ không sốt", "theo_doi"),
        ],
    },
    "tieu_hoa": {
        "train_symptoms": [
            ("nôn ra máu đỏ tươi đi ngoài phân đen mùi tanh khắm", "cap_cuu"),
            ("đau quặn bụng dữ dội vùng hố chậu phải sốt nhẹ nôn ói", "cap_cuu"),
            ("tiêu chảy xối xả liên tục mất nước kiệt sức mắt trũng", "khan_cap"),
            ("nôn ói nhiều không giữ được nước kèm khô miệng", "khan_cap"),
            ("đau rát bỏng vùng thượng vị sau ức ợ chua cồn cào", "tieu_chuan"),
            ("đau âm ỉ quanh rốn kèm rối loạn phân lỏng sệt", "tieu_chuan"),
            ("chướng bụng đầy hơi khó tiêu sau khi ăn đồ dầu mỡ", "theo_doi"),
            ("ợ hơi ậm ạch táo bón lâu ngày muốn tư vấn men vi sinh", "theo_doi"),
        ],
        "eval_symptoms": [
            ("bụng cứng như gỗ đau chói khắp bụng huyết áp tụt", "cap_cuu"),
            ("đau quặn thắt hạ sườn phải lan ra sau lưng sốt rét run", "cap_cuu"),
            ("đi ngoài phân lỏng toé nước trên 8 lần một ngày đuối sức", "khan_cap"),
            ("nóng rát dạ dày cồn cào buồn nôn sau khi ăn no", "tieu_chuan"),
            ("bụng ậm ạch sôi bụng đi ngoài phân sống sau ăn đồ lạ", "theo_doi"),
        ],
    },
    "da_lieu": {
        "train_symptoms": [
            ("phù nề phù mạch mi mắt môi khó thở kèm mày đay toàn thân", "cap_cuu"),
            ("nổi bóng nước lở loét diện rộng toàn thân trợt da đau rát", "khan_cap"),
            ("phát ban đỏ ngứa ngáy nhiều sau khi ăn hải sản lạ", "khan_cap"),
            ("nổi mẩn đỏ ngứa rát nhiều ở nếp gấp tay chân có mụn nước", "tieu_chuan"),
            ("mụn mủ sưng đau nổi rải rác vùng mặt và lưng", "tieu_chuan"),
            ("da khô nứt nẻ tróc vảy mẩn ngứa về đêm vùng cẳng tay", "theo_doi"),
            ("nổi mề đay cục bộ ngứa râm ran vài nốt ở bắp đùi", "theo_doi"),
            ("dát sạm màu nâu nhẹ hai bên gò má không ngứa", "theo_doi"),
        ],
        "eval_symptoms": [
            ("phản vệ sưng vù mặt họng nghẹn thở nổi mề đay khắp người", "cap_cuu"),
            ("bong tróc da hoại tử lở rát miệng sau khi uống thuốc kháng sinh", "cap_cuu"),
            ("sẩn ngứa mụn nước mọc thành chùm rát bỏng một bên sườn", "khan_cap"),
            ("vảy nến tróc phấn đỏ da từng mảng ở cùi chỏ và đầu gối", "tieu_chuan"),
            ("ngứa ngáy râm ran sau khi tắm nước nóng da hơi khô ráp", "theo_doi"),
        ],
    },
    "than_kinh": {
        "train_symptoms": [
            ("méo miệng liệt nửa người đột ngột nói ngọng mất thăng bằng", "cap_cuu"),
            ("co giật toàn thân sùi bọt mép mê man mất ý thức", "cap_cuu"),
            ("đau đầu dữ dội như sét đánh kèm cứng gáy nôn vọt", "cap_cuu"),
            ("chóng mặt quay cuồng nhà cửa nghiêng ngả buồn nôn khi đổi tư thế", "khan_cap"),
            ("đau nửa đầu âm ỉ theo nhịp đập sợ ánh sáng và tiếng ồn", "tieu_chuan"),
            ("tê bì châm chích râm ran hai bàn tay bàn chân kéo dài", "tieu_chuan"),
            ("mất ngủ kinh niên khó vào giấc mệt mỏi uể oải ban ngày", "theo_doi"),
            ("đau mỏi vai gáy lan xuống bả vai do ngồi văn phòng nhiều", "theo_doi"),
        ],
        "eval_symptoms": [
            ("mờ đột ngột một mắt liệt tay chân không nhấc lên được", "cap_cuu"),
            ("cơn vắng ý thức gọi không đáp mắt nhìn chằm chằm", "khan_cap"),
            ("đau giật nhói nửa đầu theo mạch đập kèm nôn nao sợ tiếng động", "tieu_chuan"),
            ("cảm giác như kiến bò tê rần các đầu ngón tay ngón chân", "tieu_chuan"),
            ("nặng đầu căng thẳng mất ngủ trằn trọc 3 hôm nay", "theo_doi"),
        ],
    },
}

TRAIN_DEMOGRAPHICS = {
    "tre_em": [
        "Bé nhà em 4 tuổi", "Cháu nhà tôi 2 tuổi", "Con gái em 6 tuổi",
        "Bé trai sơ sinh 9 tháng tuổi", "Con tôi 5 tuổi", "Cháu nhỏ 3 tuổi",
    ],
    "nguoi_lon": [
        "Tôi năm nay 32 tuổi", "Em 28 tuổi làm nhân viên văn phòng", "Mình là nam 35 tuổi",
        "Bản thân em 25 tuổi", "Tôi 40 tuổi kỹ sư xây dựng", "Em là nữ 30 tuổi",
    ],
    "nguoi_cao_tuoi": [
        "Bà nội tôi 78 tuổi có tiền sử huyết áp", "Bố tôi 68 tuổi tiền sử tiểu đường",
        "Mẹ em 65 tuổi tim mạch mãn tính", "Ông ngoại em 82 tuổi",
        "Bác gái tôi 70 tuổi thể trạng yếu", "Cụ bà nhà mình 75 tuổi",
    ],
    "phu_nu_mang_thai": [
        "Em đang mang thai tuần thứ 18", "Vợ tôi đang có bầu tháng thứ 6",
        "Em bầu tập đầu tuần 24", "Mình đang có thai 32 tuần",
        "Em gái tôi đang mang bầu 3 tháng đầu", "Vợ em bầu 12 tuần",
    ],
}

EVAL_DEMOGRAPHICS = {
    "tre_em": [
        "Bé con nhà mình 15 tháng tuổi", "Cháu bé 7 tuổi học lớp hai",
        "Con trai nhỏ của tôi 3 tuổi rưỡi", "Bé gái 18 tháng",
    ],
    "nguoi_lon": [
        "Tôi là nam giới 29 tuổi", "Mình là nữ 45 tuổi nội trợ",
        "Em 23 tuổi sinh viên mới ra trường", "Tôi 38 tuổi làm công nhân",
    ],
    "nguoi_cao_tuoi": [
        "Ông nội tôi năm nay 85 tuổi", "Mẹ tôi 72 tuổi có bệnh tim",
        "Bố em 66 tuổi từng đặt stent", "Bác trai tôi 74 tuổi huyết áp cao",
    ],
    "phu_nu_mang_thai": [
        "Thai phụ 27 tuổi mang thai 20 tuần", "Mình đang mang bầu tuần thứ 30",
        "Vợ tôi có thai con so được 16 tuần", "Em đang có thai quý hai tuần 22",
    ],
}

TRAIN_OPENERS = [
    "Bác sĩ ơi tư vấn giúp em,",
    "Chào bác sĩ, xin xem giúp trường hợp này,",
    "Alo tổng đài khám bệnh từ xa,",
    "Nhờ bác sĩ trực hỗ trợ gấp,",
    "Xin hỏi bác sĩ trực tuyến,",
    "Em cần tư vấn y tế khẩn cấp,",
]

EVAL_OPENERS = [
    "Kính chào bác sĩ trực,",
    "Dạ em gửi câu hỏi tư vấn bác sĩ,",
    "Xin chào đội ngũ y tế trực tuyến,",
    "Nhờ bác sĩ phòng khám từ xa chẩn đoán sơ bộ,",
    "Bác sĩ tư vấn khẩn cấp giúp gia đình em,",
]

TRAIN_TIMES = [
    "khởi phát đột ngột khoảng 30 phút nay",
    "kéo dài liên tục từ sáng đến giờ",
    "bị từ đêm qua đến nay không thuyên giảm",
    "bị âm ỉ tái đi tái lại cả tuần nay",
    "mới xuất hiện cách đây 2 tiếng",
    "diễn tiến ngày càng nặng hơn",
    "bị vài ngày nay uống thuốc hạ sốt không đỡ",
]

EVAL_TIMES = [
    "mới bắt đầu xuất hiện dồn dập độ 45 phút trước",
    "tình trạng phát tác dữ dội từ rạng sáng nay",
    "kéo dài dai dẳng suốt hai ngày qua",
    "tái phát đợt này nặng hơn nhiều so với trước",
    "bị liên tục từ chiều tối qua đến nay",
]

TRAIN_CLOSINGS = [
    "Xin hỏi tình trạng này có nguy hiểm không và phải xử trí thế nào?",
    "Nhờ bác sĩ phân luồng giúp xem nên làm gì ngay lúc này?",
    "Tình huống này cần cấp cứu hay có thể uống thuốc theo dõi tại nhà?",
    "Gia đình đang rất lo lắng, mong bác sĩ hướng dẫn bước tiếp theo.",
]

EVAL_CLOSINGS = [
    "Xin bác sĩ chỉ định ngay khoa khám và cách sơ cứu khẩn cấp.",
    "Bác sĩ đánh giá mức độ khẩn cấp và hướng dẫn xử lý giúp em.",
    "Trường hợp này nên nhập viện ngay hay đặt lịch khám online được ạ?",
    "Rất mong nhận được hướng dẫn xử trí chính xác từ bác sĩ.",
]

DISPOSITION_RULES = {
    "cap_cuu": "goi_115",
    "khan_cap": "den_phong_kham",
    "tieu_chuan": "hen_kham_online",
    "theo_doi": "tu_cham_soc",
}

INSTRUCTION = (
    "Phân loại yêu cầu sàng lọc y tế từ xa sau thành JSON với đúng 4 khóa: "
    "chuyen_khoa, muc_do_khan_cap, nhom_doi_tuong, huong_xu_ly. Chỉ trả về JSON, không giải thích.\n\n"
    "chuyen_khoa thuộc: tim_mach | ho_hap | tieu_hoa | da_lieu | than_kinh\n"
    "muc_do_khan_cap thuộc: cap_cuu | khan_cap | tieu_chuan | theo_doi\n"
    "nhom_doi_tuong thuộc: tre_em | nguoi_lon | nguoi_cao_tuoi | phu_nu_mang_thai\n"
    "huong_xu_ly thuộc: goi_115 | den_phong_kham | hen_kham_online | tu_cham_soc"
)


def make_sample(rng: random.Random, is_eval: bool, specialty: str, demo_group: str) -> dict:
    pool = SPECIALTY_DATA[specialty]["eval_symptoms"] if is_eval else SPECIALTY_DATA[specialty]["train_symptoms"]
    demos = EVAL_DEMOGRAPHICS if is_eval else TRAIN_DEMOGRAPHICS
    openers = EVAL_OPENERS if is_eval else TRAIN_OPENERS
    times = EVAL_TIMES if is_eval else TRAIN_TIMES
    closings = EVAL_CLOSINGS if is_eval else TRAIN_CLOSINGS

    symptom, urgency = rng.choice(pool)
    demo_str = rng.choice(demos[demo_group])
    opener = rng.choice(openers)
    time_str = rng.choice(times)
    closing = rng.choice(closings)

    # Vulnerable populations may upgrade standard cases to urgent
    if demo_group in ("phu_nu_mang_thai", "tre_em", "nguoi_cao_tuoi"):
        if urgency == "tieu_chuan" and rng.random() < 0.35:
            urgency = "khan_cap"

    disposition = DISPOSITION_RULES[urgency]
    body = f"{opener} {demo_str}, hiện đang bị {symptom}, {time_str}. {closing}"
    label = {
        "chuyen_khoa": specialty,
        "muc_do_khan_cap": urgency,
        "nhom_doi_tuong": demo_group,
        "huong_xu_ly": disposition,
    }
    return {
        "text": body,
        "label": label,
    }


def to_record(sample: dict) -> dict:
    return {
        "instruction": INSTRUCTION,
        "input": sample["text"],
        "output": json.dumps(sample["label"], ensure_ascii=False),
        "label": sample["label"],
    }


def check_decontamination(train: list[dict], eval_set: list[dict]) -> dict:
    train_inputs = {r["input"] for r in train}
    eval_inputs = {r["input"] for r in eval_set}
    
    exact_leak = train_inputs.intersection(eval_inputs)

    # Check symptom phrase leakage: no eval symptom should appear in train inputs
    eval_symptom_leak = 0
    for spec in SPECIALTY_DATA:
        for symp, _ in SPECIALTY_DATA[spec]["eval_symptoms"]:
            for tr in train:
                if symp in tr["input"]:
                    eval_symptom_leak += 1

    return {
        "exact_leak_count": len(exact_leak),
        "eval_symptom_leak": eval_symptom_leak,
        "is_clean": len(exact_leak) == 0 and eval_symptom_leak == 0,
    }


def main():
    rng = random.Random(SEED)
    DATA.mkdir(exist_ok=True)
    
    specialties = list(SPECIALTY_DATA.keys())
    demo_groups = list(TRAIN_DEMOGRAPHICS.keys())
    
    # 1. Generate 250 train samples
    train_seen: Set[str] = set()
    train_records: List[dict] = []
    while len(train_records) < 250:
        spec = rng.choice(specialties)
        demo = rng.choice(demo_groups)
        s = make_sample(rng, is_eval=False, specialty=spec, demo_group=demo)
        if s["text"] in train_seen:
            continue
        train_seen.add(s["text"])
        train_records.append(to_record(s))

    # 2. Generate 50 eval samples (strictly distinct pool)
    eval_seen: Set[str] = set()
    eval_records: List[dict] = []
    while len(eval_records) < 50:
        spec = rng.choice(specialties)
        demo = rng.choice(demo_groups)
        s = make_sample(rng, is_eval=True, specialty=spec, demo_group=demo)
        if s["text"] in eval_seen or s["text"] in train_seen:
            continue
        eval_seen.add(s["text"])
        eval_records.append(to_record(s))

    # 3. Rigorous decontamination check
    decontam = check_decontamination(train_records, eval_records)
    assert decontam["is_clean"], f"Decontamination FAILED: {decontam}"

    # 4. Save to files
    train_path = DATA / "custom_train.jsonl"
    eval_path = DATA / "custom_eval.jsonl"
    
    with train_path.open("w", encoding="utf-8") as fh:
        for r in train_records:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
            
    with eval_path.open("w", encoding="utf-8") as fh:
        for r in eval_records:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
            
    print(f"Generated {len(train_records)} custom train samples at {train_path}")
    print(f"Generated {len(eval_records)} custom eval samples at {eval_path}")
    print(f"Decontamination check: PASSED (Exact leaks: {decontam['exact_leak_count']}, Symptom leaks: {decontam['eval_symptom_leak']})")
    
    # Distribution stats
    spec_dist: Dict[str, int] = {}
    urgency_dist: Dict[str, int] = {}
    for r in train_records + eval_records:
        lbl = r["label"]
        spec_dist[lbl["chuyen_khoa"]] = spec_dist.get(lbl["chuyen_khoa"], 0) + 1
        urgency_dist[lbl["muc_do_khan_cap"]] = urgency_dist.get(lbl["muc_do_khan_cap"], 0) + 1
        
    print("\nSpecialty distribution (Total 300):", spec_dist)
    print("Urgency distribution (Total 300):", urgency_dist)


if __name__ == "__main__":
    main()
