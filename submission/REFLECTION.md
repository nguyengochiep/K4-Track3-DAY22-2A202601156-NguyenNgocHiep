# Reflection — Lab 22 (DPO/ORPO Alignment)

**Tên:** Nguyễn Ngọc Hiệp
**Cohort:** 4
**Tier đã chạy:** T4
**Date:** 2026-08-24

---

## 1. Setup

| Item | Value |
|---|---|
| GPU | Free Colab Tesla T4, 15.6 GB (Turing, compute capability 7.5, `Bfloat16 = FALSE`) |
| CUDA / driver | CUDA 12.8 · Torch 2.10.0+cu128 · Transformers 5.5.0 · Unsloth 2026.4.8 |
| Base model | `unsloth/Qwen2.5-3B-bnb-4bit` (NF4 4-bit) |
| LoRA | r=16, α=32, 29,933,568 tham số huấn luyện (0.96% của 3.12B) |
| SFT dataset slice | `saillab/alpaca-vietnamese-cleaned` · 1000 mẫu · 1 epoch · 125 step · 7 phút 53 |
| Preference dataset slice | `argilla/ultrafeedback-binarized-preferences-cleaned` · 2000 cặp → lọc còn 877 |
| `COMPUTE_TIER` env | T4 |
| Total cost | $0 (free Colab) |

Ghi chú: lab chỉ định `5CD-AI/Vietnamese-alpaca-cleaned`, nhưng repo đó nay trả HTTP 401 (đã gỡ hoặc chuyển gated), nên tôi thay bằng `saillab/alpaca-vietnamese-cleaned` — cùng schema `instruction/input/output`, 41.6k dòng. Dataset này ghi ô `input` rỗng bằng chuỗi literal `"nan"` chứ không phải `None`, nên phép kiểm tra truthiness thông thường sẽ nối `"\n\nnan"` vào khoảng 70% số prompt; tôi lọc riêng trường hợp đó trước khi áp chat template.

---

## 2. Kết quả

| Metric | SFT-only | SFT + DPO |
|---|---:|---:|
| Thời gian train | 7:53 (125 step) | không hoàn thành — xem §3 |
| Loss đầu → cuối | 1.891 → 1.506 (final 1.5509) | — |
| Reward gap cuối | n/a | — |

SFT-mini hội tụ bình thường: loss giảm từ 1.891 (step 10) xuống 1.468–1.506 ở các step cuối. Sinh thử cho kết quả tiếng Việt mạch lạc.

Một chi tiết đáng ghi nhận ở bước dựng policy cho DPO: `FastLanguageModel.get_peft_model()` in ra `Unsloth: Already have LoRA adapters! We shall skip this step.` và `peft_config` chỉ có một adapter tên `default`. Nghĩa là DPO **không** tạo adapter thứ hai chồng lên SFT như comment trong notebook mô tả — nó huấn luyện tiếp chính adapter SFT. Hệ quả: `adapters/dpo/` (nếu được tạo ra) là adapter tự chứa cả SFT lẫn DPO, và reference model mà TRL suy ra bằng cách tắt adapter chính là **base model thô**, không phải model SFT. Implicit reward `β·log(π/π_ref)` do đó bao gồm sẵn cả phần dịch chuyển do SFT tạo ra, và reward curve sẽ không xuất phát từ 0.

---

## 3. Reward curves analysis

Tôi không dựng được reward curve vì DPO không chạy được trên phần cứng được cấp.

`DPOTrainer` gọi `xformers.memory_efficient_attention` với layout GQA 5 chiều (BMGHK: batch 2 do ghép chosen+rejected, 2 nhóm KV, 8 head query, head_dim 128). Trên T4 — Turing, compute capability 7.5 — kernel backward duy nhất khả dụng là `cutlassB`, và nó không hỗ trợ BMGHK; hai kernel có hỗ trợ (`fa2B`, `fa3B`) yêu cầu capability ≥ 8.0, tức Ampere trở lên:

```
NotImplementedError: No operator found for `memory_efficient_attention_backward`
query: shape=(2, 269, 2, 8, 128) (torch.float16)
  fa3B@0.0.0:  requires device with capability >= (8, 0) but your GPU has capability (7, 5)
               operator does not support BMGHK format
  fa2B@2.5.7:  requires device with capability >= (8, 0) but your GPU has capability (7, 5)
               operator does not support BMGHK format
  cutlassB-pt: operator does not support BMGHK format
```

Đây là giới hạn phần cứng chứ không phải lỗi cấu hình. SFT ở NB1 thoát được vì Unsloth bật padding-free với batch 1; DPO buộc phải giữ đồng thời chosen và rejected trong cùng một batch nên rơi đúng vào đường thiếu kernel.

Tôi đã thử ba hướng khắc phục. **Một**, đặt `attn_implementation="eager"` khi load qua Unsloth — không tác dụng, vì Unsloth patch attention ở tầng module và bỏ qua tham số config. **Hai**, bỏ Unsloth, load bằng `AutoModelForCausalLM` với `attn_implementation="sdpa"` — vượt được lỗi xformers, nhưng lộ ra rằng TRL 0.19 phụ thuộc `MODEL_FOR_VISION_2_SEQ_MAPPING_NAMES`, một API đã bị transformers 5.5 gỡ bỏ, và Unsloth vốn âm thầm vá hộ. **Ba**, tự shim tên đó rồi lần lượt xử lý chuỗi phụ thuộc `mergekit` → `immutables`, cuối cùng bế tắc ở mergekit 0.1.4 pin `pydantic~=2.10.6` trong khi môi trường có pydantic 2.13.4 nên không dựng nổi schema cho `torch.Tensor`.

Điều rút ra: ma trận tương thích Unsloth × TRL × transformers × kiến trúc GPU là rủi ro thực của công việc alignment, không phải chi tiết vặt. Unsloth không chỉ tăng tốc — nó còn giữ cho TRL chạy được trên transformers 5.x. Chỉ khi gỡ nó ra tôi mới thấy TRL thực sự phụ thuộc những gì. Nếu làm lại, tôi sẽ kiểm tra compute capability và pin cứng phiên bản transformers **trước** khi tiêu thời gian GPU cho SFT.

---

## 4. Qualitative comparison

Không thực hiện được: NB4 cần `adapters/dpo/` mà NB3 không tạo ra (xem §3). Bộ 8 prompt đánh giá (4 helpfulness + 4 safety) đã được sinh và lưu ở `data/eval/prompts.json`.

Một lỗi khác đã phát hiện và sửa trong lúc chạy: `unsloth/Qwen2.5-3B-bnb-4bit` là **base model**, không phải Instruct, nên tokenizer của nó không kèm `chat_template`. Mọi lời gọi `apply_chat_template` trong NB1/NB2/NB4 đều hỏng. Vocab đã có sẵn `<|im_start|>` (151644) và `<|im_end|>` (151645), nên tôi gắn template ChatML thủ công và đặt `eos_token = "<|im_end|>"` — nếu để nguyên `<|endoftext|>` thì `model.generate` không dừng đúng chỗ và output sẽ tràn qua cả lượt của user.

---

## 5. β trade-off — giả thuyết

Không chạy được β-sweep. Dự đoán: β nhỏ (0.05) nới ràng buộc KL nên policy đi xa reference hơn, reward gap lớn nhất nhưng dễ lệch phân phối và output ngắn lại; β lớn (0.5) giữ policy sát reference, gap nhỏ và thay đổi ít; β = 0.1 là điểm cân bằng, khớp với lựa chọn mặc định của deck §3.3.

---

## 6. Quyết định quan trọng nhất

Quyết định đáng kể nhất không phải chọn β mà là **lọc dữ liệu preference trước khi train**. NB2 in ra cảnh báo rằng chỉ 44,2% số cặp lọt trong `MAX_LEN=512`. Ban đầu tôi định bỏ qua vì trainer vẫn chạy được — nó tự cắt phần thừa. Nhưng khi nhìn kỹ phân bố độ dài, `chosen` có median 400 token còn `rejected` chỉ 278. Nghĩa là việc cắt **không đối xứng**: câu trả lời tốt bị cắt cụt thường xuyên hơn câu trả lời kém. DPO khi đó học rằng đoạn văn bị cụt giữa chừng là đáng thưởng — đúng cơ chế sinh ra hiện tượng `chosen_reward` tụt mà deck §3.4 gọi là likelihood displacement. Tôi đã suýt ghi nhận nó như một "phát hiện thú vị về failure mode" trong khi thực chất đó là lỗi dữ liệu do chính mình gây ra.

Phương án thay thế là nâng `MAX_LEN` lên 1024, nhưng T4 16 GB không đủ VRAM cho DPO ở độ dài đó, vì mỗi step phải giữ cả chosen lẫn rejected qua hai lượt forward. Tôi chọn lọc, giữ 877/2000 cặp nguyên vẹn — mất 56% dữ liệu, đổi lấy phần còn lại sạch. Với DPO tôi tin đánh đổi này đúng: preference learning học từ *sự tương phản* giữa hai câu trả lời, nên một cặp bị méo còn hại hơn là thiếu một cặp.

Quyết định thứ hai là nâng `learning_rate` từ 5e-7 lên 5e-6. Con số 5e-7 trong deck là dành cho full fine-tuning; tôi đang train LoRA, vốn cần lr cao hơn khoảng một bậc — chính NB1 dùng 2e-4 cho SFT. Với chỉ khoảng 110 optimizer step, 5e-7 gần như chắc chắn cho reward gap phẳng, và tôi sẽ mất phần lớn giá trị chẩn đoán của biểu đồ.

Nếu làm lại lab này ngày mai: kiểm tra tương thích GPU **trước tiên**, và chọn tier phần cứng theo yêu cầu của bước *nặng nhất* trong pipeline (DPO) chứ không theo bước đầu tiên (SFT). Bài học đắt nhất hôm nay là SFT chạy trơn tru đã cho tôi một cảm giác an toàn sai lệch.

---

## 7. Benchmark

Không thực hiện (NB6 là bonus, và phụ thuộc `adapters/dpo/`).

---

## Bonus

- [ ] Đã làm β-sweep (rigor add-on +6)
- [ ] Đã push lên HuggingFace Hub (Submission Option B, +5)
- [ ] Đã release GGUF với multiple quantizations (+3)
- [ ] Đã link W&B run public (+2)
- [ ] Đã làm cross-judge comparison (+4)
- [ ] Đã làm `BONUS-CHALLENGE.md` provocation (ungraded)

---

## Điều ngạc nhiên nhất khi làm lab này

Phần khó nhất không phải thuật toán DPO mà là ma trận tương thích thư viện. Unsloth âm thầm vá nhiều chỗ transformers 5.x đã gỡ bỏ, nên chỉ khi gỡ Unsloth ra tôi mới thấy TRL thực sự phụ thuộc những gì — và mới hiểu rằng "tăng tốc" chỉ là một nửa việc nó làm.
