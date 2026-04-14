INFERENCE CONFIDENCE & REJECTION STRATEGY
==========================================

VẤNS LIVE: Tại sao "no_sign"?
-----------------------------

Trong app.py, hàm predict_from_seq() có logic:

```python
# Check 1: Activity
if not _is_sign_activity_sufficient(raw_seq):
    return "no_sign"  ← Video quá tĩnh

# Check 2: Confidence
top1 = probs[top_sorted[0]]  # Best prediction
top2 = probs[top_sorted[1]]  # Second best
margin = top1 - top2

if top1 < MIN_TOP1_CONFIDENCE or margin < MIN_TOP12_MARGIN:
    return "no_sign"  ← Không chắc chắn
```


SCENARIO 1: Confidence Quá Thấp
================================

Video: D0001B.webm (Albania)
Model output:
    Albania: 0.35 (← < old threshold 0.45)
    Do Thái: 0.25
    ...

OLD (sai):
    top1 = 0.35 < 0.45 → Return "no_sign" ❌

NEW (đúng):
    top1 = 0.35 > 0.25 ✓
    margin = 0.35 - 0.25 = 0.10 > 0.02 ✓
    → Return "Albania" ✅


SCENARIO 2: Top 2 Quá Gần
============================

Video: D0005N.webm
Model output:
    Tuy Hoà: 0.50
    Miến Điện: 0.48  ← Quá gần!
    ...

OLD (reject):
    margin = 0.50 - 0.48 = 0.02 < 0.08 → "no_sign" ❌
    (Model không chắc là Tuy Hoà hay Miến Điện)

NEW (accept):
    margin = 0.02 ≥ 0.02 ✓
    top1 = 0.50 > 0.25 ✓
    → Return "Tuy Hoà" ✅
    (Accept đó là Tuy Hoà, dù không 100% chắc)


SCENARIO 3: Video Quá Tĩnh
============================

Video: person just standing, no sign
Frame analysis:
    active_ratio = 0.05 (< 0.20 old, < 0.10 new) ❌
    motion_energy = 0.0001 (< 0.0005 new) ❌

Result:
    Activity check fails → Return "no_sign" ✅
    (Đúng, vì thực sự không có sign!)


CONFIGURATION GIẢI THÍCH
=========================

MIN_TOP1_CONFIDENCE = 0.25
    ↳ Mô hình chỉ cần "chắc chắn 25%" là được
    ↳ Thấp hơn = dễ dàng hơn để accept prediction
    ↳ CAO hơn = khó khắc hơn, bỏ sót sign nên

MIN_TOP12_MARGIN = 0.02
    ↳ Top 1 và Top 2 chỉ cần khác nhau 2%
    ↳ Thấp hơn = cảng chấp nhận ambiguity
    ↳ CAO hơn = càng strict, bắt phải rõ ràng


TRADEOFF
========

Confidence thấp:
    ✓ Tốt: Recognize nhiều cases
    ✗ Bad: False positives (nhận lầm)

Confidence cao:
    ✓ Tốt: Chính xác khi recognize
    ✗ Bad: Miss recognition (bỏ sót)

BALANCE: Phải tune dựa vào use case


GIÁ TRỊ KHUYẾN NGHỊ
==================

Để tìm giá trị tối ưu:

1. Collect test videos (30-50 videos)
2. Run inference trên tất cả
3. Track:
   - Correct predictions
   - False positives (wrong class)
   - False negatives ("no_sign" khi là sign)

4. Adjust thresholds:
   - Nếu quá nhiều false negatives:
     ↓ MIN_TOP1_CONFIDENCE
     ↓ MIN_TOP12_MARGIN
   
   - Nếu quá nhiều false positives:
     ↑ MIN_TOP1_CONFIDENCE
     ↑ MIN_TOP12_MARGIN


TESTING CÁC NGƯỠNG
===================

Script để test multiple thresholds:

# In app.py, thêm environment variables:
import os

MIN_TOP1_CONFIDENCE = float(os.environ.get('MIN_CONF', '0.25'))
MIN_TOP12_MARGIN = float(os.environ.get('MIN_MARGIN', '0.02'))

# Run với config khác nhau:
MIN_CONF=0.3 MIN_MARGIN=0.05 python app.py
MIN_CONF=0.2 MIN_MARGIN=0.01 python app.py

→ Xem kết quả nào tốt nhất


RECOMMENDATION
==============

Current values (updated):
    MIN_TOP1_CONFIDENCE = 0.25  ← OK cho most cases
    MIN_TOP12_MARGIN = 0.02     ← OK cho most cases
    MIN_ACTIVE_FRAME_RATIO = 0.10
    MIN_MOTION_ENERGY = 0.0005

Nếu vẫn bị "no_sign" nhiều:
    → Hạ thêm: 0.2 and 0.01
    → Or implement Test-Time Augmentation

Nếu bị false positives (nhận lầm class):
    → Tăng lên: 0.3 and 0.05
    → Or improve model training quality


DEBUGGING STEPS
================

1. Check activity first:
   python diagnostic.py
   → Xem "active_ratio" & "motion_energy"

2. If passes activity, check model confidence:
   Thêm logging vào app.py:
   
   print(f"Top-1: {probs[top1_idx]:.4f}")
   print(f"Top-2: {probs[top2_idx]:.4f}")
   print(f"Margin: {probs[top1_idx]-probs[top2_idx]:.4f}")

3. If confidence low, likely causes:
   - Data distribution mismatch (train vs inference)
   - Model not trained properly
   - Class not in training data
   - Need Test-Time Augmentation
