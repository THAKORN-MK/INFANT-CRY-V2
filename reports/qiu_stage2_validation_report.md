# รายงานผล Validation: การเปรียบเทียบโมเดล Stage 2 ตามแนวทาง Qiu et al. (2024)

**สถานะ:** วิเคราะห์ Train/Validation และตรวจความสอดคล้องของ artifacts แล้ว; ยังไม่มีผล Test
**วันที่จัดทำ:** 11 พฤศจิกายน 2026
**ขอบเขต:** รายงานนี้ใช้ Validation predictions ของ candidate ทั้ง 4 ตัวเท่านั้น

## สรุปสำหรับอ่านเร็ว

- ทั้ง 4 training runs จบด้วยสถานะ `complete`; จำนวน Validation เท่ากัน 183 ตัวอย่าง และ hash ของ artifacts ตรงกับข้อมูลที่ freeze ในแต่ละ run
- `qiu_mmt_lstm` ได้ Validation Macro-F1 สูงสุด 78.67%; `qiu_mfcc_reslstm` ได้ Balanced Accuracy สูงสุด 79.52%
- Candidate ที่ใกล้โครงสร้างหลักของ Qiu มากที่สุดคือ `qiu_mmt_reslstm`; ผล Validation ของ candidate นี้คือ Accuracy 81.97%, Balanced Accuracy 72.23%, Macro-F1 74.03%
- คะแนนดังกล่าวเป็นผลจาก split เดียว (seed 42) และจำนวนตัวอย่างบางคลาสน้อย จึงยังไม่ใช่หลักฐานว่า candidate ใดดีกว่าในข้อมูลทั่วไป
- **ไม่ได้ประเมิน Test ในรายงานนี้**: protocol ของโครงการบันทึกว่ามีการ claim การประเมิน Test ของ split นี้แล้ว แต่การประเมินล้มเหลว และกำหนดห้ามลองซ้ำ/reset/delete split เดิม การประเมินยืนยันผลจึงต้องใช้ test partition ใหม่ที่ยังไม่ถูกเปิดหรือข้อมูลอิสระ ภายใต้ protocol ที่อาจารย์อนุมัติ

## 1. คำถามและเงื่อนไขการทดลอง

เปรียบเทียบ representation และ classifier สำหรับการจำแนกเสียงร้อง InfantCry-DBL เป็น 5 คลาส โดยใช้ข้อมูล Train/Validation, split, class order และ seed เดียวกัน

- Dataset: InfantCry-DBL Mendeley V1; `model_generation: v2` เป็นชื่อรุ่นภายในโครงการ ไม่ใช่ Dataset release V2
- Tier: `InfantCry-1214`
- Split: `seed-42-a241c8eba2da`; seed 42; Train 848 และ Validation 183
- Class order: `eairh`, `eh`, `heh`, `neh`, `owh`
- การฝึก: สูงสุด 50 epochs, batch size 32, Adam, learning rate 0.001, early-stopping patience 8; checkpoint metric คือ Validation Macro-F1
- Runtime ที่ใช้ฝึก: GPU ตาม config ของ Qiu comparison, พร้อม mixed precision
- วิธีอ้างอิง: Qiu et al., “Classification of Infant Cry Based on Hybrid Audio Features and ResLSTM,” *Journal of Voice* (2024), [DOI: 10.1016/j.jvoice.2024.08.022](https://doi.org/10.1016/j.jvoice.2024.08.022)

การทดลองนี้เป็น **การเปรียบเทียบที่ได้แรงบันดาลใจจากบทความ** ไม่ใช่ exact reproduction: ใช้ TensorFlow/Keras, split และ training recipe ของโครงการ และ MMT fusion layout เป็นการตีความที่ระบุไว้ใน config ไม่ได้ยืนยันว่าเหมือน implementation ต้นฉบับทุกประการ

## 2. ผลรวมของแต่ละ candidate

| Candidate | Run ID | จำนวน epochs | Best epoch* | Loss | Accuracy | Balanced accuracy | Macro-F1 |
|---|---|---:|---:|---:|---:|---:|---:|
| `qiu_mfcc_reslstm` | `20261110T202046390835Z_a6214a3e` | 43 | 35 | 0.7720 | 82.51% | **79.52%** | 76.83% |
| `qiu_mmt_resnet` | `20261110T202329009983Z_6555c05e` | 30 | 22 | 1.1491 | 78.69% | 66.84% | 70.15% |
| `qiu_mmt_lstm` | `20261110T202541583966Z_23f4ae7b` | 18 | 10 | 0.7704 | **83.61%** | 77.70% | **78.67%** |
| `qiu_mmt_reslstm` | `20261110T202708160301Z_59f790cb` | 29 | 21 | 0.6854 | 81.97% | 72.23% | 74.03% |

\* `Best epoch` แสดงเป็นเลขเริ่มนับจาก 1; ในไฟล์ metrics ต้นทางบันทึกเป็น zero-based epoch

Run IDs ข้างต้นใช้ระบุแหล่งที่มาของผล; model weights, predictions, metrics JSON และ run manifests เก็บไว้ในเครื่องและไม่รวมใน public repository เพื่อไม่นำ artifacts หรือข้อมูลที่สร้างจากรายชื่อไฟล์ตัวอย่างขึ้นเผยแพร่

## 3. จำนวนตัวอย่างและผลแยกตามคลาส

Validation มี class imbalance: `owh` มี 100 ตัวอย่าง ขณะที่ `eairh` และ `heh` มีคลาสละ 16 ตัวอย่าง ค่าในตารางเป็น Precision / Recall / F1

| คลาส | Support | `qiu_mfcc_reslstm` | `qiu_mmt_resnet` | `qiu_mmt_lstm` | `qiu_mmt_reslstm` |
|---|---:|---:|---:|---:|---:|
| `eairh` | 16 | 68.4 / 81.2 / 74.3% | 83.3 / 62.5 / 71.4% | 70.6 / 75.0 / 72.7% | 72.7 / 50.0 / 59.3% |
| `eh` | 30 | 92.0 / 76.7 / 83.6% | 73.3 / 73.3 / 73.3% | 78.1 / 83.3 / 80.6% | 95.2 / 66.7 / 78.4% |
| `heh` | 16 | 65.0 / 81.2 / 72.2% | 66.7 / 62.5 / 64.5% | 84.6 / 68.8 / 75.9% | 100.0 / 56.2 / 72.0% |
| `neh` | 21 | 57.7 / 71.4 / 63.8% | 75.0 / 42.9 / 54.5% | 78.9 / 71.4 / 75.0% | 55.6 / 95.2 / 70.2% |
| `owh` | 100 | 93.5 / 87.0 / 90.2% | 81.6 / 93.0 / 86.9% | 88.2 / 90.0 / 89.1% | 87.7 / 93.0 / 90.3% |

## 4. Confusion matrices

แถวคือคลาสจริง; คอลัมน์คือคลาสที่โมเดลทำนาย

### `qiu_mfcc_reslstm`

| จริง \ ทำนาย | eairh | eh | heh | neh | owh |
|---|---:|---:|---:|---:|---:|
| eairh | 13 | 0 | 0 | 3 | 0 |
| eh | 1 | 23 | 4 | 0 | 2 |
| heh | 1 | 0 | 13 | 1 | 1 |
| neh | 0 | 0 | 3 | 15 | 3 |
| owh | 4 | 2 | 0 | 7 | 87 |

### `qiu_mmt_resnet`

| จริง \ ทำนาย | eairh | eh | heh | neh | owh |
|---|---:|---:|---:|---:|---:|
| eairh | 10 | 2 | 1 | 1 | 2 |
| eh | 0 | 22 | 1 | 0 | 7 |
| heh | 0 | 2 | 10 | 1 | 3 |
| neh | 1 | 0 | 2 | 9 | 9 |
| owh | 1 | 4 | 1 | 1 | 93 |

### `qiu_mmt_lstm`

| จริง \ ทำนาย | eairh | eh | heh | neh | owh |
|---|---:|---:|---:|---:|---:|
| eairh | 12 | 0 | 0 | 2 | 2 |
| eh | 0 | 25 | 0 | 1 | 4 |
| heh | 1 | 0 | 11 | 1 | 3 |
| neh | 2 | 1 | 0 | 15 | 3 |
| owh | 2 | 6 | 2 | 0 | 90 |

### `qiu_mmt_reslstm`

| จริง \ ทำนาย | eairh | eh | heh | neh | owh |
|---|---:|---:|---:|---:|---:|
| eairh | 8 | 0 | 0 | 5 | 3 |
| eh | 1 | 20 | 0 | 3 | 6 |
| heh | 1 | 0 | 9 | 3 | 3 |
| neh | 0 | 0 | 0 | 20 | 1 |
| owh | 1 | 1 | 0 | 5 | 93 |

## 5. การตีความผล

1. **`qiu_mmt_lstm` มีผลรวม Validation ดีที่สุดตาม Macro-F1** ในการทดลองนี้ โดย F1 ของทั้งห้าคลาสอยู่ระหว่าง 72.7% ถึง 89.1% อย่างไรก็ตาม Recall ของ `heh` อยู่ที่ 68.8%; ยังมีตัวอย่างคลาสเล็กที่ทำนายผิด
2. **`qiu_mfcc_reslstm` ได้ Balanced accuracy สูงสุด** และ Recall ของ `eairh`/`heh` เท่ากับ 81.2% แต่ F1 ของ `neh` ต่ำสุดใน candidate นี้ที่ 63.8%
3. **`qiu_mmt_resnet` มีจุดอ่อนที่ `neh`**: ทำนายถูก 9 จาก 21 ตัว และ 9 ตัวถูกทำนายเป็น `owh`
4. **`qiu_mmt_reslstm` จับ `neh` ได้ดี (20/21)** แต่ Recall ของ `eairh` เท่ากับ 50.0% และ `heh` เท่ากับ 56.2%; นี่เป็นจุดอ่อนที่ควรตรวจหาสาเหตุ ไม่ควรสรุปจาก accuracy รวมอย่างเดียว
5. การเปรียบเทียบนี้เป็นการเทียบ pipeline ที่มี representation และสถาปัตยกรรมต่างกัน ไม่ใช่การแยกผลเชิงสาเหตุของ feature หรือ architecture เพียงองค์ประกอบเดียว

## 6. Preflight และความถูกต้องของ artifacts

- โหลด `qiu_mmt_reslstm` จาก `best_model.keras` บน CPU ด้วย tensor จำลอง (ไม่ใช้ WAV): input shape `(None, 84, 48, 1)`, output `(1, 5)`, ค่า finite และผลรวม softmax เท่ากับ 1
- Validation predictions ของทั้ง 4 run มี 183 แถว และใช้ sample IDs กับ true labels ชุดเดียวกัน
- คำนวณ Accuracy, Balanced accuracy และ Macro-F1 จาก predictions ซ้ำแล้ว ตรงกับ `validation_metrics.json` ของทุก run
- ผลรวม probability ของแต่ละตัวอย่างใกล้ 1 โดยความคลาดเคลื่อนสูงสุดไม่เกิน `1.5 × 10⁻⁷`
- SHA-256 ของ training artifacts ที่ระบุใน `metadata.json` ตรวจตรงกันทั้ง 4 run
- การ preflight นี้ยืนยันการโหลดโมเดล/shape และความสอดคล้องของ Validation artifacts แต่ **ไม่ได้รัน end-to-end test data loader** เพราะ Test ของ split นี้ไม่สามารถเปิดซ้ำได้ตาม protocol

ระหว่างดึง MMT features มีคำเตือน librosa `n_fft=512 is too large for input signal ...` คำเตือนนี้เกิดในเส้นทางคำนวณ Tonnetz/CQT ภายใน; ตัวเลขความยาวในข้อความไม่ได้ระบุความยาว WAV ต้นฉบับโดยตรง และไม่ใช่ error ที่ทำให้ training run ล้มเหลว ควรเก็บเป็นข้อสังเกตและตรวจการทำงานกับคลิปสั้นก่อนอ้างผลทั่วไป

## 7. สถานะการเลือก candidate และ Test

Config ของ comparison ระบุ Validation Macro-F1 เป็นเกณฑ์เลือกไว้ล่วงหน้า ดังนั้นหากทำตามเกณฑ์นั้น `qiu_mmt_lstm` เป็นอันดับหนึ่งใน Validation แต่ candidate ที่ใกล้โครงสร้างหลักของบทความคือ `qiu_mmt_reslstm` ซึ่งมีจุดประสงค์ทางวิจัยต่างกัน การเลือก `qiu_mmt_reslstm` เป็นวิธีหลักเพราะต้องการศึกษาตาม Qiu ไม่ควรเขียนว่าเป็นผู้ชนะตาม Macro-F1

selection manifest เดิมที่ `manifests/selections/seed-42-a241c8eba2da.json` เป็น selection ของ candidate set เดิม 5 ตัว (เลือก `mfcc_svm`) ไม่ใช่ selection สำหรับ Qiu candidates 4 ตัว จึงใช้แทนกันไม่ได้ และไม่มี selection record ใหม่สำหรับ Qiu set ในรายงานนี้; generated manifests ไม่รวมใน public repository

เอกสาร protocol ระบุว่า split `seed-42-a241c8eba2da` เคยถูก claim เพื่อ Test evaluation แล้วแต่ล้มเหลว และห้าม retry/reset/delete split เดิม ดังนั้น:

- รายงานนี้ไม่ได้อ่าน Test labels/audio และไม่ได้เรียก `evaluate-test`
- ยังไม่มี Test accuracy, Test confusion matrix หรือข้อสรุป generalization
- ไม่ควรนำ selection เก่ามาใช้กับ Qiu candidates
- การประเมินยืนยันผลถัดไปต้องมี protocol ที่อาจารย์อนุมัติ และ test partition ใหม่ที่ยังไม่ถูกเปิดหรือ Dataset อิสระ

## 8. ข้อจำกัด

- ใช้ split เดียวและ seed เดียว; ไม่มี repeated-seed uncertainty
- Validation มี 183 ตัวอย่าง และคลาสมีจำนวนไม่เท่ากัน; คะแนนคลาสที่มี support 16–21 ตัวอย่างไม่เสถียรเท่าคลาสที่มี 100 ตัวอย่าง
- `clip_id` ใช้เป็น group key ตาม Dataset recommendation แต่ยังยืนยันความหมายว่าเป็น infant/session identifier ไม่ได้ จึงไม่อ้างว่าเป็น subject-independent evaluation
- ใช้ Dataset corpus เดียว; ไม่มี external validation
- labels เป็นป้ายกำกับใน Dataset ไม่ใช่การวินิจฉัยทางคลินิก
- รายงานนี้เป็นผล Validation-only ไม่ใช่ผลยืนยันประสิทธิภาพภายนอกหรือหลักฐานการตีพิมพ์/การยอมรับใน Scopus

## 9. ขั้นตอนที่ต้องตัดสินใจก่อนทำต่อ

1. ยืนยันว่าจุดประสงค์หลักคือ (ก) ศึกษา/ทำตามวิธี MMT + ResLSTM ของ Qiu โดยให้ `qiu_mmt_reslstm` เป็นวิธีหลัก หรือ (ข) เลือก candidate ตาม Validation Macro-F1 ซึ่งขณะนี้ชี้ไปที่ `qiu_mmt_lstm`
2. ขออาจารย์อนุมัติแผน test ใหม่ที่ไม่ใช้ split ที่เคย claim แล้ว หรือจัดหา Dataset ภายนอกที่มีสิทธิ์ใช้และ label ตรวจสอบได้
3. เมื่อ protocol ใหม่ถูกกำหนดไว้ล่วงหน้า จึงล็อก candidate set/selection rule และประเมินตามแผนนั้นเพียงครั้งเดียว
4. ก่อนรายงานผลวิจัย ให้เพิ่ม repeated seeds หรือ group-aware cross-validation ตามที่อาจารย์เห็นชอบ และรายงานข้อจำกัดเรื่อง subject identity อย่างตรงไปตรงมา
