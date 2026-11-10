# InfantCry-DBL Research

พื้นที่ทดลองวิจัยใหม่สำหรับ Stage 2: จำแนกเสียง InfantCry-DBL เป็น 5 ป้ายกำกับ โดยแยกจากโปรเจกต์เดิม

## สถานะ

- โฟกัสปัจจุบัน: Stage 2 เท่านั้น
- `model_generation: v2` เป็นชื่อรุ่นงาน/โมเดลภายในโครงการ ไม่ใช่ release version ของ Dataset
- Dataset ที่อยู่ในเครื่องระบุ release `1.0` และตรงกับระเบียน Mendeley V1 (DOI `10.17632/x493z8nmwc.1`); ชื่อโฟลเดอร์ที่ลงท้าย `_v2` ใช้แยกรุ่นโมเดล ไม่ได้หมายความว่า Dataset เป็น V2
- ฝึกและเปรียบเทียบ Qiu-informed Stage 2 candidates แล้ว; README นี้จงใจไม่แสดงตัวเลขผลลัพธ์ ให้ดู [Report Hub](reports/report_hub.md)
- พบ exact-content duplicate 3 ครอบครัวที่มี label ขัดกัน (รวม 6 records); บันทึกคัดออกจากการวิเคราะห์ไว้ใน `data/curation/infantcry_dbl_v2_exclusions.csv` โดยไม่ลบหรือแก้ไฟล์เสียงและ metadata ต้นฉบับ
- เตรียมข้อมูลก่อน train เสร็จแล้ว: audit ที่แก้การแยก version อยู่ใน `manifests/model_v2_dataset_v1/`; locked split อยู่ที่ `manifests/splits/model_v2_dataset_v1/seed-42-a241c8eba2da/`
- split นี้มี Train 848, Validation 183 และ Test 180 records; Test ถูกล็อกไว้สำหรับการประเมินครั้งเดียวหลังเลือก candidate ด้วย Validation เท่านั้น
- Test ของ split นี้ยังไม่มีผลประเมินที่ใช้รายงานได้ และห้าม retry split เดิมตาม protocol; รายละเอียดและข้อจำกัดอยู่ใน [Report Hub](reports/report_hub.md)
- `requirements-lock.txt` ยังเป็น placeholder ไม่ใช่ dependency lock ที่ติดตั้งได้; ตรวจ dependency/runtime ตาม `pyproject.toml` และบันทึกสภาพแวดล้อมของแต่ละ run ก่อนทำซ้ำ

## เตรียมข้อมูล

ผู้ใช้เป็นผู้คัดลอกชุดข้อมูลไว้ที่:

```text
data/raw/infantcry_dbl_v2/
```

เก็บโครงสร้างและ metadata ต้นฉบับไว้ ไม่เปลี่ยนชื่อหรือย้ายไฟล์เพื่อให้ audit เทียบ path ได้ โฟลเดอร์ข้อมูลและ run ถูกกันออกจาก Git โดย `.gitignore`

เริ่มจากดูรายการไฟล์และชื่อ header จริงก่อน โดยยังไม่เปลี่ยนไฟล์:

```bash
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
python -m infantcry_research.cli inspect-dataset --dataset-root data/raw/infantcry_dbl_v2
```

นำ header ที่พบมาใส่ใน `configs/dataset_dbl_v2.yaml` ตามความหมายที่ยืนยันจาก metadata/README ของชุดข้อมูล ห้ามเดาชื่อ field. หากมีทั้ง label ใน metadata และ label จากโฟลเดอร์ ให้เปิด `cross_check_parent_label: true` หลังตรวจ parent depth แล้ว เพื่อให้ audit ปฏิเสธค่าที่ขัดกัน. หากไม่มีคอลัมน์ class ให้ใช้ `label_source: parent_folder` เฉพาะเมื่อโครงสร้างโฟลเดอร์ที่ตรวจพบตรงกับเอกสาร release. `grouping_identity_verified` ต้องคงเป็น `false` จนกว่าจะยืนยันจากเอกสารต้นทางได้ว่า ID เป็นหน่วยกลุ่มอิสระจริง. ตรวจ release ก่อนรัน audit:

```bash
python -m infantcry_research.cli audit --dataset-root data/raw/infantcry_dbl_v2
```

ระบบอ่านรุ่น release จาก `README.md` และ `DATASET_DESCRIPTION.md` ที่มากับชุดข้อมูลได้ หากไม่มีหลักฐาน release ที่อ่านได้ ให้ตรวจเอกสารต้นทางก่อนใช้ `--confirm-release`:

```bash
python -m infantcry_research.cli audit --dataset-root data/raw/infantcry_dbl_v2 --confirm-release
```

คำสั่ง audit จะตรวจ decode เสียงทุกไฟล์ และสร้าง `manifests/dataset_audit.json`, `manifests/records.csv` และ `manifests/excluded_records.csv` เฉพาะเมื่อ schema, path และไฟล์เสียงผ่าน validation. ไฟล์ผลเดิมจะไม่ถูกเขียนทับจนกว่าจะใส่ `--overwrite` เอง

### การคัดกรอง label ที่ขัดกัน

ตรวจพบ exact-content duplicate 3 ครอบครัวที่ปรากฏใน metadata คนละ label รวม 6 records. เนื่องจากยังยืนยัน label ที่ถูกต้องจากแหล่งต้นฉบับไม่ได้ จึงคัดทั้งครอบครัวออกจากชุดวิเคราะห์ด้วย ledger ที่ระบุ path, SHA-256 และเหตุผลครบถ้วน; ไฟล์เสียงและ metadata ต้นฉบับยังคงอยู่โดยไม่มีการแก้ไข. Audit เดิมที่ `manifests/dataset_audit.json` และ `manifests/records.csv` ถูกเก็บไว้ ไม่เขียนทับ

สร้าง audit ชุดใหม่แยกจากหลักฐานเดิมด้วยคำสั่ง:

```bash
python -m infantcry_research.cli audit \
  --dataset-root data/raw/infantcry_dbl_v2 \
  --config configs/dataset_dbl_v2.yaml \
  --output-dir manifests/model_v2_dataset_v1
```

Audit ที่แก้ config แล้วอยู่ใน `manifests/model_v2_dataset_v1/`: 1,551 source records, 1,545 records สำหรับวิเคราะห์ และ 6 records ที่คัดออก. Audit เก่าที่ `manifests/curated_v1/` ใช้ config รุ่นก่อนซึ่งสับสน `model_generation` กับ `dataset_version`; เก็บไว้เพื่อ provenance เท่านั้น ห้ามใช้เป็น split/training evidence. ระเบียนทางการยืนยัน Mendeley V1, DOI `.1`, และ 1,551 rows ([Mendeley Data V1](https://data.mendeley.com/datasets/x493z8nmwc/1)).

Exclusion ledger ผูกกับ SHA-256 ของไฟล์เสียงที่ตรวจ หากข้อมูลเปลี่ยน ledger จะปฏิเสธ hash mismatch จนกว่าจะตรวจสอบใหม่.

Audit และ locked split สำหรับข้อมูลสำเนาปัจจุบันสร้างและตรวจแล้ว. Split ใช้ `group_id` เป็นหน่วยจัดกลุ่ม โดยมี Train 848, Validation 183, Test 180 records และจำนวนกลุ่ม 754/162/162. `grouping_identity_verified` ยังเป็น `false`: การแบ่งกลุ่มนี้ไม่ได้ยืนยันว่าเป็น subject/infant-independent. อย่าสร้าง split ทับหรือเปลี่ยน seed โดยไม่ตั้งใจ เพราะ split ปัจจุบันเป็นหลักฐานที่ใช้ร่วมกันของ candidate ทั้งหมด.

ไฟล์ `train_validation_records.csv` ในโฟลเดอร์ split บรรจุเฉพาะแถว Train/Validation พร้อม hash ที่ตรวจสอบได้; คำสั่ง train ใช้ไฟล์นี้และเสียง Train/Validation เท่านั้น. การอ่านแถว Test จาก `records.csv` และการโหลดเสียง Test เพื่อคำนวณผลทำหลัง validation selection ถูกตรึงและ ledger claim สำเร็จแล้วเท่านั้น.

`group_id` จะใช้ field ที่กำหนดใน config. ชื่อคอลัมน์อย่าง `infant_id` ไม่ได้พิสูจน์ความหมายของ ID ด้วยตัวเอง: CI แบบ group bootstrap จะรายงานได้ต่อเมื่อผู้ทำวิจัยตรวจเอกสารจริงและตั้ง `grouping_identity_verified: true`; หากยังไม่ยืนยัน ระบบจะแสดง CI ว่า unavailable แทนการสมมติว่าแต่ละคลิปเป็นกลุ่มอิสระ. Exact-content grouping เพียงอย่างเดียวไม่ใช่ subject-independent evaluation.

## วิธีประเมินที่วางไว้

โค้ดชุดนี้ทำ Stage 2 แบบ `train → validation selection → one-time paired locked-test comparison`. การทดลอง Qiu-informed ปัจจุบันฝึกและวิเคราะห์ Validation แล้ว; ขั้นตอน Test ของ split นี้ถูกระงับตาม protocol ที่บันทึกไว้. ดูสถานะล่าสุดที่ [Report Hub](reports/report_hub.md).

เปิด terminal ใน VS Code ที่เชื่อมกับ WSL/Ubuntu แล้วตั้ง environment จาก root ของ `INFANT CRY NEW`:

```bash
cd "/path/to/INFANT CRY NEW"
source "$HOME/.venvs/audio-ml-gpu/bin/activate"
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
AUDIT_DIR="manifests/model_v2_dataset_v1"
SPLIT_MANIFEST="manifests/splits/model_v2_dataset_v1/seed-42-a241c8eba2da/split_manifest.json"
```

ไฟล์ config ระบุชัดว่า `model_generation: v2` เป็นชื่อรุ่นโมเดลภายใน ส่วน `dataset_version: "1.0"` เป็นรุ่น Dataset. หากนำข้อมูลสำเนาใหม่เข้ามา ต้องทำ inspect/audit/split แยกใหม่; อย่านำ manifest ของสำเนาปัจจุบันไปใช้อัตโนมัติ.

ก่อน train ให้ตรวจ sample rate ใน `configs/models_stage2.yaml`; profiles ที่เกี่ยวข้องตั้งไว้ที่ 16 kHz mono และสถานะ `locked`. ระบบจะปฏิเสธ train หากสถานะไม่ `locked`.

## Candidate ที่ลงทะเบียน

ระบบฝึก candidate ที่ `enabled: true` ใน `configs/models_stage2.yaml` โดยทุกตัวใช้ split, labels, seed และ training controls ร่วมกัน; แต่ละตัวมี feature/model configuration ของตนเองซึ่งถูกบันทึก hash ไว้กับ run.

| Candidate | วิธีหลัก | Input |
|---|---|---|
| `majority` | baseline จาก label ที่พบบ่อยใน Train | labels only |
| `mfcc_svm` | MFCC mean/std + RBF-SVM | feature profile `stage2_hybrid` |
| `logmel_cnn` | CNN baseline | Log-Mel 64 bins จาก profile `stage2_hybrid` |
| `cnn_bilstm_attention` | โมเดลหลักเดิมของโครงการ | MFCC + Delta + Delta2 + Log-Mel + Chroma `(196,128,1)` |
| `mmt_reslstm` | candidate แบบ paper-informed ตาม Qiu et al. | MFCC + Mel + Tonnetz `(84,48,1)`; ResLSTM |

`mmt_reslstm` ใช้แนวทางจาก Qiu et al., “Classification of Infant Cry Based on Hybrid Audio Features and ResLSTM” ([DOI](https://doi.org/10.1016/j.jvoice.2024.08.022)). PDF บทความไม่ได้รวมใน repository นี้. เป็นการนำแนวคิดมาทดลองกับข้อมูลและ protocol ของโครงการ ไม่ใช่ exact reproduction: framework, dataset release, split และ training schedule แตกต่างกัน. ลำดับแกนตอนรวม Tonnetz/Mel ในบทความไม่ชัดพอ จึงใช้การตีความที่ให้ shape ตามที่รายงาน และบันทึกข้อจำกัดนี้ไว้ใน protocol.

ใช้ split manifest เดียวกันฝึก candidates ทุกตัว โดยแต่ละคำสั่งสร้าง run แยกและใช้ validation เท่านั้น. หลังตั้งค่าตัวแปรใน terminal ตามบล็อกก่อนหน้า ให้รัน candidate ทีละคำสั่ง:

```bash
python -m infantcry_research.cli train --dataset-root data/raw/infantcry_dbl_v2 --audit-dir "$AUDIT_DIR" --split-manifest "$SPLIT_MANIFEST" --model majority --device cpu
python -m infantcry_research.cli train --dataset-root data/raw/infantcry_dbl_v2 --audit-dir "$AUDIT_DIR" --split-manifest "$SPLIT_MANIFEST" --model mfcc_svm --device cpu
python -m infantcry_research.cli train --dataset-root data/raw/infantcry_dbl_v2 --audit-dir "$AUDIT_DIR" --split-manifest "$SPLIT_MANIFEST" --model logmel_cnn --device gpu --require-gpu --mixed-precision
python -m infantcry_research.cli train --dataset-root data/raw/infantcry_dbl_v2 --audit-dir "$AUDIT_DIR" --split-manifest "$SPLIT_MANIFEST" --model cnn_bilstm_attention --device gpu --require-gpu --mixed-precision
python -m infantcry_research.cli train --dataset-root data/raw/infantcry_dbl_v2 --audit-dir "$AUDIT_DIR" --split-manifest "$SPLIT_MANIFEST" --model mmt_reslstm --device gpu --require-gpu --mixed-precision
```

แทน `<SPLIT_ID>` ด้วยชื่อโฟลเดอร์จริง และเก็บ `run_dir` ที่แต่ละคำสั่งพิมพ์ออกมา. หลังครบ candidates จึงตรึงการเลือกจาก validation macro-F1:

```bash
python -m infantcry_research.cli select-candidate \
  --run-dir runs/<MAJORITY_RUN_ID> \
  --run-dir runs/<MFCC_SVM_RUN_ID> \
  --run-dir runs/<LOGMEL_CNN_RUN_ID> \
  --run-dir runs/<CNN_BILSTM_RUN_ID> \
  --run-dir runs/<MMT_RESLSTM_RUN_ID>
```

ก่อน fit โมเดล run แรกจะบันทึก SHA-256 ของ `models_stage2.yaml` และรายชื่อ candidates ที่เปิดใช้งานไว้ใน run config. ค่า device และ mixed precision อยู่ใน config ของแต่ละ candidate. ทุก candidate และคำสั่ง selection ต้องใช้ config เดิมทุก byte. Selection ต้องมี run ที่สมบูรณ์ของ candidate ที่เปิดใช้งานครบทุกตัว ใช้ split/dataset evidence, class order, seed และ training controls ตรงกัน; feature contract ต่างกันได้เมื่อถูกลงทะเบียนไว้ล่วงหน้า. การเลือกผู้สมัครทำด้วย validation macro-F1 เท่านั้น. หลัง Selection ถูกตรึงแล้วจึงเปิด test หนึ่งครั้งเพื่อประเมิน candidate ทั้งชุด:

```bash
python -m infantcry_research.cli evaluate-test \
  --selection-file manifests/selections/<SPLIT_ID>.json \
  --dataset-root data/raw/infantcry_dbl_v2 \
  --audit-dir "$AUDIT_DIR" \
  --output-dir runs/comparisons \
  --bootstrap-resamples 2000 \
  --device cpu
```

คำสั่งต้องมี Selection ที่แสดง candidate ครบทุกตัว; มันจะสร้าง ledger claim หนึ่งรายการและ package เดียวที่ `runs/comparisons/<SPLIT_ID>/` พร้อม metrics, predictions และ confusion matrix ของทุก candidate บน test IDs ชุดเดียวกัน. หาก process หยุดหลัง claim จะไม่อนุญาตให้เปิด test ซ้ำหรือเพิ่ม candidate สำหรับ split เดิม. Test metrics ใช้เพื่อรายงานเชิงพรรณนาเท่านั้น ห้ามใช้เลือก จูน จัดอันดับ หรือ promote โมเดล. สร้างรายงานจาก package:

```bash
python -m infantcry_research.cli report \
  --comparison-dir runs/comparisons/<SPLIT_ID> \
  --output-dir reports/stage2
```

ระบบตรวจ config, model artifacts และ Selection ก่อน claim; หลัง claim จึงเปิดแถว Test จาก `records.csv` และโหลดเสียง Test เพื่อทำ paired evaluation. ขอบเขตนี้เริ่มหลัง split ถูกล็อก: ขั้น audit/split ก่อนหน้านั้นต้องตรวจไฟล์และ label metadata ทั้งชุดเพื่อสร้าง stratified partition. รายงาน comparison มี confusion matrix CSV/PNG, predictions, metrics และความไม่แน่นอนแยกตาม candidate; group-bootstrap interval จะแสดงได้เฉพาะเมื่อยืนยัน grouping identity แล้วเท่านั้น. Tier transfer, repeated-seed summary, ablations/augmentation และการประเมิน Sinc-Mamba-KAN ยังไม่ได้ implement/run ใน pipeline นี้.

ไม่มีการอ้างว่าเป็นการวินิจฉัยทางการแพทย์ หรือว่าใช้ได้กับประชากร/อุปกรณ์อื่นจากผลภายใน DBL เพียงชุดเดียว

## เอกสาร

- [Dataset provenance and citation](DATASET.md)
- [Research protocol](docs/research_protocol.md)
- [Model comparison plan](docs/model_comparison.md)
- [Limitations](docs/limitations.md)
- [Report hub](reports/report_hub.md)
