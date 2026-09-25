# Báo cáo Lab Ngày 08: Học chủ động cho bộ phát hiện xe

Họ và tên: Nguyễn Công Thành

Công cụ gán nhãn đã dùng: CVAT (Docker local), import/export định dạng Ultralytics YOLO Detection 1.0

Mọi con số truy được từ `reports/rounds_table.md`, `outputs/selection_round1.csv`,
`outputs/metrics_round*.json` hoặc `outputs/round1_diff.md`. Nhãn test do mô hình tạo, chưa được người
rà, nên không được coi là chân lý tuyệt đối.

## 1. Dữ liệu và cách chia tập

Tại sao tập chưa gán nhãn (pool) và tập kiểm thử (test set) được chia theo trục thời gian, có vùng
đệm ở giữa, thay vì chia ngẫu nhiên? Nếu chia ngẫu nhiên, số đo trên tập kiểm thử sẽ bị lệch theo
hướng nào, và vì sao?

Dữ liệu là 400 ảnh lấy từ một video 160 s (2.5 ảnh/s) quay bằng camera cố định trên cầu vượt. Hai ảnh
liền nhau chỉ cách 0.4 s nên gần như giống hệt, và mỗi chiếc xe ở trong khung hình vài giây. Nếu chia
ngẫu nhiên, cùng một chiếc xe ở gần cùng một vị trí sẽ xuất hiện cả trong tập train lẫn tập test. Mô
hình khi đó được chấm trên những chiếc xe nó đã học, nên số đo trên test bị **lệch lên, cao hơn thực
tế** (rò rỉ dữ liệu).

Vì vậy dữ liệu được chia theo thời gian: 20 ảnh test lấy ở 4 đoạn quanh giây 20, 60, 100 và 140; 112
ảnh vùng đệm quanh các đoạn test bị loại; 268 ảnh còn lại làm pool. Ảnh pool gần ảnh test nhất vẫn cách
4.4 s (`data/DATA.md`, `data/frames.csv`), đủ để các xe trong test không phải là cùng những chiếc xe
trong ảnh train. Cách chia này vẫn không loại được hoàn toàn độ giống nhau vì nền cảnh và góc camera
không đổi, nên test đo khả năng tổng quát trên cùng một camera chứ không phải camera mới.

## 2. Mô hình khởi đầu lạnh (cold start)

Chép dòng vòng 0 từ `rounds_table.md`. Dựa vào `outputs/compare_round0.jpg`, cho biết mô hình khởi
đầu lạnh không khớp nhãn tham chiếu ở những loại xe nào. Độ phủ (recall) theo kích thước xe cho
thấy điều gì? Một trường hợp nào cần người rà lại nhãn tham chiếu trước khi kết luận mô hình sai?

| vòng | model | ảnh train | box train | AP50 | P@0.25 | R@0.25 | F1 | R small | R medium | R large |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | yolov8n cold start (COCO car+bus+truck) | 0 | 0 | 0.771 | 0.925 | 0.489 | 0.640 | 0.182 | 0.547 | 0.561 |

Trên 403 box tham chiếu (đã bỏ 14 box cao dưới 16 px), cold start có TP 197, FP 16, FN 206. Precision
cao (0.925) nhưng recall chỉ 0.489: mô hình ít vẽ sai nhưng bỏ sót khoảng một nửa số xe.

Theo ảnh so sánh, mô hình không khớp nhãn tham chiếu chủ yếu ở:
- **xe nhỏ ở xa**, chỉ còn cụm đèn đỏ/đèn pha: recall xe nhỏ chỉ 0.182 trên 66 box;
- **xe gần camera bị chói đèn pha**, thân xe tối hòa vào mặt đường;
- **xe đi sát nhau**: một số box đỏ (FP) là box ôm hai xe cùng lúc hoặc ôm vệt đèn ở xa.

Recall theo kích thước cho thấy mô hình COCO chưa quen cảnh đêm nhìn từ trên cao: càng nhỏ càng sót
(0.182 → 0.547 → 0.561). Ngay cả xe lớn cũng chỉ đạt 0.561, tức vấn đề không chỉ nằm ở kích thước mà
còn ở ánh sáng ban đêm.

Trường hợp cần rà lại nhãn tham chiếu: xe bị mép ảnh cắt mất một phần ở frame test `frame_0250` (mép
phải). Nhãn tham chiếu có box ở đó nhưng cold start không vẽ. Vì nhãn test cũng do mô hình tạo, cần
người xem box tham chiếu có ôm đúng phần xe nhìn thấy không trước khi kết luận đây là lỗi bỏ sót.

## 3. Chiến lược chọn mẫu

Giải thích bằng lời công thức `score = W_U·U + W_A·A + W_D·D` và vai trò của `MIN_GAP_S`.
Dẫn ba frame trong `reports/SELECTION.md` và một frame khác để chứng minh cách bạn cân nhắc
độ bất định, ảnh gần trùng và công gán nhãn. Điểm bất định có chứng minh ảnh đó sẽ cải thiện
mô hình không? Vì sao?

Mỗi ảnh trong pool được chấm `score = 0.5·U + 0.3·A + 0.2·D`:
- **U (độ bất định):** trung bình 5 box khó nhất của ảnh, mỗi box tính `1 − |2·conf − 1|`, lớn nhất
  khi conf = 0.5 (mô hình phân vân nhất).
- **A (độ mập mờ):** số box có conf trong khoảng 0.15–0.50, chia cho giá trị lớn nhất trong pool.
- **D (độ đa dạng):** khoảng cách thời gian tới ảnh đã gán gần nhất, chặn ở 10 s. Ở vòng 1 chưa có
  ảnh nào được gán nên D = 1.0 cho mọi ảnh, tức thành phần này chưa có tác dụng.

`MIN_GAP_S = 2.0` bắt buộc hai ảnh trong cùng lô cách nhau ít nhất 2 s. Nếu không có ràng buộc này,
các ảnh liền kề (cách 0.4 s, gần như trùng) sẽ chiếm lô vì có điểm gần bằng nhau.

Dẫn chứng từ `reports/SELECTION.md`:
- **frame_0182** (rank 1, score 0.959): điểm cao nhờ A = 1.0 (18 box mập mờ). Sau khi sửa, số box tăng
  từ 13 lên 26, tức những box mập mờ đa phần là xe thật mà mô hình chưa đủ tự tin.
- **frame_0331** (rank 5, score 0.915): 47 box dự đoán; khi sửa phải xoá 4 box sai và thêm 7. Đây là
  ảnh tốn công rà.
- **frame_0392** (rank 15, score 0.887): U = 0.975 cao nhất top 15 nhưng A chỉ 0.667, tức ít box mập mờ
  nhưng các box khó có conf rất sát 0.5.
- **Frame khác, frame_0330** (rank 12, score 0.890, 53 box): cách frame_0331 chỉ 0.4 s nên bị
  `MIN_GAP_S` loại. Đây là ảnh gần trùng: rà nó tốn nhiều công nhất trong top 50 mà gần như không thêm
  thông tin mới.

Điểm bất định **không chứng minh** ảnh đó sẽ cải thiện mô hình. Điểm được tính từ dự đoán của chính mô
hình, không so với nhãn đúng, nên nó đo chỗ mô hình phân vân chứ không đo chỗ mô hình sai. Mô hình cũng
có thể tự tin sai (bỏ sót xe hoàn toàn), và những ảnh như vậy có điểm thấp nên không được chọn. Thực tế
ở lab này: fine-tune trên đúng 12 ảnh điểm cao nhất nhưng AP50 lại giảm (mục 4).

## 4. Các vòng học chủ động (active learning)

Chép bảng từ `rounds_table.md`. Với mỗi vòng, trình bày:

- mức độ bạn đã sửa nhãn gợi ý (số box giữ nguyên, chỉnh sửa, xoá, thêm mới, lấy từ
  `outputs/round*_diff.md`);
- AP50 thay đổi bao nhiêu so với khởi đầu lạnh và so với vòng trước;
- nhóm xe nào tốt lên hoặc xấu đi theo số đo trên cùng tập test.

Dựa vào các ảnh `compare_round*.jpg`, chỉ ra một ca kết quả đổi sau fine-tune (tốt hơn hoặc xấu
đi), cùng lý do có thể kiểm. Dùng `BLIND_SCAN.md`, `REVIEW_LOG.csv` và `round1_diff.md` phân biệt
quan sát độc lập, lỗi pre-label đã sửa và kết quả mô hình sau train. Mô tả một ca khó theo guideline.

Ghi chú: `BLIND_SCAN.md` được khóa lại lúc 04:54 UTC chỉ để thêm đuôi `.jpg` vào tên frame; nội dung
quan sát giữ nguyên bản gốc khóa lúc 03:26 UTC.

| vòng | model | ảnh train | box train | AP50 | Δ AP50 so cold start | P@0.25 | R@0.25 | F1 | R small | R medium | R large |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | yolov8n cold start (COCO car+bus+truck) | 0 | 0 | 0.771 | — | 0.925 | 0.489 | 0.640 | 0.182 | 0.547 | 0.561 |
| 1 | yolov8n fine-tune vong 1..1 | 12 | 284 | 0.578 | -0.194 | 1.000 | 0.223 | 0.365 | 0.000 | 0.240 | 0.463 |

**Mức sửa nhãn vòng 1** (`outputs/round1_diff.md`): mô hình đề xuất 169 box, sau khi sửa còn 284 box:
giữ nguyên 156, chỉnh 2, xoá 11, thêm 126. Accept rate 92% cho thấy box mô hình đề xuất phần lớn đúng,
nhưng phải thêm 126 box (gần 75% số box đề xuất) vì mô hình bỏ sót nhiều, khớp với recall 0.489 của
cold start.

**Thay đổi số đo:** AP50 giảm 0.194 so với cold start (0.771 → 0.578); vòng 1 cũng là vòng trước nên
mức giảm so với vòng trước là như nhau. Precision tăng lên 1.000 (không còn FP) nhưng recall giảm ở mọi
nhóm: xe nhỏ 0.182 → 0.000, xe trung bình 0.547 → 0.240, xe lớn 0.561 → 0.463.

**Ca kết quả đổi sau fine-tune** (`outputs/compare_round1.jpg`):

| frame test | cold start | vòng 1 |
| --- | --- | --- |
| frame_0050 | TP 11, FP 2, FN 7 | TP 5, FP 0, FN 13 |
| frame_0150 | TP 10, FP 2, FN 10 | TP 4, FP 0, FN 16 |
| frame_0250 | TP 6, FP 2, FN 9 | TP 3, FP 0, FN 12 |
| frame_0350 | TP 9, FP 2, FN 14 | TP 7, FP 0, FN 16 |

Tốt hơn: các box sai ở cold start (box ôm hai xe, box ôm vệt đèn) biến mất ở cả 4 frame. Xấu đi: ở
frame_0250, hai xe lớn ngay cuối ảnh gần camera cũng bị bỏ sót, dù đây là xe rõ nhất. Lý do có thể
kiểm: khi fine-tune, đầu phân loại chuyển từ lớp COCO sang một lớp `car` duy nhất, nên mô hình mất một
phần khả năng nhận diện xe đã học từ COCO, trong khi 12 ảnh (284 box, 50 epoch) chưa đủ để học lại. Nhiều
khả năng mô hình vẫn thấy xe nhưng với conf dưới ngưỡng 0.25; có thể kiểm bằng cách hạ ngưỡng conf hoặc
xem đường precision–recall.

**Phân biệt ba nguồn bằng chứng:**
- *Quan sát độc lập* (`BLIND_SCAN.md`, khóa trước khi xem pre-label): ở frame_0099 tôi đếm được 26 xe
  và dự đoán hai xe dễ bị bỏ sót, đều là xe tối màu ở góc dưới bên trái và bên phải, bị đèn pha làm chói.
- *Lỗi pre-label đã sửa* (`round1_diff.md`, `REVIEW_LOG.csv`): ở frame_0099 mô hình đề xuất 13 box, nhãn
  cuối có 19 box (thêm 6), xác nhận mô hình bỏ sót như dự đoán. Nhãn cuối vẫn ít hơn 26 xe đếm bằng mắt;
  phần chênh chủ yếu là xe ở xa chỉ còn cụm đèn, box cao dưới khoảng 16 px, mà guideline cho phép không
  gán vì không được tính khi chấm. Trong `REVIEW_LOG.csv` tôi ghi 3 ca: frame_0099 chỉnh box ôm cả xe
  khác, frame_0187 xoá box ôm ánh đèn đường, frame_0182 thêm xe bị che một phần.
- *Kết quả mô hình sau train* (`compare_round1.jpg`, `metrics_round1.json`): đo trên 20 ảnh test khác,
  không liên quan trực tiếp tới ảnh tôi đã sửa.

**Ca khó theo guideline:** ở frame_0182, một xe ở vùng tối bị xe con phía trước che một phần. Mô hình
không đề xuất box. Theo guideline, xe bị che vẫn phải gán nếu còn nhìn thấy thân xe, và box chỉ ôm phần
nhìn thấy, không đoán phần bị che. Tôi thêm box cho phần nhìn thấy.

## 5. Kết luận và giới hạn

Kết quả vòng này so với cold start ra sao? Vì sao bạn dừng hoặc tiếp tục? Đề xuất hai ca còn yếu
hoặc bất định cho vòng sau, kèm chi phí rà nhãn và nguy cơ ảnh gần trùng. Tập kiểm thử chỉ 20 ảnh,
có luật bỏ qua xe quá nhỏ và nhãn tham chiếu do mô hình tạo chưa được rà thủ công; các giới hạn đó
ảnh hưởng thế nào đến kết luận? Nếu AP50 giảm, bạn sẽ kiểm tra điều gì trước khi train thêm?

**So với cold start:** vòng 1 kém hơn (AP50 0.578 so với 0.771, F1 0.365 so với 0.640). Mô hình sau
fine-tune không còn vẽ sai nhưng bỏ sót nhiều hơn hẳn, nhất là xe nhỏ.

**Vì sao dừng:** tôi dừng sau vòng 1 vì giới hạn thời gian và công rà nhãn. Mỗi ảnh cần thêm trung bình
khoảng 10 box, và pre-label vòng 2 do chính mô hình vòng 1 (recall 0.223) tạo ra nên sẽ bỏ sót nhiều
hơn, tốn công hơn. Ngoài ra, với kết quả giảm như trên, cần tìm nguyên nhân trước khi thêm dữ liệu.

**Hai ca đề xuất cho vòng sau** (lô `to_label/round2/`):
- *Xe nhỏ ở xa* (recall 0.000 sau vòng 1): cần ảnh có nhiều xe nhỏ và gán đủ box trên 16 px. Chi phí cao
  vì box nhỏ khó kéo chính xác; nên dùng SAM trong CVAT để hỗ trợ.
- *Xe gần camera bị chói đèn pha* (như ở frame_0250): mô hình vòng 1 bỏ sót cả xe lớn. Chi phí thấp hơn vì
  xe to, dễ vẽ. Nguy cơ ảnh gần trùng: lô vòng 2 có frame_0002 và frame_0016, frame_0068 và frame_0074
  cách nhau vài giây, cảnh xe có thể lặp lại.

**Giới hạn:** tập test chỉ 20 ảnh (403 box) từ 4 đoạn thời gian, nên chênh lệch một vài xe đã làm số đo
dao động đáng kể. Luật bỏ 14 box dưới 16 px làm recall xe nhỏ không phản ánh các xe rất xa. Nhãn tham
chiếu do mô hình tạo, chưa người rà: nếu nhãn đó sót hoặc sai, AP50 đo mức khớp với một mô hình khác chứ
không phải độ chính xác thật. Các kết luận trên chỉ đúng cho camera và cảnh đêm này.

**Nếu AP50 giảm, cần kiểm trước khi train thêm:**
1. Nhãn train có đủ xe nhỏ và xe bị chói chưa, có nhất quán giữa các ảnh không.
2. Ngưỡng conf 0.25 có quá cao với mô hình mới không: xem AP50 (không phụ thuộc ngưỡng) và thử hạ ngưỡng.
3. Mô hình có overfit 12 ảnh không: so loss train với kết quả test, thử ít epoch hơn hoặc freeze backbone.
4. Có nên fine-tune giữ lại lớp `car` của COCO thay vì đầu phân loại mới, để không mất tri thức sẵn có.
