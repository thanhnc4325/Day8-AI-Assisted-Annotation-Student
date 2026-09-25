# Vì sao chọn lô này?

Trong 50 dòng đứng đầu `outputs/selection_round1.csv`, chọn năm frame bạn sẽ ưu tiên nếu chỉ có
ngân sách rà năm ảnh. Ghi tên, điểm, thời điểm, thứ tự và lý do; tối thiểu một quyết định phải xét
ảnh gần trùng hoặc trường hợp model không dự đoán được box:

| Ưu tiên | Frame | Rank CSV | Score | t (s) | Lý do |
| ---: | --- | ---: | ---: | ---: | --- |
| 1 | frame_0182.jpg | 1 | 0.959 | 72.8 | Điểm cao nhất pool; A = 1.0 (18 box mập mờ, nhiều nhất pool). Đại diện đoạn giữa video. |
| 2 | frame_0369.jpg | 2 | 0.932 | 147.6 | U = 0.932, 43 box dự đoán; đại diện đoạn cuối video, cảnh xe dày đặc. |
| 3 | frame_0326.jpg | 4 | 0.916 | 130.4 | Chọn thay cho frame_0331 (rank 5, 132.4 s): hai ảnh chỉ cách 2.0 s, đúng bằng `MIN_GAP_S`, cảnh gần như giống nhau. frame_0326 có ít box dự đoán hơn (39 so với 47) nên rà rẻ hơn mà vẫn phủ cùng đoạn 124–133 s. |
| 4 | frame_0099.jpg | 8 | 0.906 | 39.6 | U = 0.946, cao nhất trong top 10; 29 box nên chi phí rà thấp. Là frame sớm nhất trong top 10, phủ đoạn đầu video mà các frame trên chưa có. |
| 5 | frame_0270.jpg | 13 | 0.888 | 108.0 | Lấp khoảng trống 75–125 s giữa các frame đã chọn. Trên contact sheet có xe tải/xe lớn ở giữa làn; recall xe lớn của cold start chỉ 0.561 nên đáng rà. |

Không đưa frame_0380 (rank 3, 152.0 s) vào top 5 dù điểm cao: nó chỉ cách frame_0369 4.4 s, cùng cụm
cảnh xe dày ở cuối video, nên với ngân sách năm ảnh, thêm một đoạn thời gian khác có ích hơn.

Ba frame thuộc lô 12 ảnh model chọn và bằng chứng trong CSV/ảnh contact sheet:

- **frame_0182.jpg** (rank 1, score 0.959, U 0.918, A 1.000, 28 box, 18 box mập mờ). Điểm cao chủ yếu
  nhờ A. Sau khi sửa (`round1_diff.md`), model chỉ đề xuất 13 box ở conf ≥ 0.25 nhưng nhãn cuối có 26
  box (thêm 13): nhiều box mập mờ thực chất là xe mà model không đủ tự tin.
- **frame_0331.jpg** (rank 5, score 0.915, U 0.831, A 1.000, 47 box, 18 box mập mờ). U thấp hơn các
  frame xung quanh nhưng A tối đa đẩy điểm lên. Đây là frame phải xoá nhiều nhất (4 box sai trên 20
  đề xuất) và vẫn thêm 7 box, nên tốn công rà. Contact sheet cho thấy xe chen nhau ở giữa ảnh, dễ ra
  box trùng hoặc box ôm hai xe.
- **frame_0392.jpg** (rank 15, score 0.887, U 0.975, A 0.667, 35 box, 12 box mập mờ). Ngược với
  frame_0331: U cao nhất trong top 15 nhưng A thấp, tức ít box mập mờ nhưng những box khó nhất có
  conf rất gần 0.5. Contact sheet có xe lớn màu trắng bên phải; sau sửa có 1 box chỉnh, 1 box xoá và
  12 box thêm.

Một frame có điểm cao nhưng không chọn hoặc một frame có điểm thấp vẫn nên xem, và lý do:

- **Điểm cao nhưng không chọn: frame_0330.jpg** (rank 12, score 0.890, t = 132.0 s, 53 box, nhiều nhất
  top 50). Frame này chỉ cách frame_0331 đã chọn 0.4 s, nên `MIN_GAP_S = 2.0` loại nó. Đây là ảnh gần
  trùng: rà thêm tốn công nhất (53 box) mà gần như không mang thông tin mới. Tương tự, frame_0372
  (rank 6) và frame_0368 (rank 9) bị loại vì cách frame_0369 1.2 s và 0.4 s.
- **Điểm thấp vẫn nên xem: frame_0195.jpg** (rank 268/268, score 0.572, t = 78.0 s, 20 box, 3 box mập
  mờ). Điểm thấp nghĩa là model tự tin, không có nghĩa là model đúng. Frame này chỉ có 20 box dự đoán,
  ít hơn phần lớn frame trong top 50 (thường 25–45 box), nên có thể model bỏ sót xe hẳn (conf < 0.05) chứ không
  phải ảnh dễ. Cold start có recall 0.489, nên tự tin sai là khả năng thực tế.

Điều phép chọn này chưa chứng minh về chất lượng mô hình:

- Điểm được tính từ dự đoán của chính model, không dựa trên nhãn đúng. U và A đo model *phân vân*,
  không đo model *sai*; confidence của model chưa được hiệu chỉnh nên chưa chắc phản ánh xác suất đúng.
- Điểm cao không chứng minh ảnh đó giúp cải thiện model: sau khi train trên đúng 12 ảnh điểm cao nhất,
  AP50 trên test giảm từ 0.771 xuống 0.578.
- Ở vòng 1 chưa có ảnh nào được gán nên D = 1.0 cho mọi frame. Trọng số đa dạng thời gian chưa có tác
  dụng; độ phủ thời gian chỉ nhờ `MIN_GAP_S`, và lô vẫn dồn vào hai cụm 124–133 s và 147–157 s.
- Frame model bỏ sót hoàn toàn (tự tin sai) không bao giờ được chọn theo cách chấm điểm này.
