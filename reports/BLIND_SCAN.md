# Quét độc lập trước khi xem pre-label

Frame: frame_0099.jpg
Số xe nhìn thấy bằng mắt: 26

Hai vị trí dễ bị AI bỏ sót hoặc vẽ sai, kèm mô tả xe: 
1. Góc dưới bên phải
- Vị trí: sát mép phải, phía dưới ảnh.
- Mô tả xe: một ô tô màu tối/đen, chỉ nhìn thấy một phần thân xe do bị cắt bởi mép ảnh; phía trước có đèn pha sáng.
- Khó phát hiện: xe bị khuất/cắt một phần, ánh sáng yếu và đèn pha gây chói → AI dễ bỏ sót hoặc vẽ box quá rộng.
2. Góc dưới bên trái
- Vị trí: gần mép trái, phía dưới ảnh.
- Mô tả xe: một ô tô màu tối, đang chạy hướng về camera; hai đèn pha sáng rõ, phần thân xe khá tối.
- Khó phát hiện: xe nằm trong vùng tối, tương phản thấp với mặt đường và bị ánh đèn phía trước làm mất chi tiết → AI có thể không nhận diện được toàn bộ thân xe hoặc vẽ bounding box lệch.

Chạy `python3 tools/lock_blind.py` ngay sau khi điền. Sau đó giữ file này nguyên vẹn.
