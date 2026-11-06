# Báo cáo cá nhân — Day 01: Lọc tin nhắn rác bằng ML, DL, Embedding và LLM

- **SEED** — `408`

## 1. Bằng chứng

| Tệp | Trạng thái |
| --- | --- |
| [`day01_lab.ipynb`](day01_lab.ipynb) | Có notebook đã chạy và lưu output. Đổi tên bản đính kèm thành `day01_lab.ipynb` khi đưa lên repo. |
| [`lab1_report.json`](lab1_report.json) | Có báo cáo do notebook sinh ra. 8/8 kiểm tra đã ghi nhận ở Phần 1–3 đạt. |
| [`hand_calc.jpg`](hand_calc.jpg) | Chưa có ảnh trong bộ tệp hiện tại. |

## 2. Kết quả chính

Sau khi bỏ trùng, bộ dữ liệu còn **5.169 SMS**, gồm 653 spam và 4.516 ham. Em dùng 3.619 tin để train, 775 tin để validation và 775 tin để test. Tập SUB lấy từ test, gồm 20 spam và 20 ham. Nếu luôn đoán ham, accuracy trên toàn bộ dữ liệu đã bỏ trùng vẫn đạt 87,37%, dù recall của lớp spam bằng 0. Vì vậy em dùng F1 của lớp spam để so sánh các mô hình.

| Phương án | Precision test | Recall test | F1 test | F1 SUB 40 | p95 (ms/tin) | Tin/s (cả lô) | USD/1.000 tin |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| BoW + LogReg | 0,987 | 0,736 | 0,843 | 0,750 | 0,12 | 12.465 | 0 |
| BoW + MLP | 1,000 | 0,887 | 0,940 | 0,974 | 2,63 | 10.148 | 0 |
| MiniLM + kNN | 0,896 | 0,972 | 0,932 | 1,000 | 233,36 | 15,62 | 0 |
| MiniLM + LogReg | 0,971 | 0,953 | 0,962 | 1,000 | 38,49 | 34,82 | 0 |
| MiniLM + MLP head | 0,990 | 0,943 | 0,966 | 1,000 | 39,04 | 43,54 | 0 |
| LLM nhỏ, zero-shot | — | — | — | — | — | — | — |
| LLM lớn, zero-shot | — | — | — | — | — | — | — |
| LLM nhỏ, few-shot | — | — | — | — | — | — | — |

Nguồn số liệu là [`lab1_report.json`](lab1_report.json). Chi phí trong bảng là chi phí API. Ba cấu hình LLM chưa có kết quả đo trong lần chạy này. Cả ba cấu hình MiniLM đều đạt F1 SUB bằng 1,000, nhưng SUB chỉ có 40 tin. Trên toàn bộ test, MiniLM + MLP head đạt F1 cao nhất trong các phương án đã đo, hơn MiniLM + LogReg khoảng 0,004.

## 3. Dự đoán và thực tế

| Thí nghiệm | Dự đoán của em | Kết quả và giải thích |
| --- | --- | --- |
| Bỏ ReLU (`dl_relu`) | F1 sẽ giảm vì hai lớp Linear liên tiếp tương đương một phép biến đổi affine. | F1 có và bỏ ReLU cùng hiển thị **0,940**. Giới hạn biểu diễn tuyến tính là có thật, nhưng trong lần chạy này nó chưa làm giảm F1 ở độ chính xác hiển thị. |
| Tìm kiếm theo nghĩa (`emb_search`) | Hàng xóm gần nhất là ham. | **Cả 6 hàng xóm là spam.** Truy vấn dùng cụm “complimentary holiday”, nhưng toàn câu vẫn mang ý mời nhận thưởng và yêu cầu phản hồi. Ý này gần với các tin báo trúng chuyến du lịch trong tập train. |
| Sáu tin ngụy trang (`emb_tricky`) | MiniLM + LogReg đúng nhiều hơn BoW + LogReg. | **BoW đúng 5/6, MiniLM + LogReg đúng 4/6.** BoW bắt đúng tin báo số điện thoại được chọn nhận chuyến du lịch, trong khi MiniLM + LogReg gán ham. Trong trường hợp này, các token như `chosen` và `holiday` là tín hiệu mà BoW có thể tận dụng. |

## 4. Phân tích 50 case khó

Ô stress test dừng tại bước kiểm tra dự đoán `hard_eval`, nên em chưa có F1 trên 50 case để xác định mô hình tụt nhiều nhất hoặc nhóm có accuracy thấp nhất. Hai ví dụ sau là **giả thuyết lỗi cần đối chiếu khi chạy**, dựa trên nội dung của bộ stress test.

| Case | Nhãn thật | Cơ chế có thể dẫn đến lỗi |
| --- | --- | --- |
| **#1**, tin bạn bè báo thắng cá cược rồi hẹn gọi lại | ham | Các từ `Congrats`, `WON` và `Call` có thể đẩy điểm BoW về phía spam. Nội dung cả câu cho thấy đây là cuộc trò chuyện cá nhân. |
| **#30**, lời mời tham gia quay thưởng viết `F R E E` và `t x t` | spam | Tokenizer có thể tách các chữ cái thành token riêng, khiến BoW mất các dấu hiệu quen thuộc như `free` và `txt`. |

Tập test thường có thể chứa mẫu gần trùng với train vì bước làm sạch chỉ bỏ tin trùng tuyệt đối. Bộ 50 case viết tay dùng để tìm các kiểu lỗi cụ thể như né từ khóa hoặc prompt injection. Tỉ lệ lỗi trên bộ này không đại diện cho toàn bộ lưu lượng SMS thực tế.

## 5. Thử thách cá nhân

Với SEED 408, thử thách được gán là **đề số 1**, so sánh precision, recall và F1 khi đổi ngưỡng quyết định của BoW + LogReg sang 0,2 và 0,8, dùng ngưỡng 0,5 để đối chiếu. Bản notebook hiện có chỉ ghi kết quả mặc định ở ngưỡng 0,5. Hai ngưỡng của thử thách và phần giải thích cần được bổ sung sau khi em tự chạy bài.

## 6. Phương án cho sản phẩm thật

Em chọn **BoW + MLP** làm tầng lọc chính cho bài toán 10 triệu SMS/ngày, tương đương khoảng 116 tin/giây trung bình. Trong lần đo của notebook, mô hình có **p95 2,63 ms**, F1 test **0,940**, F1 SUB **0,974** và thông lượng khoảng **10.148 tin/giây khi xử lý cả lô**. Độ trễ đo được còn nhiều khoảng cách so với giới hạn 50 ms. Thông lượng cả lô là số đo trong lab, nên cần đo lại dưới tải và với cách gọi thực tế trước khi xác nhận SLA.

MiniLM + LogReg và MiniLM + MLP head đều có F1 test cao hơn và đạt p95 dưới 50 ms trong lần đo. Tuy vậy, mỗi phương án đạt khoảng 35–44 tin/giây trên cấu hình đo hiện tại. Em sẽ thử **BoW + MLP → MiniLM + LogReg** cho các tin có điểm gần ngưỡng. Cần đo p95 đầu cuối và precision/recall của cả hệ thống vì lợi ích phụ thuộc vào số tin được chuyển sang tầng thứ hai.

Nếu chạy LLM cho toàn bộ tin trong 30 ngày, chi phí tháng bằng **300.000 × chi phí API trên 1.000 tin**. Lần chạy hiện tại chưa có đơn giá đo từ thí nghiệm LLM để điền số tiền.

## 7. AI log và điều đã học

| Phần | Em hỏi AI gì? | Em nhận được gì? | Đối chiếu với bài làm |
| --- | --- | --- | --- |
| 0–1 | Cách chia dữ liệu, cách tạo và kiểm tra ma trận BoW. | Giải thích số mẫu theo từng tập và ý nghĩa của các giá trị 0/1. | Đối chiếu số lượng mẫu với output của notebook. Kiểm tra `vectorize` đạt. |
| 2 | Vòng lặp mini-batch chọn mẫu thế nào và các bước train chạy theo thứ tự nào? | Giải thích `randperm`, chỉ số batch, forward, loss, backward và cập nhật trọng số. | Hai kiểm tra MLP đạt. Phần ghi train loss cần rà lại vì đang cộng loss trung bình từng batch rồi chia cho số mẫu. |
| 3 | Vì sao kNN báo lỗi khi dùng kết quả tìm hàng xóm để index? | Giải thích hàm tìm hàng xóm trả về cả chỉ số lẫn điểm cosine. | Notebook đã tách `idx, _` và kiểm tra `knn_predict` đạt. |
| 4 | Cách làm bốn TODO LLM và cách nhập API key. | Code tham khảo cho phần được phép dùng AI, hướng dẫn mock test và cấu hình key. | AI đã kiểm tra code tham khảo bằng mock. Notebook đính kèm chưa ghi nhận kết quả mock hoặc lời gọi API thật. |
| Báo cáo | Viết báo cáo cá nhân theo mẫu Markdown. | Bản nháp tổng hợp kết quả và nêu rõ các mục còn thiếu. | Số liệu lấy từ JSON và output notebook. |

Em hiểu rõ hơn vì sao accuracy cao vẫn có thể đi cùng với việc bỏ sót nhiều spam. Em cũng thấy việc chọn mô hình phải dựa vào cả cách biểu diễn tin nhắn, lỗi trên từng nhóm và chi phí dự đoán. Cùng một MiniLM, kNN có recall cao nhất trong lần chạy này nhưng chậm hơn đáng kể so với hai đầu phân loại đã huấn luyện.
