# QA

## Pha Reader 
Chúng tôi sử dụng mô hình XLM-R base để fine-tune với dữ liệu trên 2 nguồn như sau:
* UIT-vquad 
* MLQA (https://github.com/facebookresearch/MLQA)
	
Chúng tôi tăng cường dữ liệu bằng cách paraphase các câu hỏi trong bộ UIT-vQUAD với cách làm như sau :
```
    Câu hỏi tiếng Việt --(dịch)-> câu tiếng Trung --(dịch)-> câu tiếng Việt
    Ví dụ: 
      Câu gốc : Qua đầu thế kỷ 21, Jackson bắt đầu hợp tác cùng các nhà soạn nhạc nổi tiếng nào?
      Câu paraphase: Jackson bắt đầu hợp tác với những nhà soạn nhạc nổi tiếng nào vào đầu những năm 2000?
```
Chúng tôi dùng tool Deep Translator [https://github.com/nidhaloff/deep-translator] của để gọi API dịch câu của Google.

Dữ liệu gồm 44326 cặp context-question với context và question gồm cả tiếng Anh và tiếng Việt. 

Mô hình có thể hoạt động tốt trên các câu hỏi dạng (What, when, where) khi context và question là 1 trong 2 ngôn ngữ Anh, Việt.

### Cách dùng: 
```
from transformers import pipeline

# Replace this with your own checkpoint
model_checkpoint = "chieunq/XLM-R-base-finetuned-uit-vquad-1"
question_answerer = pipeline("question-answering", model=model_checkpoint)

context = """
Nhóm của chúng tôi là sinh viên năm 4 trường ĐH Công Nghệ - ĐHQG Hà Nội. Nhóm gồm 3 thành viên : Nguyễn Quang Chiều, Nguyễn Quang Huy và Nguyễn Trần Anh Đức . Đây là pha Reader trong dự án cuồi kì môn Các vấn đề hiện đại trong CNTT của nhóm . 
"""
question = "Who are the 3 members of the group?"
question_answerer(question=question, context=context)

```
### Output
```
{'score': 0.998,
 'start': 98,
 'end': 158,
 'answer': 'Nguyễn Quang Chiều, Nguyễn Quang Huy và Nguyễn Trần Anh Đức.'}
```
