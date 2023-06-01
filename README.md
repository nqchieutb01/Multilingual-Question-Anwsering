# Multilingual Question Answering Project


## Reader Phase
We used XLM-R-base to fine-tuning on 2 source datasets:
* UIT-vquad 
* MLQA (https://github.com/facebookresearch/MLQA)

We used  augmentation technique to enhancing model performance. Specifically, we papraphased the questions in the data by:
```
    Vietnamese question --(Translate)-> Chinese question --(Translate)-> Vietnamese question
    For exmaple: 
      Origin : Qua đầu thế kỷ 21, Jackson bắt đầu hợp tác cùng các nhà soạn nhạc nổi tiếng nào?
      After paraphasing: Jackson bắt đầu hợp tác với những nhà soạn nhạc nổi tiếng nào vào đầu những năm 2000?
```

Deep Translator is used [https://github.com/nidhaloff/deep-translator] to call Google API.

Our model can work well in some questions type like What, When, Where in both English and Vietnamese.

### Usage: 
```
from transformers import pipeline

# My check-point 've already push to huggingface. 
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
