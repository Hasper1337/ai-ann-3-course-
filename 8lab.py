import torch
from transformers import GPT2Tokenizer, T5ForConditionalGeneration
from datasets import load_dataset

# https://huggingface.co/RussianNLP/FRED-T5-Summarizer

tokenizer = GPT2Tokenizer.from_pretrained('RussianNLP/FRED-T5-Summarizer',eos_token='</s>')
model = T5ForConditionalGeneration.from_pretrained('RussianNLP/FRED-T5-Summarizer')
device='cpu'
model.to(device)

# https://huggingface.co/datasets/RussianNLP/russian_super_glue
# dataset = load_dataset("RussianNLP/russian_super_glue", "muserc")
dataset = load_dataset("json", data_files="test.jsonl", split="train")

for i, example in enumerate(dataset):
  if i >= 1:
    break

  paragraph = example['passage']['text']

  for j, ques in enumerate(example['passage']['questions']):
    if j >= 2:
      break
    question = ques['question']

    input_text=f'<LM> Вопрос: {question} Ответь, используя текст: {paragraph}'

    input_ids=torch.tensor([tokenizer.encode(input_text)]).to(device)
    outputs=model.generate(input_ids,eos_token_id=tokenizer.eos_token_id,
                      num_beams=5,
                      min_new_tokens=17,
                      max_new_tokens=200,
                      do_sample=True,
                      no_repeat_ngram_size=4,
                      top_p=0.9)
    print(f"Вопрос: {question}")
    print("Ответ модели: ", tokenizer.decode(outputs[0][1:]))
