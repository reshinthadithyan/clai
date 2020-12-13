from transformers import pipeline

model_path = r"/home/reshinth-adith/reshinth/clai/submission-code/src/submission_code/t5_actual/best_tfmr_1212"

Text2Text = pipeline("text2text-generation", model=model_path, tokenizer=model_path)
print(
    Text2Text(r'Display all lines containing \"IP_MROUTE\" in the current kernel')
)
