from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Trainer, TrainingArguments, DataCollatorForSeq2Seq
from datasets import load_dataset
from peft import get_peft_model, PromptTuningConfig, PromptTuningInit
import torch
import os

class PromptTuningService:
    def __init__(self, base_model_name="google/flan-t5-small", device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.base_model_name = base_model_name
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(base_model_name)

    def apply_prompt_tuning(self, num_virtual_tokens=8):
        prompt_config = PromptTuningConfig(
            task_type="SEQ_2_SEQ_LM",
            num_virtual_tokens=num_virtual_tokens,
            prompt_tuning_init=PromptTuningInit.TEXT,
            prompt_tuning_init_text="Generate an IT interview question:",
            tokenizer_name_or_path=self.base_model_name
        )
        self.model = get_peft_model(self.model, prompt_config)

    def train_prompt_tuning(self, dataset_path, output_dir="results"):
        dataset = load_dataset("json", data_files=dataset_path, split="train")

        def tokenize(example):
            return self.tokenizer(
                example["prompt"],
                text_target=example["completion"],
                truncation=True,
                padding="max_length",
                max_length=128
            )

        tokenized_dataset = dataset.map(tokenize, remove_columns=dataset.column_names)
        data_collator = DataCollatorForSeq2Seq(tokenizer=self.tokenizer, model=self.model)

        training_args = TrainingArguments(
            output_dir=output_dir,
            per_device_train_batch_size=4,
            num_train_epochs=3,
            logging_dir=os.path.join(output_dir, "logs"),
            save_strategy="no",
            learning_rate=5e-4
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=tokenized_dataset,
            tokenizer=self.tokenizer,
            data_collator=data_collator,
        )

        trainer.train()

    def generate_from_prompt(self, prompt, max_length=128):
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        self.model.to(self.device)
        output_ids = self.model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_length=max_length,
            do_sample=True,
            temperature=0.7
        )
        return self.tokenizer.decode(output_ids[0], skip_special_tokens=True)


def main():
    service = PromptTuningService(base_model_name="google/flan-t5-small")
    service.apply_prompt_tuning()

    service.train_prompt_tuning(dataset_path="data.jsonl")

    prompts = [
        "Generate an IT interview question for a Junior DevOps Engineer skilled in AWS.",
        "Generate an IT interview question for a Mid-Level Frontend Developer skilled in React.",
        "Generate an IT interview question for a Senior Security Engineer skilled in threat modeling."
    ]

    print("\n🧪 Generated Questions:")
    for i, prompt in enumerate(prompts):
        result = service.generate_from_prompt(prompt)
        print(f"\n[{i+1}] Prompt: {prompt}\n=> {result}")


if __name__ == "__main__":
    main()
