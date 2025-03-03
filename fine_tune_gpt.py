import logging
from transformers import (
    GPT2Tokenizer,
    GPT2LMHeadModel,
    TextDataset,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

# Set up logg
logging.basicConfig(level=logging.INFO)


def load_dataset(file_path, tokenizer, block_size=128):
    """
    Loads a text dataset for language modeling from a file.

    Args:
        file_path (str): Path to the text file.
        tokenizer (PreTrainedTokenizer): Tokenizer to use.
        block_size (int, optional): Block size for the dataset. Defaults to 128.

    Returns:
        TextDataset: The loaded text dataset.
    """
    return TextDataset(
        tokenizer=tokenizer,
        file_path=file_path,
        block_size=block_size,
        overwrite_cache=True,
    )


def fine_tune_gpt2(train_file, output_dir, num_train_epochs=3, per_device_train_batch_size=2):
    """
    Fine-tunes GPT-2 on the movie data and saves the model.

    Args:
        train_file (str): Path to the training data file.
        output_dir (str): Directory to save the fine-tuned model.
        num_train_epochs (int, optional): Number of training epochs. Defaults to 3.
        per_device_train_batch_size (int, optional): Batch size per device during training. Defaults to 2.
    """
    model_name = "gpt2"
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name)

    train_dataset = load_dataset(train_file, tokenizer)
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=False)

    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        save_steps=500,
        save_total_limit=2,
        prediction_loss_only=True,
        logging_steps=100,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
    )

    trainer.train()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Model saved to {output_dir}")


if __name__ == "__main__":
    train_file = "../data/movie_data.txt"
    output_dir = "../gpt2-movie-finetuned"
    fine_tune_gpt2(train_file, output_dir)
