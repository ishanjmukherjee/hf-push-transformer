import argparse
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name",
        type=str,
        default="ishanjmukherjee/tiny-shakespeare-gpt1l",
        help="Hugging Face model repo to load"
    )
    parser.add_argument(
        "-p", "--prompt",
        type=str,
        default="Once upon a time,",
        help="Text prompt to start generation from"
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=50,
        help="How many tokens to generate past the prompt"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run on (cuda or cpu)"
    )
    args = parser.parse_args()

    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForCausalLM.from_pretrained(args.model_name).to(args.device)

    # Encode prompt
    inputs = tokenizer(args.prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].to(args.device)

    # Generate
    outputs = model.generate(
        input_ids,
        max_new_tokens=args.max_new_tokens,
        do_sample=True,
        top_k=50,
        top_p=0.95,
        temperature=0.8,
        pad_token_id=tokenizer.eos_token_id,
    )

    # Decode and print
    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(text)

if __name__ == "__main__":
    main()
