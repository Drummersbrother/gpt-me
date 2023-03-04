import argparse

import torch
import tqdm

from lang_model import BigramLanguageModel, estimate_loss
from src.data_utils import load_model, save_model
from src.dataset import get_datasets


def train(args: argparse.Namespace):
    train_ds, val_ds = get_datasets(args)
    tokenizer = train_ds.tokenizer

    args.vocab_size = tokenizer.vocab_size

    model = BigramLanguageModel(args).to(args.device)

    try:
        model_sd = load_model(args.save_to)
        model.load_state_dict(model_sd)
    except FileNotFoundError:
        print(f"Training from scratch, going to save to {args.save_to}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)

    for it in tqdm.trange(args.max_iters, desc="Train batches"):
        # Eval periodically
        if it % args.eval_interval == 0:
            losses = estimate_loss(model, train_ds, val_ds, args)
            print(f"step {it}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

        if it % args.save_interval == 0:
            print("Saving model checkpoint...")
            model_sd = model.state_dict()
            save_model(args.save_to, model_sd)
            del model_sd
            print("Saved.")

        # Get a sample
        xb, yb = train_ds.get_batch()

        # Evaluate loss
        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    model = model.eval()

    model_sd = model.state_dict()
    save_model(args.save_to, model_sd)
    return model, tokenizer


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    hyperparams = dict(
        batch_size=256,
        block_size=64,
        max_iters=int(5e4),
        eval_interval=5000,
        save_interval=5000,
        learning_rate=3e-4,
        device="cuda" if torch.cuda.is_available() else "cpu",
        eval_iters=200,
        n_embed=384,
        n_head=6,
        n_layer=6,
        dropout=0.2,
    )

    hyperparam_rep = repr(hyperparams).replace("'", "").replace(" ", "") \
                         .replace(",", "-").replace(":", "=")[1:-1]
    hyperparams["save_to"] = f"model_with_params_{hyperparam_rep}.pth"

    for k, v in hyperparams.items():
        parser.add_argument(f"--{k}", default=v, type=type(v))
    passed_args = parser.parse_args()

    model, tokenizer = train(passed_args)
    context = torch.zeros((1, 1), dtype=torch.long, device=passed_args.device)
    print(tokenizer.decode(model.generate(context, max_new_tokens=5000)[0].tolist()))
