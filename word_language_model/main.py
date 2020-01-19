# coding: utf-8
import argparse
import logging
import math
import os
import time
import torch.optim

import torch
import torch.jit
import torch.nn as nn
import torch.onnx

import data
import model

FORMAT = "%(asctime)-15s, " + logging.BASIC_FORMAT
logging.basicConfig(format=FORMAT)
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

parser = argparse.ArgumentParser(
    description="PyTorch Wikitext-2 RNN/LSTM Language Model"
)
parser.add_argument(
    "--data", type=str, default="./data/wikitext-2", help="location of the data corpus"
)
parser.add_argument(
    "--model",
    type=str,
    default="LSTM",
    help="type of recurrent net (RNN_TANH, RNN_RELU, LSTM, GRU, Transformer, custom_LSTM)",
)
parser.add_argument("--emsize", type=int, default=200, help="size of word embeddings")
parser.add_argument(
    "--nhid", type=int, default=200, help="number of hidden units per layer"
)
parser.add_argument("--nlayers", type=int, default=2, help="number of layers")

parser.add_argument("--lr", type=float, default=0.001, help="initial learning rate")
parser.add_argument("--clip", type=float, default=0.25, help="gradient clipping")
parser.add_argument(
    "--weight_decay",
    type=float,
    default=0.0,
    help="weight decay used for regularizing the model",
)

parser.add_argument("--epochs", type=int, default=40, help="upper epoch limit")
parser.add_argument(
    "--batch_size", type=int, default=20, metavar="N", help="batch size"
)
parser.add_argument("--bptt", type=int, default=35, help="sequence length")

parser.add_argument(
    "--dropout",
    type=float,
    default=0.0,
    help="dropout applied to layers only in the Transformer model(0 = no dropout)",
)

parser.add_argument(
    "--inter_layer_dropout",
    type=float,
    default=0.0,
    help="dropout applied to inter layer connection of each LSTM layer",
)
parser.add_argument(
    "--recurrent_dropout",
    type=float,
    default=0.0,
    help="dropout applied to the recurrent state of of each LSTM layer",
)
parser.add_argument(
    "--input_dropout",
    type=float,
    default=0.0,
    help="dropout applied to the input of LSTM",
)
parser.add_argument(
    "--output_dropout",
    type=float,
    default=0.0,
    help="dropout applied to the output of the LSTM",
)

parser.add_argument(
    "--lstm_skip_connection",
    action="store_true",
    help="Summing the output of all the LSTM layers instead of returning the last one",
)
parser.add_argument(
    "--tied", action="store_true", help="tie the word embedding and softmax weights"
)
parser.add_argument("--seed", type=int, default=1111, help="random seed")
parser.add_argument("--device", type=str, default="cpu", help="device name")
parser.add_argument(
    "--log-interval", type=int, default=200, metavar="N", help="report interval"
)
parser.add_argument(
    "--save", type=str, default="model.pt", help="path to save the final model"
)
parser.add_argument(
    "--onnx-export",
    type=str,
    default="",
    help="path to export the final model in onnx format",
)

parser.add_argument(
    "--jit_forward",
    action="store_true",
    help="whether or not JIT compile the model's forward function",
)

parser.add_argument(
    "--nhead",
    type=int,
    default=2,
    help="the number of heads in the encoder/decoder of the transformer model",
)

args = parser.parse_args()

# Set the random seed manually for reproducibility.
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    if "cuda" not in args.device:
        logging.warning("You have a CUDA device, run with --device cuda")

device = torch.device(device=args.device)

###############################################################################
# Load data
###############################################################################

corpus = data.Corpus(args.data)


# Starting from sequential data, batchify arranges the dataset into columns.
# For instance, with the alphabet as the sequence and batch size 4, we'd get
# ┌ a g m s ┐
# │ b h n t │
# │ c i o u │
# │ d j p v │
# │ e k q w │
# └ f l r x ┘.
# These columns are treated as independent by the model, which means that the
# dependence of e. g. 'g' on 'f' can not be learned, but allows more efficient
# batch processing.


def batchify(data, bsz):
    # Work out how cleanly we can divide the dataset into bsz parts.
    nbatch = data.size(0) // bsz
    # Trim off any extra elements that wouldn't cleanly fit (remainders).
    data = data.narrow(0, 0, nbatch * bsz)
    # Evenly divide the data across the bsz batches.
    data = data.view(bsz, -1).t().contiguous()
    return data.to(device)


eval_batch_size = 10
train_data = batchify(corpus.train, args.batch_size)
val_data = batchify(corpus.valid, eval_batch_size)
test_data = batchify(corpus.test, eval_batch_size)

###############################################################################
# Build the model
###############################################################################

ntokens = len(corpus.dictionary)
if args.model == "Transformer":
    model = model.TransformerModel(
        ntokens, args.emsize, args.nhead, args.nhid, args.nlayers, args.dropout
    )
else:
    if args.dropout > 0.0:
        logging.warning("dropout argument is not used in the LSTM models")

    model = model.RNNModel(
        args.model,
        ntokens,
        args.emsize,
        args.nhid,
        args.nlayers,
        inter_layer_dropout=args.inter_layer_dropout,
        recurrent_dropout=args.recurrent_dropout,
        input_dropout=args.input_dropout,
        output_dropout=args.output_dropout,
        tie_weights=args.tied,
        lstm_skip_connection=args.lstm_skip_connection,
    )

if args.jit_forward:
    model = torch.jit.script(model)

model = model.to(device)
criterion = nn.CrossEntropyLoss()

n = sum(p.numel() for p in model.parameters() if p.requires_grad)

logger.info("Number of trainable parameters: {}".format(n))
# no_decay = []  # ['bias', 'LayerNorm.weight']
# # will use default weight decay value
# optimizer_grouped_parameters = [
#     {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
#      "weight_decay": args.weight_decay},
#     {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
#      'weight_decay': 0.0}
# ]

optimizer = torch.optim.Adam(
    params=model.parameters(),
    lr=args.lr,
    betas=(0.0, 0.999),
    eps=1e-9,
    weight_decay=args.weight_decay,
)

lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="min",
    factor=0.5,
    patience=2,
    verbose=True,
    threshold=0.0001,
    threshold_mode="abs",
    cooldown=0,
    min_lr=0.00001,
    eps=1e-08,
)


# lr = args.lr
###############################################################################
# Training code
###############################################################################


def repackage_hidden(h):
    """Wraps hidden states in new Tensors, to detach them from their history."""

    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)


# get_batch subdivides the source data into chunks of length args.bptt.
# If source is equal to the example output of the batchify function, with
# a bptt-limit of 2, we'd get the following two Variables for i = 0:
# ┌ a g m s ┐ ┌ b h n t ┐
# └ b h n t ┘ └ c i o u ┘
# Note that despite the name of the function, the subdivison of data is not
# done along the batch dimension (i.e. dimension 1), since that was handled
# by the batchify function. The chunks are along dimension 0, corresponding
# to the seq_len dimension in the LSTM.


def get_batch(source, i):
    seq_len = min(args.bptt, len(source) - 1 - i)
    data = source[i : i + seq_len]
    target = source[i + 1 : i + 1 + seq_len].view(-1)
    return data, target


def evaluate(data_source):
    # Turn on evaluation mode which disables dropout.
    model.eval()
    total_loss = 0.0
    ntokens = len(corpus.dictionary)
    if args.model != "Transformer":
        hidden = model.init_hidden(eval_batch_size)
    with torch.no_grad():
        for i in range(0, data_source.size(0) - 1, args.bptt):
            data, targets = get_batch(data_source, i)
            if args.model == "Transformer":
                output = model(data)
            else:
                output, hidden = model(data, hidden)
                hidden = repackage_hidden(hidden)
            output_flat = output.view(-1, ntokens)
            total_loss += len(data) * criterion(output_flat, targets).item()
    return total_loss / (len(data_source) - 1)


def train():
    # Turn on training mode which enables dropout.
    model.train()
    total_loss = 0.0
    start_time = time.time()
    ntokens = len(corpus.dictionary)
    if args.model != "Transformer":
        hidden = model.init_hidden(args.batch_size)
    for batch, i in enumerate(range(0, train_data.size(0) - 1, args.bptt)):
        data, targets = get_batch(train_data, i)
        # Starting each batch, we detach the hidden state from how it was previously produced.
        # If we didn't, the model would try backpropagating all the way to start of the dataset.
        # model.zero_grad()
        optimizer.zero_grad()
        if args.model == "Transformer":
            output = model(data)
        else:
            hidden = repackage_hidden(hidden)
            output, hidden = model(data, hidden)
        loss = criterion(output.view(-1, ntokens), targets)
        loss.backward()

        # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        # for p in model.parameters():
        #     p.data.add_(-lr, p.grad.data)
        optimizer.step()

        total_loss += loss.item()

        if batch % args.log_interval == 0 and batch > 0:
            cur_loss = total_loss / args.log_interval
            elapsed = time.time() - start_time
            # lr {:04.4f} |
            logging.info(
                "| epoch {:3d} | {:5d}/{:5d} batches | ms/batch {:5.2f} | "
                "loss {:5.2f} | ppl {:8.2f}".format(
                    epoch,
                    batch,
                    len(train_data) // args.bptt,
                    elapsed * 1000 / args.log_interval,
                    cur_loss,
                    math.exp(cur_loss),
                )
            )
            total_loss = 0
            start_time = time.time()


def export_onnx(path, batch_size, seq_len):
    logging.info(
        "The model is also exported in ONNX format at {}".format(
            os.path.realpath(args.onnx_export)
        )
    )
    model.eval()
    dummy_input = (
        torch.LongTensor(seq_len * batch_size).zero_().view(-1, batch_size).to(device)
    )
    hidden = model.init_hidden(batch_size)
    torch.onnx.export(model, (dummy_input, hidden), path)


# Loop over epochs.
best_val_loss = None

# At any point you can hit Ctrl + C to break out of training early.
try:
    for epoch in range(1, args.epochs + 1):
        epoch_start_time = time.time()
        train()
        val_loss = evaluate(val_data)
        logging.info("-" * 89)
        logging.info(
            "| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | "
            "valid ppl {:8.2f}".format(
                epoch, (time.time() - epoch_start_time), val_loss, math.exp(val_loss)
            )
        )
        logging.info("-" * 89)
        # Save the model if the validation loss is the best we've seen so far.
        lr_scheduler.step(val_loss)
        if not best_val_loss or val_loss < best_val_loss:
            with open(args.save, "wb") as f:
                torch.save(
                    {
                        "step": epoch,
                        "model_state_dict": model.state_dict(),
                        # 'optimizer_state_dict': optimizer.state_dict()
                    },
                    f,
                )
            best_val_loss = val_loss
        # else:
        #     # Anneal the learning rate if no improvement has been seen in the validation dataset.
        #     # lr /= 4.0
        #     logger.warning("implement lr schedule to lower learning rate on no improvements")
except KeyboardInterrupt:
    logging.info("-" * 89)
    logging.info("Exiting from training early")

# Load the best saved model.
with open(args.save, "rb") as f:
    checkpoint = torch.load(f)
    model.load_state_dict(checkpoint["model_state_dict"])
    # after load the rnn params are not a continuous chunk of memory
    # this makes them a continuous chunk, and will speed up forward pass
    # Currently, only rnn model supports flatten_parameters function.
    if args.model in ["RNN_TANH", "RNN_RELU", "LSTM", "GRU"]:
        model.rnn.flatten_parameters()

# Run on test data.
test_loss = evaluate(test_data)
logging.info("=" * 89)
logging.info(
    "| End of training | test loss {:5.2f} | test ppl {:8.2f}".format(
        test_loss, math.exp(test_loss)
    )
)
logging.info("=" * 89)

if len(args.onnx_export) > 0:
    # Export the model in ONNX format.
    export_onnx(args.onnx_export, batch_size=1, seq_len=args.bptt)
