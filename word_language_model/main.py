# coding: utf-8
import argparse
import datetime
import logging
import math
import os
import time

import numpy as np
import torch
import torch.jit
import torch.nn as nn
import torch.onnx
import torch.optim

import data
import model

parser = argparse.ArgumentParser(description="PyTorch Wikitext-2 RNN/LSTM Language Model")

parser.add_argument("--data", type=str, default="./data/ptb", help="location of the data corpus")
parser.add_argument(
    "--model",
    type=str,
    default="LSTM",
    help="type of recurrent net (RNN_TANH, RNN_RELU, LSTM, GRU, Transformer, custom_LSTM)")
parser.add_argument("--emsize", type=int, default=200, help="size of word embeddings")
parser.add_argument(
    "--nhid", type=int, default=200, help="number of hidden units per layer"
)
parser.add_argument("--nlayers", type=int, default=2, help="number of layers")

parser.add_argument("--lr", type=float, default=0.001, help="initial learning rate")
parser.add_argument("--clip", type=float, default=10, help="gradient clipping")
parser.add_argument("--weight_decay", type=float,
                    default=0.0, help="weight decay used for regularizing the model")

parser.add_argument("--l2_penalty", type=float,
                    default=0.0, help="l2 penalty added to the training loss")

parser.add_argument(
    "--weight_decay_mode", type=str, default="all", help="all, all_except_bias, all_except_bias_embedding"
)

parser.add_argument("--epochs", type=int, default=40, help="upper epoch limit")
parser.add_argument(
    "--batch_size", type=int, default=64, metavar="N", help="batch size"
)
parser.add_argument("--bptt", type=int, default=35, help="sequence length")

parser.add_argument(
    "--dropout",
    type=float,
    default=0.0,
    help="dropout applied to layers only in the Transformer model(0 = no dropout)",
)

parser.add_argument(
    "--input_dropout",
    type=float,
    default=0.0,
    help="dropout applied to the input of LSTM",
)

parser.add_argument(
    "--input_noise_std",
    type=float,
    default=0.0,
    help="Standard deviation of the noise added to the input",
)

parser.add_argument(
    "--recurrent_dropout",
    type=float,
    default=0.0,
    help="dropout applied to the recurrent state of of each LSTM layer",
)
parser.add_argument(
    "--inter_layer_dropout",
    type=float,
    default=0.0,
    help="dropout applied to inter layer connection of each LSTM layer",
)
parser.add_argument(
    "--output_dropout",
    type=float,
    default=0.0,
    help="dropout applied to the output of the LSTM",
)
parser.add_argument(
    "--output_noise_std",
    type=float,
    default=0.0,
    help="Standard deviation of the noise applied to the output of the LSTM",
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
    "--log-interval", type=int, default=207, metavar="N", help="report interval"
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
parser.add_argument("--optimizer", type=str, default="adam")

parser.add_argument("--lr_schedule_mode", type=str, default="min")
parser.add_argument("--lr_schedule_factor", type=float, default=0.5)
parser.add_argument("--lr_schedule_patience", type=int, default=15)
parser.add_argument("--lr_schedule_verbose", type=bool, default=True)
parser.add_argument("--lr_schedule_threshold", type=float, default=.01)
parser.add_argument("--lr_schedule_threshold_mode", type=str, default="abs")
parser.add_argument("--lr_schedule_cooldown", type=int, default=0)
parser.add_argument("--lr_schedule_min_lr", type=float, default=0.0005)
parser.add_argument("--lr_schedule_eps", type=float, default=1e-08)

parser.add_argument("--lr_asgd", type=float, default=0.01)

parser.add_argument("--up_project_embedding", type=bool, default=False)
parser.add_argument("--up_project_hidden", type=bool, default=False)

parser.add_argument("--drop_state_probability", type=float, default=0.01)

args = parser.parse_args()


def _default_output_dir():
    try:
        dataset_name = args.data
    except ValueError:
        dataset_name = "unknown"
    dir_name = "{model_name}_{dataset_name}_{timestamp}".format(
        model_name=args.model,
        dataset_name=dataset_name.replace("/", "_"),
        timestamp=datetime.datetime.now().strftime("%Y%m%d_%H%M%S"),
    )
    dir_path = os.path.join(".", "out", dir_name)
    return dir_path


output_dir = _default_output_dir()
output_dir = os.path.expanduser(output_dir)
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

logging.basicConfig(format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                    datefmt='%H:%M:%S',
                    level=logging.DEBUG)
logging.getLogger().setLevel(logging.DEBUG)

file_handler = logging.FileHandler(output_dir + '/output.log')
file_handler.setLevel(logging.DEBUG)
logging.getLogger().addHandler(file_handler)

logging.info("Output dir: {}".format(output_dir))
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
test_batch_size = 1

train_data = batchify(corpus.train, args.batch_size)
# train_eval_data = batchify(corpus.train, eval_batch_size)
val_data = batchify(corpus.valid, eval_batch_size)
test_data = batchify(corpus.test, test_batch_size)


###############################################################################
# Build the model
###############################################################################

def get_ntokens(corpus):
    n = len(corpus.dictionary)
    if n % 8 != 0:
        n = (n // 8 + 1) * 8
    return n


ntokens = get_ntokens(corpus)
if args.model == "Transformer":
    model = model.TransformerModel(
        ntokens, args.emsize, args.nhead, args.nhid, args.nlayers, args.dropout
    )
else:
    if args.dropout > 0.0:
        logging.warning("dropout argument is not used in the LSTM models")

    model = model.RNNModel(
        rnn_type=args.model,
        num_tokens=ntokens,
        embedding_size=args.emsize,
        hidden_size=args.nhid,
        num_layers=args.nlayers,
        input_dropout=args.input_dropout,
        input_noise_std=args.input_noise_std,
        recurrent_dropout=args.recurrent_dropout,
        inter_layer_dropout=args.inter_layer_dropout,
        output_dropout=args.output_dropout,
        output_noise_std=args.output_noise_std,
        up_project_embedding=args.up_project_embedding,
        up_project_hidden=args.up_project_hidden,
        tie_weights=args.tied,
        lstm_skip_connection=args.lstm_skip_connection,
        drop_state_probability=args.drop_state_probability
    )

if args.jit_forward:
    model = torch.jit.script(model)

model = model.to(device)
criterion = nn.CrossEntropyLoss()
nll_loss = nn.NLLLoss()
n = sum(p.numel() for p in model.parameters() if p.requires_grad)

logging.info("Number of trainable parameters: {}".format(n))
logging.info('Args: {}'.format(args.__str__()))


def get_param_weight_decay_dict(param_group_name_list):
    return [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in param_group_name_list)],
         "weight_decay": args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in param_group_name_list)],
         'weight_decay': 0.0}
    ]


no_decay_weights = []
if args.weight_decay_mode == "all":
    pass
elif args.weight_decay_mode == "all_except_bias":
    no_decay_weights = ["bias"]
elif args.weight_decay_mode == "all_except_bias_embedding":
    no_decay_weights = ["bias", "encoder"]
elif args.weight_decay_mode == "only_embedding":
    no_decay_weights = [n for n, p in model.named_parameters() if "encoder" in n]
else:
    raise ValueError("Invalid argument for weight_decay_mode: {}".format(args.weight_decay_mode))

optimizer_grouped_parameters = get_param_weight_decay_dict(no_decay_weights)


if args.optimizer == "adam":
    optimizer_class = torch.optim.Adam
elif args.optimizer == "adam_w":
    optimizer_class = torch.optim.AdamW
else:
    raise ValueError("Invalid args for optimizer {}".format(args.optimizer))


betas = (0.0, 0.999)
epsilon = 1e-08

optimizer = optimizer_class(
        params=optimizer_grouped_parameters,
        lr=args.lr,
        betas=betas,
        eps=epsilon)


lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode=args.lr_schedule_mode,
    factor=args.lr_schedule_factor,
    patience=args.lr_schedule_patience,
    verbose=args.lr_schedule_verbose,
    threshold=args.lr_schedule_threshold,
    threshold_mode=args.lr_schedule_threshold_mode,
    cooldown=args.lr_schedule_cooldown,
    min_lr=args.lr_schedule_min_lr,
    eps=args.lr_schedule_eps,
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
    data = source[i: i + seq_len]
    target = source[i + 1: i + 1 + seq_len].view(-1)
    return data, target


def evaluate(data_source, batch_size, tune_softmax=False):
    # Turn on evaluation mode which disables dropout.
    model.eval()
    total_loss = 0.0
    ntokens = get_ntokens(corpus)
    if args.model != "Transformer":
        hidden = model.init_hidden(batch_size)
    # probabilities = []

    if tune_softmax:
        softmax_temps_list = np.arange(0.8, 1.1, 0.02)
    else:
        softmax_temps_list = [1]
    loss_dict = {}
    for softmax_temp in softmax_temps_list:
        loss_dict[softmax_temp] = 0.

    with torch.no_grad():
        for i in range(0, data_source.size(0) - 1, args.bptt):
            data, targets = get_batch(data_source, i)
            if args.model == "Transformer":
                output = model(data)
            else:
                output, hidden = model(data, hidden)
                hidden = repackage_hidden(hidden)
            output_flat = output.view(-1, ntokens)

            for softmax_temp in softmax_temps_list:
                output_flat_tuned = torch.exp(output_flat / softmax_temp)
                output_flat_tuned_sum = output_flat_tuned.sum(dim=1).unsqueeze(dim=1).expand(-1,
                                                                                             output_flat_tuned.size(1))
                output_tuned_prob = torch.log(output_flat_tuned / output_flat_tuned_sum)

                loss_dict[softmax_temp] += len(data) * nll_loss(output_tuned_prob, targets).item()
                # output_prob_flat = output_flat.softmax(dim=1)
                # probabilities += [output_prob_flat[i][targets[i]].item() for i in range(targets.size(0))]

            # total_loss += len(data) * criterion(output_flat, targets).item()

    # probabilities = torch.tensor(probabilities)
    # min_prob = torch.min(probabilities).item()
    # max_prob = torch.max(probabilities).item()
    # median_prob = torch.median(probabilities).item()
    # var, mean = torch.var_mean(probabilities)
    # logging.info("mean: {}, var: {}, median: {}".format(mean.item(), var.item(), median_prob))
    temp, loss = min(loss_dict.items(), key=lambda x: x[1])

    if tune_softmax:
        logging.info("Optimal Temp: {:5.2f}".format(temp))
    return loss / (len(data_source) - 1)


def train():
    # Turn on training mode which enables dropout.
    model.train()
    total_loss = 0.0
    start_time = time.time()
    ntokens = get_ntokens(corpus)
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

        if args.l2_penalty > 0.0:
            loss = loss + args.l2_penalty * torch.tensor(
                [p.norm() for p in model.parameters() if p.requires_grad]).sum()

        loss.backward()

        # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        # for p in model.parameters():
        #     p.data.add_(-lr, p.grad.data)
        optimizer.step()
        if isinstance(lr_scheduler, torch.optim.lr_scheduler.CyclicLR):
            lr_scheduler.step()

        # total_loss += loss.item()
        # if batch % args.log_interval == 0 and batch > 0:
        #     cur_loss = total_loss / args.log_interval
        #     elapsed = time.time() - start_time
        #     # lr {:04.4f} |
        #     logging.info(
        #         "| epoch {:3d} | {:5d}/{:5d} batches | ms/batch {:5.2f} | "
        #         "loss {:5.2f} | ppl {:8.2f}".format(
        #             epoch,
        #             batch,
        #             len(train_data) // args.bptt,
        #             elapsed * 1000 / args.log_interval,
        #             cur_loss,
        #             math.exp(cur_loss),
        #         )
        #     )
        #     total_loss = 0
        #     start_time = time.time()


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


def print_model_info(model):
    for n, p in model.named_parameters():
        var, mean = torch.var_mean(p)
        logging.info("{}_mean: {}".format(n, mean.item()))
        logging.info("{}_var: {}".format(n, var.item()))


# Loop over epochs.
best_val_loss = None
# At any point you can hit Ctrl + C to break out of training early.
try:
    for epoch in range(1, args.epochs + 1):
        epoch_start_time = time.time()
        train()
        # print_model_info(model)
        if isinstance(optimizer, torch.optim.ASGD):
            tmp = {}
            for prm in model.parameters():
                tmp[prm] = prm.data.clone()
                prm.data = optimizer.state[prm]['ax'].clone()

        train_loss = evaluate(train_data, args.batch_size)
        val_loss = evaluate(val_data, eval_batch_size, tune_softmax=True)

        if isinstance(optimizer, torch.optim.ASGD):
            for prm in model.parameters():
                prm.data = tmp[prm].clone()

        logging.info("-" * 89)
        logging.info(
            "| end of epoch {:3d} | time: {:5.2f}s | "
            "train loss {:5.2f} | train ppl {:8.2f} | "
            "valid loss {:5.2f} | valid ppl {:8.2f}".format(
                epoch, (time.time() - epoch_start_time),
                train_loss, math.exp(train_loss),
                val_loss, math.exp(val_loss)
            )
        )
        logging.info("-" * 89)
        # Save the model if the validation loss is the best we've seen so far.
        if isinstance(lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            lr_scheduler.step(val_loss)
        if not best_val_loss or val_loss < best_val_loss:
            file_name = output_dir + "/" + args.save
            logging.info("Saving model: {}".format(file_name))
            with open(file_name, "wb") as file:
                torch.save(
                    {"step": epoch,
                     "model_state_dict": model.state_dict(),
                     'optimizer_state_dict': optimizer.state_dict()},
                    file)
            # last_epoch_model_saved = epoch
            best_val_loss = val_loss

        # if isinstance(lr_scheduler,
        #               torch.optim.lr_scheduler.ReduceLROnPlateau) and (
        #         lr_scheduler.optimizer.param_groups[0]["lr"] == args.lr_schedule_min_lr):
        #     logging.info("Switching to CyclicLR")
        #     lr_scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer,
        #                                                      base_lr=args.lr_schedule_min_lr * 1e-1,
        #                                                      max_lr=args.lr_schedule_min_lr,
        #                                                      step_size_up=500,
        #                                                      cycle_momentum=False)

        if isinstance(optimizer, (torch.optim.Adam, torch.optim.AdamW)) and (epoch > .8 * args.epochs or
                                                                             lr_scheduler.optimizer.param_groups[0][
                                                                                 "lr"] == args.lr_schedule_min_lr):
            logging.info('Switching to ASGD')
            optimizer = torch.optim.ASGD(
                params=model.parameters(),
                lr=args.lr_asgd,
                t0=0,
                lambd=0.,
                weight_decay=args.weight_decay,
            )
            lr_scheduler = None
            # lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            #     optimizer,
            #     mode=args.lr_schedule_mode,
            #     factor=args.lr_schedule_factor,
            #     patience=args.lr_schedule_patience,
            #     verbose=args.lr_schedule_verbose,
            #     threshold=args.lr_schedule_threshold,
            #     threshold_mode=args.lr_schedule_threshold_mode,
            #     cooldown=args.lr_schedule_cooldown,
            #     min_lr=args.lr_asgd * 1e-1,
            #     eps=args.lr_schedule_eps,
            # )
        # else:
        #     # Anneal the learning rate if no improvement has been seen in the validation dataset.
        #     # lr /= 4.0
        #     logger.warning("implement lr schedule to lower learning rate on no improvements")
except KeyboardInterrupt:
    logging.info("-" * 89)
    logging.info("Exiting from training early")

# Load the best saved model.
file_name = output_dir + "/" + args.save
with open(file_name, "rb") as file:
    checkpoint = torch.load(file)
    model.load_state_dict(checkpoint["model_state_dict"])
    # after load the rnn params are not a continuous chunk of memory
    # this makes them a continuous chunk, and will speed up forward pass
    # Currently, only rnn model supports flatten_parameters function.
    if args.model in ["RNN_TANH", "RNN_RELU", "LSTM", "GRU"]:
        model.rnn.flatten_parameters()

# Run on test data.
test_loss = evaluate(test_data, test_batch_size, tune_softmax=True)
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
