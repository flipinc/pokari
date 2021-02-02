import logging
from typing import Any, Optional, Union

import editdistance
import torch
from modules.transducer_decoder import TransducerDecoder
from pytorch_lightning.metrics import Metric


def gather_all_tensors(result: Union[torch.Tensor], group: Optional[Any] = None):
    """
    Function to gather all tensors from several ddp processes onto a list that
    is broadcasted to all processes

    Args:
        result: the value to sync
        group: the process group to gather results from. Defaults to
            all processes (world)

    Return:
        gathered_result: list with size equal to the process group where
            gathered_result[i] corresponds to result tensor from process i
    """
    if group is None:
        group = torch.distributed.group.WORLD

    # convert tensors to contiguous format
    result = result.contiguous()

    world_size = torch.distributed.get_world_size(group)

    gathered_result = [torch.zeros_like(result) for _ in range(world_size)]

    # sync and broadcast all
    torch.distributed.barrier(group=group)
    torch.distributed.all_gather(gathered_result, result, group)

    return gathered_result


def dim_zero_cat(x):
    x = x if isinstance(x, (list, tuple)) else [x]
    return torch.cat(x, dim=0)


def dim_zero_sum(x):
    return torch.sum(x, dim=0)


def dim_zero_mean(x):
    return torch.mean(x, dim=0)


class TransducerWER(Metric):
    """
    This metric computes numerator and denominator for Overall Word Error Rate (WER)
    between prediction and reference texts. When doing distributed training/evaluation
    the result of res=WER(predictions, targets, target_lengths) calls will be
    all-reduced between all workers using SUM operations. Here contains two numbers
    res=[wer_numerator, wer_denominator]. WER=wer_numerator/wer_denominator.

    If used with PytorchLightning LightningModule, include wer_numerator and
    wer_denominators inside validation_step results. Then aggregate (sum) then at the
    end of validation epoch to correctly compute validation WER.

    Example:
        def validation_step(self, batch, batch_idx):
            ...
            wer_num, wer_denom = self.__wer(predictions, transcript, transcript_len)
            return {
                'val_loss': loss_value,
                'val_wer_num': wer_num,
                'val_wer_denom': wer_denom
            }

        def validation_epoch_end(self, outputs):
            ...
            wer_num = torch.stack([x['val_wer_num'] for x in outputs]).sum()
            wer_denom = torch.stack([x['val_wer_denom'] for x in outputs]).sum()
            tensorboard_logs = {
                'validation_loss': val_loss_mean,
                'validation_avg_wer': wer_num / wer_denom
            }
            return {'val_loss': val_loss_mean, 'log': tensorboard_logs}

    Args:
        decoding: RNNTDecoding object that will perform autoregressive decoding of the
            RNNT model.
        batch_dim_index: Index of the batch dimension.
        use_cer: Whether to use Character Error Rate isntead of Word Error Rate.
        log_prediction: Whether to log a single decoded sample per call.

    Returns:
        res: a torch.Tensor object with two elements: [wer_numerator, wer_denominator].
            To correctly compute average
        text word error rate, compute wer=wer_numerator/wer_denominator
    """

    def __init__(
        self,
        decoder: TransducerDecoder,
        batch_dim_index=0,
        use_cer=False,
        log_prediction=True,
        dist_sync_on_step=False,
    ):
        super().__init__(dist_sync_on_step=dist_sync_on_step, compute_on_step=False)
        self.decoder = decoder
        self.batch_dim_index = batch_dim_index
        self.use_cer = use_cer
        self.log_prediction = log_prediction
        self.blank_id = self.decoder.blank_id
        self.labels_map = self.decoder.labels_map

        self.add_state(
            "scores", default=torch.tensor(0), dist_reduce_fx="sum", persistent=False
        )
        self.add_state(
            "words", default=torch.tensor(0), dist_reduce_fx="sum", persistent=False
        )

    def update(
        self,
        encoder_output: torch.Tensor,
        encoded_lengths: torch.Tensor,
        targets: torch.Tensor,
        target_lengths: torch.Tensor,
    ) -> torch.Tensor:
        words = 0.0
        scores = 0.0
        references = []
        with torch.no_grad():
            # prediction_cpu_tensor = tensors[0].long().cpu()
            targets_cpu_tensor = targets.long().cpu()
            tgt_lenths_cpu_tensor = target_lengths.long().cpu()

            # iterate over batch
            for ind in range(targets_cpu_tensor.shape[self.batch_dim_index]):
                tgt_len = tgt_lenths_cpu_tensor[ind].item()
                target = targets_cpu_tensor[ind][:tgt_len].numpy().tolist()

                reference = self.decoder.decode_tokens_to_str(target)
                references.append(reference)

            hypotheses, _ = self.decoder.generate_hypotheses(
                encoder_output, encoded_lengths
            )

        if self.log_prediction:
            logging.info("\n")
            logging.info(f"reference :{references[0]}")
            logging.info(f"predicted :{hypotheses[0]}")

        for h, r in zip(hypotheses, references):
            if self.use_cer:
                h_list = list(h)
                r_list = list(r)
            else:
                h_list = h.split()
                r_list = r.split()
            words += len(r_list)
            # Compute Levenshtein's distance
            scores += editdistance.eval(h_list, r_list)

        self.scores += torch.tensor(
            scores, device=self.scores.device, dtype=self.scores.dtype
        )
        self.words += torch.tensor(
            words, device=self.words.device, dtype=self.words.dtype
        )
        # return torch.tensor([scores, words]).to(predictions.device)

    def compute(self):
        wer = self.scores.float() / self.words
        return wer, self.scores.detach(), self.words.detach()
