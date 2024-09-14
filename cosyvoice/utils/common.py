# Copyright (c) 2020 Mobvoi Inc (Binbin Zhang)
#               2024 Alibaba Inc (authors: Xiang Lyu)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# Modified from ESPnet(https://github.com/espnet/espnet)
"""Unility functions for Transformer."""

from typing import List

import torch
import logging


IGNORE_ID = -1

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def pad_list(xs: List[torch.Tensor], pad_value: int):
    """Perform padding for the list of tensors.

    Args:
        xs (List): List of Tensors [(T_1, `*`), (T_2, `*`), ..., (T_B, `*`)].
        pad_value (float): Value for padding.

    Returns:
        Tensor: Padded tensor (B, Tmax, `*`).

    Examples:
        >>> x = [torch.ones(4), torch.ones(2), torch.ones(1)]
        >>> x
        [tensor([1., 1., 1., 1.]), tensor([1., 1.]), tensor([1.])]
        >>> pad_list(x, 0)
        tensor([[1., 1., 1., 1.],
                [1., 1., 0., 0.],
                [1., 0., 0., 0.]])

    """
    max_len = max([len(item) for item in xs])
    batchs = len(xs)
    ndim = xs[0].ndim
    if ndim == 1:
        pad_res = torch.zeros(batchs,
                              max_len,
                              dtype=xs[0].dtype,
                              device=xs[0].device)
    elif ndim == 2:
        pad_res = torch.zeros(batchs,
                              max_len,
                              xs[0].shape[1],
                              dtype=xs[0].dtype,
                              device=xs[0].device)
    elif ndim == 3:
        pad_res = torch.zeros(batchs,
                              max_len,
                              xs[0].shape[1],
                              xs[0].shape[2],
                              dtype=xs[0].dtype,
                              device=xs[0].device)
    else:
        raise ValueError(f"Unsupported ndim: {ndim}")
    pad_res.fill_(pad_value)
    for i in range(batchs):
        pad_res[i, :len(xs[i])] = xs[i]
    return pad_res


def th_accuracy(pad_outputs: torch.Tensor, pad_targets: torch.Tensor,
                ignore_label: int) -> torch.Tensor:
    """Calculate accuracy.

    Args:
        pad_outputs (Tensor): Prediction tensors (B * Lmax, D).
        pad_targets (LongTensor): Target label tensors (B, Lmax).
        ignore_label (int): Ignore label id.

    Returns:
        torch.Tensor: Accuracy value (0.0 - 1.0).

    """
    pad_pred = pad_outputs.view(pad_targets.size(0), pad_targets.size(1),
                                pad_outputs.size(1)).argmax(2)
    mask = pad_targets != ignore_label
    numerator = torch.sum(
        pad_pred.masked_select(mask) == pad_targets.masked_select(mask))
    denominator = torch.sum(mask)
    return (numerator / denominator).detach()


def get_padding(kernel_size, dilation=1):
    return int((kernel_size * dilation - dilation) / 2)


def init_weights(m, mean=0.0, std=0.01):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        m.weight.data.normal_(mean, std)


# Repetition Aware Sampling in VALL-E 2
def ras_sampling(weighted_scores, decoded_tokens, sampling, top_p=0.8, top_k=25, win_size=10, tau_r=0.1):
    try:
        top_ids = nucleus_sampling(weighted_scores, top_p=top_p, top_k=top_k)
        rep_num = (torch.tensor(decoded_tokens[-win_size:]).to(weighted_scores.device) == top_ids).sum().item()
        if rep_num >= win_size * tau_r:
            top_ids = random_sampling(weighted_scores, decoded_tokens, sampling)
    except Exception as e:
        print(f"Error in nucleus sampling: {e}. Falling back to random sampling.")
        top_ids = random_sampling(weighted_scores, decoded_tokens, sampling)
    return top_ids


def nucleus_sampling(weighted_scores, top_p=0.8, top_k=25):
    # Áp dụng log_softmax để tính xác suất một cách ổn định hơn
    log_probs = torch.nn.functional.log_softmax(weighted_scores, dim=0)
    probs = torch.exp(log_probs)
    
    # Loại bỏ các giá trị không hợp lệ
    probs = torch.nan_to_num(probs, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Đảm bảo tổng xác suất là 1
    probs = probs / probs.sum()
    
    sorted_probs, sorted_indices = probs.sort(descending=True)
    cumulative_probs = torch.cumsum(sorted_probs, dim=0)
    
    # Chọn top-k và top-p
    sorted_indices_to_remove = cumulative_probs > top_p
    sorted_indices_to_remove[..., :min(top_k, sorted_indices_to_remove.size(-1))] = 0
    
    indices_to_remove = sorted_indices_to_remove.scatter(0, sorted_indices, sorted_indices_to_remove)
    probs[indices_to_remove] = 0
    
    # Chuẩn hóa lại xác suất sau khi loại bỏ
    probs = probs / probs.sum()
    
    # Kiểm tra xem có bất kỳ giá trị nào không hợp lệ không
    if torch.isnan(probs).any() or torch.isinf(probs).any() or (probs < 0).any():
        print("Warning: Invalid probabilities detected. Using uniform distribution.")
        probs = torch.ones_like(probs) / probs.size(0)
    
    # Lấy mẫu
    top_ids = torch.multinomial(probs, 1)
    
    return top_ids


def random_sampling(weighted_scores, decoded_tokens, sampling):
    top_ids = weighted_scores.softmax(dim=0).multinomial(1, replacement=True)
    return top_ids


def fade_in_out(fade_in_mel, fade_out_mel, window):
    device = fade_in_mel.device
    fade_in_mel, fade_out_mel = fade_in_mel.cpu(), fade_out_mel.cpu()
    mel_overlap_len = int(window.shape[0] / 2)
    fade_in_mel[:, :, :mel_overlap_len] = fade_in_mel[:, :, :mel_overlap_len] * window[:mel_overlap_len] + \
        fade_out_mel[:, :, -mel_overlap_len:] * window[mel_overlap_len:]
    return fade_in_mel.to(device)
