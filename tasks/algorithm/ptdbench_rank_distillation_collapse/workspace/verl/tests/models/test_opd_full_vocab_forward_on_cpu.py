# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""CPU tests for the full-vocabulary OPD modes of the fused torch-backend forward.

The frozen side's ``opd_return_hidden`` must hand back exactly the LM-head input,
and the student's ``opd_target`` path must reproduce the dense logits' log-probs
and the reverse KL to a teacher rebuilt from that hidden.
"""

import pytest
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, Qwen3Config, Qwen3MoeConfig

from verl.models.transformers.dense_common import forward_with_torch_backend
from verl.trainer.opd.full_vocab import FrozenScorer, FullVocabTarget

VOCAB = 97


def _tiny(config_cls, seed):
    kwargs = dict(
        vocab_size=VOCAB,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        max_position_embeddings=128,
        tie_word_embeddings=False,
    )
    if config_cls is Qwen3MoeConfig:
        kwargs.update(num_experts=4, num_experts_per_tok=2, moe_intermediate_size=16, decoder_sparse_step=1)
    torch.manual_seed(seed)
    model = AutoModelForCausalLM.from_config(config_cls(**kwargs), attn_implementation="eager").float().eval()
    return model


@pytest.mark.parametrize("config_cls", [Qwen3Config, Qwen3MoeConfig])
def test_hidden_and_target_modes(config_cls):
    student = _tiny(Qwen3Config, seed=0)
    teacher = _tiny(config_cls, seed=1)
    input_ids = torch.randint(0, VOCAB, (1, 13), generator=torch.Generator().manual_seed(2))
    labels = torch.roll(input_ids, shifts=-1, dims=-1)

    with torch.no_grad():
        t_out = forward_with_torch_backend(teacher, input_ids=input_ids, return_dict=True, opd_return_hidden=True)
        t_hidden = t_out.lm_head_hidden.squeeze(0)
        t_logits = teacher(input_ids=input_ids).logits.squeeze(0)
    torch.testing.assert_close(t_hidden @ teacher.lm_head.weight.t(), t_logits, rtol=1e-4, atol=1e-4)

    row_mask = torch.zeros(13, dtype=torch.bool)
    row_mask[5:12] = True
    target = FullVocabTarget([FrozenScorer(t_hidden, teacher.lm_head.weight.detach())], row_mask=row_mask, chunk_size=4)
    out = forward_with_torch_backend(
        student, input_ids=input_ids, return_dict=True, shift_labels=labels, opd_target=target
    )
    (out.opd_rkl * row_mask).sum().backward()
    grad_kernel = student.lm_head.weight.grad.clone()
    student.zero_grad()

    s_logits = student(input_ids=input_ids).logits.squeeze(0)
    logp = F.log_softmax(s_logits, dim=-1)
    logq = F.log_softmax(t_logits, dim=-1)
    ref_log_probs = logp.gather(-1, labels.squeeze(0).unsqueeze(-1)).squeeze(-1)
    ref_rkl = (logp.exp() * (logp - logq)).sum(-1) * row_mask
    torch.testing.assert_close(out.log_probs.squeeze(0), ref_log_probs, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(out.opd_rkl.squeeze(0), ref_rkl, rtol=1e-4, atol=1e-4)
    ref_rkl.sum().backward()
    torch.testing.assert_close(grad_kernel, student.lm_head.weight.grad, rtol=1e-4, atol=1e-4)
