import math
from contextlib import nullcontext
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from base_llama import LlamaConfig, LlamaPreTrainedModel
from rope import apply_rotary_emb
from utils import *


# Root Mean Square Layer Normalization (https://arxiv.org/abs/1910.07467)
# borrowed from the official Llama implementation:
# https://github.com/facebookresearch/llama/blob/main/llama/model.py
class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        """
        Initialize the RMSNorm normalization layer.

        Args:
            dim (int): The dimension of the input tensor.
            eps (float, optional): A small value added to the denominator for numerical stability. Default is 1e-6.

        Attributes:
            eps (float): A small value added to the denominator for numerical stability.
            weight (nn.Parameter): Learnable scaling parameter.

        """
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        """
        Compute the root mean square normalization. Use Equation 4 under
        Section 4 of https://arxiv.org/abs/1910.07467 as a reference. Add
        the given epsilon value (self.eps) to the tensor's norm (i.e. inside
        the square root in Equation 4) before normalizing the tensor.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The normalized tensor.
        """
        # TODO
        rms = torch.sqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x / rms

    def forward(self, x):
        """
        Apply the root mean square normalizer.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor after applying RMSNorm.

        """
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


class Attention(nn.Module):
    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.n_kv_heads = config.n_heads if config.n_kv_heads is None else config.n_kv_heads
        assert config.n_heads % self.n_kv_heads == 0
        model_parallel_size = 1
        self.n_local_heads = config.n_heads // model_parallel_size
        self.n_local_kv_heads = self.n_kv_heads // model_parallel_size
        self.n_rep = self.n_local_heads // self.n_local_kv_heads
        self.head_dim = config.dim // config.n_heads
        self.max_seq_len = config.max_seq_len
        self.compute_query = nn.Linear(config.dim, config.n_heads * self.head_dim, bias=False)
        self.compute_key = nn.Linear(config.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.compute_value = nn.Linear(config.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.compute_output = nn.Linear(config.n_heads * self.head_dim, config.dim, bias=False)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.dropout = config.dropout

    def compute_query_key_value_scores(self,
                                       query: torch.Tensor,
                                       key: torch.Tensor,
                                       value: torch.Tensor) -> torch.Tensor:
        '''
        Jointly compute Scaled Dot Product Attention (see Section 3.2.1 in
        https://arxiv.org/abs/1706.03762 for details). The query, key, and
        value tensors each have shape (bs, n_local_heads, seqlen, head_dim).
        An optimal implemention will jointly computing attention for multiple
        heads (n_local_heads of them) at once using matrix/tensor operations.

        Make sure to use attention_dropout (self.attn_dropout) on the computed
        attention matrix before applying it to the value tensor.
        '''
        # TODO
        # (bs, n_local_heads, seqlen, head_dim) -> (bs, n_local_heads, seqlen, seqlen)
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn = F.softmax(scores, dim=-1)
        attn = self.attn_dropout(attn)

        # (bs, n_local_heads, seqlen, seqlen) -> (bs, n_local_heads, seqlen, head_dim)
        output = torch.matmul(attn, value)
        return output

    def forward(self, x: torch.Tensor):
        '''
        Llama2 uses Grouped-Query Attention. The details of GQA are actually
        not critical to solving this assignment; you are simply asked to
        compute Scaled Dot Product Attention (see above for details). GQA is
        a memory optimization to compute multi-head attention efficiently. See
        Section 2.2 in https://arxiv.org/abs/2305.13245 or
        https://ai.plainenglish.io/understanding-llama2-kv-cache-grouped-query-attention-rotary-embedding-and-more-c17e5f49a6d7
        for details.
        '''
        batch_size, seqlen, _ = x.shape

        query = self.compute_query(x)
        key = self.compute_key(x)
        value = self.compute_value(x)
        query = query.view(batch_size, seqlen, self.n_local_heads, self.head_dim)
        key = key.view(batch_size, seqlen, self.n_local_kv_heads, self.head_dim)
        value = value.view(batch_size, seqlen, self.n_local_kv_heads, self.head_dim)

        # RoPE relative positional embeddings
        query, key = apply_rotary_emb(query, key, self.head_dim, self.max_seq_len)

        # Grouped multiquery attention: expand out keys and values.
        # Convert both to:
        # (bs, seqlen, n_local_heads, head_dim)
        key = torch.repeat_interleave(key, dim=2, repeats=self.n_rep)
        value = torch.repeat_interleave(value, dim=2, repeats=self.n_rep)

        # Make heads into a batch dimension
        # (bs, n_local_heads, seqlen, head_dim)
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)
        output = self.compute_query_key_value_scores(query, key, value)

        # Restore time as batch dimension and concat heads
        # (bs, seqlen, n_local_heads*head_dim)
        output = output.transpose(1, 2).contiguous().view(batch_size, seqlen, -1)

        # Final projection into the residual stream
        output = self.resid_dropout(self.compute_output(output))
        return output


class FeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, multiple_of: int, dropout: float):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = 4 * dim
            hidden_dim = int(2 * hidden_dim / 3)
            hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def SwiGLU(self, x: torch.Tensor) -> torch.Tensor:
        '''
        Compute the SwiGLU activation function (see Section 2 in
        https://arxiv.org/abs/2204.02311
        '''
        return F.silu(self.w1(x)) * self.w3(x)

    def forward(self, x):
        return self.dropout(self.w2(self.SwiGLU(x)))


class LlamaLayer(nn.Module):
    def __init__(self, layer_id: int, config: LlamaConfig):
        super().__init__()
        self.n_heads = config.n_heads
        self.dim = config.dim
        self.head_dim = config.dim // config.n_heads
        self.attention = Attention(config)
        self.feed_forward = FeedForward(
            dim=config.dim,
            hidden_dim=config.hidden_dim,  # type: ignore
            multiple_of=config.multiple_of,
            dropout=config.dropout,
        )
        self.layer_id = layer_id
        self.attention_norm = RMSNorm(config.dim, eps=config.layer_norm_eps)
        self.ffn_norm = RMSNorm(config.dim, eps=config.layer_norm_eps)

    def forward(self, x):
        '''
        This is the forward pass of the basic transformer building block. This is a
        modernized version of the block shown on the left of Figure 1 on
        https://arxiv.org/pdf/1706.03762.pdf.

        The transformer block should consist of:
        1) layer normalization of the input (via Root Mean Square layer normalization)
        2) self-attention on the layer-normalized input
        3) a residual connection (i.e., add the input to the output of the self-attention)
        3) layer normalization on the output of the self-attention
        4) a feed-forward network on the layer-normalized output of the self-attention
        5) add a residual connection from the unnormalized self-attention output to the
           output of the feed-forward network
        '''
        # TODO
        attn_input = self.attention_norm(x)
        attn_output = self.attention(attn_input)
        attn_output += x
        ffn_input = self.ffn_norm(attn_output)
        ffn_output = self.feed_forward(ffn_input)
        return attn_output + ffn_output


class Llama(LlamaPreTrainedModel):
    def __init__(self, config: LlamaConfig):
        '''
        You will probably never need to call this function, unless you decide
        to pretrain a Llama model from scratch.
        '''
        super().__init__(config)
        self.params = config
        self.vocab_size = config.vocab_size
        self.n_layers = config.n_layers

        self.tok_embeddings = nn.Embedding(config.vocab_size, config.dim)
        self.dropout = nn.Dropout(config.dropout)
        self.layers = torch.nn.ModuleList()
        for layer_id in range(config.n_layers):
            self.layers.append(LlamaLayer(layer_id, config))
        self.norm = RMSNorm(config.dim, eps=config.layer_norm_eps)
        self.output = nn.Linear(config.dim, config.vocab_size, bias=False)

        # share the unembedding parameters with the embedding parameters
        self.tok_embeddings.weight = self.output.weight  # https://paperswithcode.com/method/weight-tying

        # some useful precompute for the RoPE relative positional embeddings

        # init all weights
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith('w3.weight') or pn.endswith('compute_output.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layers))

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, tokens: torch.Tensor, targets: Optional[torch.Tensor] = None):
        _batch_size, seqlen = tokens.shape
        h = self.tok_embeddings(tokens)
        h = self.dropout(h)

        for layer in self.layers:
            h = layer(h)
        h = self.norm(h)

        if targets is not None:
            # if we are given some desired targets also calculate the loss
            logits = self.output(h)  # (bs, seqlen, vocab_size)
        else:
            # inference-time mini-optimization: only forward the output on the very last position
            logits = self.output(h[:, [-1], :])  # note: using list [-1] to preserve the time dim

        return logits, h

    @torch.inference_mode()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_p=0.9):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        We perform this generation using basic temperature sampling with nucleus sampling (i.e.
        limiting ourselves to sampling from the most probable tokens with cumulative probability
        just less than top_p at each timestep). Most likely you'll want to make sure to be in
        model.eval() mode of operation for this. Also note this is a super inefficient version of
        sampling with no key/value cache, but you are free to add any optimizations on top of this.
        """
        for _ in range(max_new_tokens):
            # If the sequence context is growing too long we must crop it at block_size
            idx_cond = (idx if idx.size(1) <= self.params.max_seq_len else
                        idx[:, -self.params.max_seq_len:])
            # Forward the model to get the logits for the index in the sequence
            logits, _ = self(idx_cond)
            # Crop to just the final time step
            # (bs, seqlen, vocab_size) -> (bs, vocab_size)
            logits = logits[:, -1, :]
            # TODO
            # raise NotImplementedError

            if temperature == 0.0:
                # Select the single most likely index
                idx_next = torch.argmax(logits, dim=-1, keepdim=True)
            else:
                '''
                Perform temperature sampling with top-p (nucleus) sampling:
                1) Scale the logits with the temperature followed by normalization using Softmax.
                2) Sort the tokens according to their normalized logits
                3) Identify the tokens within the top-p cumulative. Make sure to handle any edge cases here.
                4) Filter and normalize the resulting probabilities.
                5) Sample from this scaled probability distribution.
                '''
                probs = F.softmax(logits / temperature, dim=-1)
                sort_probs, sort_prob_indices = torch.sort(probs, dim=-1, descending=True)
                # Identify tokens within top-p cumulative
                cum_probs = torch.cumsum(sort_probs, dim=-1)
                keep_mask = cum_probs < top_p
                keep_mask[:, 0] = True  # Ensure at least one token remains
                sort_probs[~keep_mask] = 0.0
                # Normalize result probabilities
                sort_probs.div_(sort_probs.sum(dim=-1, keepdim=True))
                # Sample from scaled probability distribution
                sample_indices = torch.multinomial(sort_probs, num_samples=1)
                idx_next = torch.gather(sort_prob_indices, -1, sample_indices)

            # Append sampled index to the running sequence and continue
            idx = torch.cat((idx, idx_next), dim=1)  # type: ignore

        return idx


def load_pretrained(checkpoint):
    # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1', etc.
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    #dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32' or 'bfloat16' or 'float16'
    dtype = "float32"

    torch.backends.cuda.matmul.allow_tf32 = True  # allow tf32 on matmul
    torch.backends.cudnn.allow_tf32 = True  # allow tf32 on cudnn
    device_type = 'cuda' if 'cuda' in device else 'cpu'  # for later use in torch.autocast
    ptdtype = {'float32': torch.float32,
               'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
    ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(
        device_type=device_type, dtype=ptdtype)

    # init from a model saved in a specific directory
    checkpoint_dict = torch.load(checkpoint, map_location=device)
    config = LlamaConfig(**checkpoint_dict['model_args'])
    model = Llama(config)
    state_dict = checkpoint_dict['model']
    unwanted_prefix = '_orig_mod.'
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict, strict=False)
    return model
