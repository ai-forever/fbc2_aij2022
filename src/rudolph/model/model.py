# -*- coding: utf-8 -*-
import torch
import torch.nn.functional as F
from einops import rearrange

from .utils import init_method_normal
from .transformer import SparseTransformer


class ruDolphModel(torch.nn.Module):
    def __init__(self,
                 device,
                 num_layers,
                 vocab_size,
                 hidden_size,
                 num_attention_heads,
                 embedding_dropout_prob,
                 attention_dropout_prob,
                 output_dropout_prob,
                 l_text_seq_length=64,
                 r_text_seq_length=64,
                 kernel_size=7,
                 last_kernel_size=9,
                 image_tokens_per_dim=16,
                 image_vocab_size=8192,
                 text_special_tokens=0,
                 image_special_tokens=0,
                 cogview_sandwich_layernorm=True,
                 cogview_pb_relax=True,
                 is_bool_mask=True,
                 mlp_activation='gelu_jit',
                 gradient_checkpointing=None):
        super(ruDolphModel, self).__init__()
        self.device = device
        self.image_tokens_per_dim = image_tokens_per_dim
        self.image_seq_length = image_tokens_per_dim ** 2
        self.l_text_seq_length = l_text_seq_length
        self.r_text_seq_length = r_text_seq_length
        self.total_seq_length = self.l_text_seq_length + self.image_seq_length + self.r_text_seq_length
        self.text_special_tokens = text_special_tokens
        self.image_special_tokens = image_special_tokens
        vocab_size = vocab_size + text_special_tokens
        image_vocab_size = image_vocab_size + image_special_tokens
        self.total_vocab_size = vocab_size + image_vocab_size
        self.vocab_size = vocab_size
        self.gradient_checkpointing = gradient_checkpointing
        self.kernel_size = kernel_size
        self.last_kernel_size = last_kernel_size
        init_method = init_method_normal(std=0.02)

        self.text_embeddings = torch.nn.Embedding(vocab_size, hidden_size)
        self.image_embeddings = torch.nn.Embedding(image_vocab_size, hidden_size)

        # Position embedding (serial).
        self.l_text_pos_embeddings = torch.nn.Embedding(l_text_seq_length + 1, hidden_size)
        self.r_text_pos_embeddings = torch.nn.Embedding(r_text_seq_length + 1, hidden_size)
        self.image_row_embeddings = torch.nn.Embedding(image_tokens_per_dim, hidden_size)
        self.image_col_embeddings = torch.nn.Embedding(image_tokens_per_dim, hidden_size)
        init_method(self.l_text_pos_embeddings.weight)
        init_method(self.r_text_pos_embeddings.weight)
        init_method(self.image_row_embeddings.weight)
        init_method(self.image_col_embeddings.weight)

        self.to_logits = torch.nn.Sequential(
            torch.nn.LayerNorm(hidden_size),
            torch.nn.Linear(hidden_size, self.total_vocab_size),
        )

        # Embeddings dropout
        self.embedding_dropout = torch.nn.Dropout(embedding_dropout_prob)

        # Transformer
        self.transformer = SparseTransformer(
            num_layers,
            hidden_size,
            num_attention_heads,
            attention_dropout_prob,
            output_dropout_prob,
            l_text_seq_length=l_text_seq_length,
            r_text_seq_length=r_text_seq_length,
            kernel_size=kernel_size,
            last_kernel_size=last_kernel_size,
            image_tokens_per_dim=image_tokens_per_dim,
            cogview_sandwich_layernorm=cogview_sandwich_layernorm,
            cogview_pb_relax=cogview_pb_relax,
            mlp_activation=mlp_activation,
            is_bool_mask=is_bool_mask,
        )

    def get_param(self, item):
        return getattr(self, item)

    def get_image_pos_embeddings(self, image_input_ids, device, past_length=0):
        input_shape = image_input_ids.size()
        row_ids = torch.div(
            torch.arange(past_length, input_shape[-1] + past_length, dtype=torch.long, device=self.device),
            self.image_tokens_per_dim,
            rounding_mode='trunc'
        )
        row_ids = row_ids.unsqueeze(0).view(-1, input_shape[-1])
        col_ids = torch.arange(past_length, input_shape[-1] + past_length,
                               dtype=torch.long, device=device) % self.image_tokens_per_dim
        col_ids = col_ids.unsqueeze(0).view(-1, input_shape[-1])
        return self.image_row_embeddings(row_ids) + self.image_col_embeddings(col_ids)

    def forward(
        self,
        input_ids,
        attention_mask,
        return_loss=False,
        use_cache=False,
        cache=None,
        lt_loss_weight=1,
        img_loss_weight=7,
        rt_loss_weight=1,
        return_hidden_states=False,
    ):
        device = input_ids.device
        l_text = input_ids[:, :self.l_text_seq_length]
        l_text_range = torch.arange(l_text.shape[1])
        l_text_range += (self.vocab_size - self.l_text_seq_length)
        l_text_range = l_text_range.to(device)
        l_text = torch.where(l_text == 0, l_text_range, l_text)
        l_text_pos = self.l_text_pos_embeddings(torch.arange(l_text.shape[1], device=device))
        l_text_embeddings = self.text_embeddings(l_text) + l_text_pos

        use_image = input_ids.shape[1] > self.l_text_seq_length
        use_r_text = input_ids.shape[1] > self.l_text_seq_length + self.image_seq_length

        embeddings = [l_text_embeddings]
        if use_image:
            image_input_ids = input_ids[:, self.l_text_seq_length:self.l_text_seq_length + self.image_seq_length]
            img_pos = self.get_image_pos_embeddings(image_input_ids, past_length=0, device=device)
            image_embeddings = self.image_embeddings(image_input_ids) + img_pos
            embeddings.append(image_embeddings)

        if use_r_text:
            r_text = input_ids[:, self.l_text_seq_length + self.image_seq_length:]
            r_text_pos = self.r_text_pos_embeddings(torch.arange(r_text.shape[1], device=device))
            r_text_embeddings = self.text_embeddings(r_text) + r_text_pos
            embeddings.append(r_text_embeddings)

        embeddings = torch.cat(embeddings, dim=1)

        alpha = 0.1
        embeddings = embeddings * alpha + embeddings.detach() * (1 - alpha)

        attention_mask = attention_mask[:, :, :embeddings.shape[1], :embeddings.shape[1]]
        transformer_output, present_cache, hidden_states = self.transformer(
            embeddings, attention_mask, cache=cache, use_cache=use_cache,
            gradient_checkpointing=self.gradient_checkpointing
        )

        logits = self.to_logits(transformer_output)
        if return_loss is False:
            outputs = (logits, present_cache)
            if return_hidden_states:
                outputs += (hidden_states,)
            return outputs

        logits = rearrange(logits, 'b n c -> b c n')
        l_text_logits = logits[
            :, :self.vocab_size, :self.l_text_seq_length if use_image else self.l_text_seq_length-1
        ].contiguous().float()
        labels = [l_text[:, 1:]]
        if use_image:
            labels.append(image_input_ids)
            a, b = self.l_text_seq_length, self.l_text_seq_length + self.image_seq_length - 1
            image_logits = logits[:, self.vocab_size:, a:b].contiguous().float()
        if use_r_text:
            r_text_logits = logits[:, :self.vocab_size, -self.r_text_seq_length:-1].contiguous().float()
            labels.append(r_text)
        labels = torch.cat(labels, dim=1).contiguous().long()

        loss, loss_weights, loss_values = 0, 0, {}
        loss_l_text = F.cross_entropy(
            l_text_logits,
            labels[:, :self.l_text_seq_length]
        )
        loss_values['l_text_loss'] = loss_l_text.data.detach().float()
        if lt_loss_weight:
            loss += loss_l_text*lt_loss_weight
            loss_weights += lt_loss_weight
        if use_image:
            loss_img = F.cross_entropy(
                image_logits,
                labels[:, self.l_text_seq_length:self.l_text_seq_length + self.image_seq_length - 1]
            )
            loss_values['image_loss'] = loss_img.data.detach().float()
            if img_loss_weight:
                loss += loss_img*img_loss_weight
                loss_weights += img_loss_weight
        if use_r_text:
            loss_r_text = F.cross_entropy(
                r_text_logits,
                labels[:, -(self.r_text_seq_length-1):],
                ignore_index=0,
            )
            loss_values['r_text_loss'] = loss_r_text.data.detach().float()
            if rt_loss_weight:
                loss += loss_r_text * rt_loss_weight
                loss_weights += rt_loss_weight

        loss = loss / loss_weights
        outputs = (loss, loss_values)
        if return_hidden_states:
            outputs += (hidden_states,)
        return outputs

    def to(self, device, *args, **kwargs):
        self.device = device
        return super().to(device, *args, **kwargs)