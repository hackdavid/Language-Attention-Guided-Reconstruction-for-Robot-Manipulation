i have this model layers 
PaliGemmaForConditionalGeneration(
  (model): PaliGemmaModel(
    (vision_tower): SiglipVisionModel(
      (vision_model): SiglipVisionTransformer(
        (embeddings): SiglipVisionEmbeddings(
          (patch_embedding): Conv2d(3, 1152, kernel_size=(14, 14), stride=(14, 14), padding=valid)
          (position_embedding): Embedding(256, 1152)
        )
        (encoder): SiglipEncoder(
          (layers): ModuleList(
            (0-26): 27 x SiglipEncoderLayer(
              (layer_norm1): LayerNorm((1152,), eps=1e-06, elementwise_affine=True)
              (self_attn): SiglipAttention(
                (k_proj): Linear(in_features=1152, out_features=1152, bias=True)
                (v_proj): Linear(in_features=1152, out_features=1152, bias=True)
                (q_proj): Linear(in_features=1152, out_features=1152, bias=True)
                (out_proj): Linear(in_features=1152, out_features=1152, bias=True)
              )
              (layer_norm2): LayerNorm((1152,), eps=1e-06, elementwise_affine=True)
              (mlp): SiglipMLP(
                (activation_fn): GELUTanh()
                (fc1): Linear(in_features=1152, out_features=4304, bias=True)
                (fc2): Linear(in_features=4304, out_features=1152, bias=True)
              )
            )
          )
        )
        (post_layernorm): LayerNorm((1152,), eps=1e-06, elementwise_affine=True)
      )
    )
    (multi_modal_projector): PaliGemmaMultiModalProjector(
      (linear): Linear(in_features=1152, out_features=2304, bias=True)
    )
    (language_model): Gemma2Model(
      (embed_tokens): Embedding(257216, 2304, padding_idx=0)
      (layers): ModuleList(
        (0-25): 26 x Gemma2DecoderLayer(
          (self_attn): Gemma2Attention(
            (q_proj): Linear(in_features=2304, out_features=2048, bias=False)
            (k_proj): Linear(in_features=2304, out_features=1024, bias=False)
            (v_proj): Linear(in_features=2304, out_features=1024, bias=False)
            (o_proj): Linear(in_features=2048, out_features=2304, bias=False)
          )
          (mlp): Gemma2MLP(
            (gate_proj): Linear(in_features=2304, out_features=9216, bias=False)
            (up_proj): Linear(in_features=2304, out_features=9216, bias=False)
            (down_proj): Linear(in_features=9216, out_features=2304, bias=False)
            (act_fn): GELUTanh()
          )
          (input_layernorm): Gemma2RMSNorm((2304,), eps=1e-06)
          (post_attention_layernorm): Gemma2RMSNorm((2304,), eps=1e-06)
          (pre_feedforward_layernorm): Gemma2RMSNorm((2304,), eps=1e-06)
          (post_feedforward_layernorm): Gemma2RMSNorm((2304,), eps=1e-06)
        )
      )
      (norm): Gemma2RMSNorm((2304,), eps=1e-06)
      (rotary_emb): Gemma2RotaryEmbedding()
    )
  )
  (lm_head): Linear(in_features=2304, out_features=257216, bias=False)
)

so in our c1 - we have to fine-tune and add a layer that will give us action-head and only update those wegit so it align with it or not current when i pass c1 yaml