import torch
import torch.nn as nn
from transformers import PaliGemmaForConditionalGeneration, PaliGemmaProcessor

class PaliGemmaBackbone(nn.Module):
    def __init__(self, config: BackboneConfig):
        super().__init__()
        self.config = config
        
        # Load Processor and Model
        self.processor = PaliGemmaProcessor.from_pretrained(config.model_id)
        self.model = PaliGemmaForConditionalGeneration.from_pretrained(
            config.model_id, 
            device_map="auto",
            torch_dtype=torch.bfloat16
        )

        # Apply Freezing Strategy (C1/C2 Logic)
        if config.freeze_backbone:
            for param in self.model.parameters():
                param.requires_grad = False
        
    def forward(self, images, text_instructions):
        """
        Input: 
            images: [Batch, 3, 224, 224]
            text_instructions: List[str]
        Output:
            features: [Batch, 196, 1024] (The Q-layer tokens)
            attention_map: [Batch, 14, 14] (The Masking source)
        """
        # 1. Preprocess inputs
        inputs = self.processor(
            text=text_instructions, 
            images=images, 
            return_tensors="pt", 
            padding=True
        ).to(self.model.device)

        # 2. Forward Pass with internal tapping
        outputs = self.model(
            **inputs,
            output_hidden_states=True,
            output_attentions=True
        )

        # 3. Extract Features (Q-Layer)
        # Hidden states: Tuple of layers. We pick the config's target layer.
        # Shape: [Batch, Seq_Len, Hidden_Dim]
        all_hidden_states = outputs.vision_tower_output.hidden_states
        features = all_hidden_states[self.config.target_layer]

        # 4. Extract Attention Map (For C3/C4/C5 masking)
        # We look at the vision tower attentions
        all_attentions = outputs.vision_tower_output.attentions
        layer_attn = all_attentions[self.config.target_layer] # [Batch, Heads, 196, 196]
        
        # Aggregate across heads (C3: Mean, C4: Selected Head)
        # For now, let's take the mean as a default starting point
        attention_map = layer_attn.mean(dim=1) # [Batch, 196, 196]
        
        # Focus on the 'Global' relationship (how patches look at each other)
        # We sum across the rows to see which patches are most 'attended to'
        spatial_attention = attention_map.sum(dim=1) # [Batch, 196]
        spatial_attention = spatial_attention.view(-1, 14, 14) # Reshape to grid

        return features, spatial_attention


    3. How to use it in your Training LoopNow your main training script stays very clean. You just swap the config to run different experiments.Python# Setup for C4 (LA-ReconVLA)
config = BackboneConfig(target_layer=8, freeze_backbone=True)
backbone = PaliGemmaBackbone(config)

# During the Loop
features, attn_grid = backbone(images=batch_images, text_instructions=batch_tasks)

# The 'features' now go to your Action Head
# The 'attn_grid' now goes to your Top-K Masking function
4. Why this works for your ExperimentsLayer Selection: By changing config.target_layer, you can "tap" into the Vision Tower at any depth (early, middle, or late) to see where the localization is strongest.Frozen Control (C1): If freeze_backbone is True, only your custom Action Head (the robot's muscles) will learn, while the "Eye" (PaliGemma) stays fixed.Bfloat16 Optimization: I added torch_dtype=torch.bfloat16. This is critical for the Colab T4 as it cuts the VRAM usage of the 3B model in half without losing much accuracy.5. Summary of the "Tapped" Outputsfeatures: These are the high-level semantic tokens. They contain the "what" and the "where."spatial_attention: This is your Attention-Guided Mask. If the model is looking at the bowl, the pixels corresponding to the bowl in this $14 \times 14$ grid will have the highest values.







In your architecture, the PaliGemma backbone acts as the Encoder (it understands the scene), and your MAE Decoder acts as the Artist (it rebuilds the scene).

1. The Math & Architecture
As per your Appendix B requirement, we are building a:

4-layer Transformer Decoder.

256 Hidden Dimension.

8 Attention Heads.

The goal is to map the feature space (what the model thinks is there) back to the pixel space (the image).

2. Defining the MAEDecoder Class
You can define this in models/mae.py. Notice that it takes the features from your Backbone and attempts to predict the original pixel patches.

Python
import torch
import torch.nn as nn

class MAEDecoder(nn.Module):
    def __init__(self, embed_dim=1024, decoder_dim=256):
        super().__init__()
        # 1. Project backbone features down to decoder dimension
        self.proj = nn.Linear(embed_dim, decoder_dim)
        
        # 2. The 4-layer Transformer Decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=decoder_dim, nhead=8, batch_first=True
        )
        self.transformer = nn.TransformerDecoder(decoder_layer, num_layers=4)
        
        # 3. Predict the patches (16x16 pixels * 3 color channels = 768)
        self.patch_head = nn.Linear(decoder_dim, 16 * 16 * 3)

    def forward(self, features, mask_indices):
        # features: [Batch, 196, 1024]
        # 1. Shrink features to decoder dim
        x = self.proj(features)
        
        # 2. Only pass the MASKED features to the decoder 
        # (The decoder doesn't need to see what's already visible!)
        masked_x = x[:, mask_indices, :]
        
        # 3. Decode
        reconstructed_patches = self.transformer(masked_x, x)
        
        # 4. Return the patch predictions
        return self.patch_head(reconstructed_patches)
3. How the "Loss" connects
This is the most critical part of your experiment. In your forward() pass of LAReconVLA, you calculate two losses:

Python
# C1-C5 Training Logic
pred_action, recon_loss = model(batch)

# Action Loss: How well did the robot move?
action_loss = F.mse_loss(pred_action, batch['action'])

# Reconstruction Loss: How well did the MAE rebuild the 'Bowl'?
# We only compare the decoder's output to the ACTUAL pixels that were masked
recon_loss = F.mse_loss(reconstructed_patches, original_patches[mask_indices])

# Combine
total_loss = action_loss + (0.5 * recon_loss) # 0.5 is your Lambda
4. Why this Decoder is "Small"
You might notice the decoder_dim is only 256.

Your backbone (PaliGemma-2) is massive (3B parameters).

Your MAE Decoder is tiny (~1-2M parameters).

The Benefit: Because the decoder is small, it acts as a bottleneck. It cannot memorize the images; it is forced to learn the "concepts" of the scene to reconstruct them. If you made the decoder too big, it would just memorize the pixel values and your model wouldn't actually "learn" the spatial reasoning you want to test in H3.

5. Implementation Checklist for Part 2
Patchification: You must ensure that your original_patches are reshaped into (Batch, 196, 768) to match the MAE input.

Stop-Gradient: When you send features to the MAEDecoder, use features.detach() if you only want the MAE to learn, OR keep the gradient flowing if you want the reconstruction task to "sharpen" the backbone's vision. For your experiment, I recommend features.detach() for C3 and C4 to isolate the backbone's performance from the MAE's performance.