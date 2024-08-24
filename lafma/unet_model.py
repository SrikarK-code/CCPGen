class CustomUNet1D(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        model_channels,
        num_res_blocks,
        attention_resolutions,
        dropout=0,
        channel_mult=(1, 2, 4, 8),
        conv_resample=True,
        use_checkpoint=False,
        num_heads=8,
        use_scale_shift_norm=False,
        use_spatial_transformer=False,
        transformer_depth=1,
        context_dim=1,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.use_checkpoint = use_checkpoint
        self.num_heads = num_heads
        self.use_spatial_transformer = use_spatial_transformer
        self.context_dim = context_dim

        self.context_proj = nn.Linear(context_dim, model_channels)
        self.final_proj = nn.Linear(model_channels * 2, out_channels)  # *2 to account for concatenated context

        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            nn.Linear(model_channels, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )

        self.input_blocks = nn.ModuleList([
            nn.Conv1d(in_channels, model_channels, 3, padding=1)
        ])

        ch = model_channels
        input_block_chans = [model_channels]
        ds = 1
        for level, mult in enumerate(channel_mult):
            for _ in range(num_res_blocks):
                layers = [
                    ResBlock(
                        ch,
                        time_embed_dim,
                        dropout,
                        out_channels=mult * model_channels,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = mult * model_channels
                if ds in attention_resolutions:
                    if use_spatial_transformer:
                        layers.append(
                            SpatialTransformer(
                                ch, num_heads, context_dim, depth=transformer_depth
                            )
                        )
                    else:
                        layers.append(AttentionBlock(ch, num_heads=num_heads))
                self.input_blocks.append(nn.Sequential(*layers))
                input_block_chans.append(ch)
            if level != len(channel_mult) - 1:
                self.input_blocks.append(
                    nn.Conv1d(ch, ch, 3, stride=2, padding=1)
                )
                input_block_chans.append(ch)
                ds *= 2

        self.middle_block = nn.Sequential(
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
            AttentionBlock(ch, num_heads=num_heads) if not use_spatial_transformer else
            SpatialTransformer(ch, num_heads, context_dim, depth=transformer_depth),
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
        )

        self.output_blocks = nn.ModuleList([])
        for level, mult in list(enumerate(channel_mult))[::-1]:
            for i in range(num_res_blocks + 1):
                layers = [
                    ResBlock(
                        ch + input_block_chans.pop(),
                        time_embed_dim,
                        dropout,
                        out_channels=model_channels * mult,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = model_channels * mult
                if ds in attention_resolutions:
                    if use_spatial_transformer:
                        layers.append(
                            SpatialTransformer(
                                ch, num_heads, context_dim, depth=transformer_depth
                            )
                        )
                    else:
                        layers.append(AttentionBlock(ch, num_heads=num_heads))
                if level and i == num_res_blocks:
                    layers.append(nn.ConvTranspose1d(ch, ch, 4, stride=2, padding=1))
                    ds //= 2
                self.output_blocks.append(nn.Sequential(*layers))

        self.out = nn.Sequential(
            nn.GroupNorm(32, ch),
            nn.SiLU(),
            nn.Conv1d(ch, out_channels, 3, padding=1),
        )

    def forward(self, x, timesteps, context=None):
        x = x.transpose(1, 2)  # transpose shape: [batch_size, emb_dim, seqlen]
        t_emb = self.time_embed(timestep_embedding(timesteps, self.model_channels))

        h = x
        hs = []
        for module in self.input_blocks:
            if isinstance(module, nn.Conv1d):
                h = module(h)
            elif isinstance(module, SpatialTransformer):
                h = module(h, context)
            elif isinstance(module, ResBlock):
                h = module(h, t_emb)
            elif isinstance(module, nn.Sequential):
                for submodule in module:
                    if isinstance(submodule, ResBlock):
                        h = submodule(h, t_emb)
                    elif isinstance(submodule, SpatialTransformer):
                        h = submodule(h, context)
                    else:
                        h = submodule(h)
            else:
                h = module(h)
            hs.append(h)

        if isinstance(self.middle_block, nn.Sequential):
            for submodule in self.middle_block:
                if isinstance(submodule, ResBlock):
                    h = submodule(h, t_emb)
                elif isinstance(submodule, SpatialTransformer):
                    h = submodule(h, context)
                else:
                    h = submodule(h)
        else:
            h = self.middle_block(h)

        for module in self.output_blocks:
            h = torch.cat([h, hs.pop()], dim=1)
            if isinstance(module, nn.Sequential):
                for submodule in module:
                    if isinstance(submodule, ResBlock):
                        h = submodule(h, t_emb)
                    elif isinstance(submodule, SpatialTransformer):
                        h = submodule(h, context)
                    else:
                        h = submodule(h)
            elif isinstance(module, SpatialTransformer):
                h = module(h, context)
            elif isinstance(module, ResBlock):
                h = module(h, t_emb)
            else:
                h = module(h)

        output = self.out(h) # [batch_size, model_channels, seq_len]
        output = output.transpose(1, 2)  # shape: [batch_size, seqlen, model_channels]

        return output
