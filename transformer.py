class LinearEmbedding(nn.Sequential):

    # input으로 
    def __init__(self, input_channels, output_channels) -> None:
        super().__init__(*[
            nn.Linear(input_channels, output_channels),
            nn.LayerNorm(output_channels),
            nn.GELU()
        ])
        
        # cls_token이름의 learnable paramter생성
        self.cls_token = nn.Parameter(torch.randn(1, output_channels))

    def forward(self, x):
        # 여기서 input x가 nn.Linear, nn.LayerNorm, nn.GELU를 거쳐져서 embedding됨.
        embedded = super().forward(x)
        # cls_token을 batch크기에 맞게 복제하고, (토큰 n, 임베딩 차원 e)를 (batch , n , e)형태로 변경, 
        # 이후 cls_token 과 embedding된 embedded를 연결함. 그러면 cls_token이 임베딩의 맨앞에 위치하게됨.
        return torch.cat([einops.repeat(self.cls_token, "n e -> b n e", b=x.shape[0]), embedded], dim=1)


class MLP(nn.Sequential):
    def __init__(self, input_channels, expansion=4):
        super().__init__(*[
            nn.Linear(input_channels, input_channels * expansion),
            nn.GELU(),
            nn.Linear(input_channels * expansion, input_channels)
        ])


class ResidualAdd(torch.nn.Module):
    def __init__(self, block):
        super().__init__()
        self.block = block

    def forward(self, x):
        return x + self.block(x)


# attention 계산.
class MultiHeadAttention(torch.nn.Module):
    def __init__(self, embed_size, num_heads, attention_store=None):
        super().__init__()
        self.queries_projection = nn.Linear(embed_size, embed_size)
        self.values_projection = nn.Linear(embed_size, embed_size)
        self.keys_projection = nn.Linear(embed_size, embed_size)
        self.final_projection = nn.Linear(embed_size, embed_size)
        self.embed_size = embed_size
        self.num_heads = num_heads

    def forward(self, x):
        assert len(x.shape) == 3
        keys = self.keys_projection(x)
        values = self.values_projection(x)
        queries = self.queries_projection(x)
        keys = einops.rearrange(keys, "b n (h e) -> b n h e", h=self.num_heads)
        queries = einops.rearrange(queries, "b n (h e) -> b n h e", h=self.num_heads)
        values = einops.rearrange(values, "b n (h e) -> b n h e", h=self.num_heads)
        energy_term = torch.einsum("bqhe, bkhe -> bqhk", queries, keys)
        divider = sqrt(self.embed_size)
        mh_out = torch.softmax(energy_term, -1)
        out = torch.einsum('bihv, bvhd -> bihd ', mh_out / divider, values)
        out = einops.rearrange(out, "b n h e -> b n (h e)")
        return self.final_projection(out)


# transforemr encoder 정의
# 위의 ResidualAdd를 통해 2개의 residual이 있고 순차적으로 적용하여 encoderlayer 정의
class TransformerEncoderLayer(torch.nn.Sequential):
    def __init__(self, embed_size=768, expansion=4, num_heads=8, dropout=0.1):
        super(TransformerEncoderLayer, self).__init__(
            *[
                
                ResidualAdd(nn.Sequential(*[
                    nn.LayerNorm(embed_size),
                    MultiHeadAttention(embed_size, num_heads),
                    nn.Dropout(dropout)
                ])),
                ResidualAdd(nn.Sequential(*[
                    nn.LayerNorm(embed_size),
                    MLP(embed_size, expansion),
                    nn.Dropout(dropout)
                ]))
            ]
        )


class Classifier(nn.Sequential):
    def __init__(self, embed_size, num_classes):
        super().__init__(*[
            Reduce("b n e -> b e", reduction="mean"),
            nn.Linear(embed_size, embed_size),
            nn.LayerNorm(embed_size),
            nn.Linear(embed_size, num_classes)
        ])


 
# 이게 본체임 일단.
# input으로는 (batch, element, channel순으로 들어옴.) // (64, 187, 1)
# train.py에서 모델 입력으로 einops.rearrange(signal, "b c e -> b e c") 형식으로 들어왔기 때문에.
class ECGformer(nn.Module):

    def __init__(self, num_layers, signal_length, num_classes, input_channels, embed_size, num_heads, expansion) -> None:
        super().__init__()
         
        # num_layers에 대해 TransformerEncoderLayer instance들의 list를 생성. 이것은 nn.ModuleList에 담김.
        self.encoder = nn.ModuleList([TransformerEncoderLayer(
            embed_size=embed_size, num_heads=num_heads, expansion=expansion) for _ in range(num_layers)])
        self.classifier = Classifier(embed_size, num_classes)
        self.positional_encoding = nn.Parameter(torch.randn(signal_length + 1, embed_size))
        self.embedding = LinearEmbedding(input_channels, embed_size)

    def forward(self, x):
        # embedding을 진행하고, cls_token을 추가함.
        embedded = self.embedding(x)

        # encoder에 저장된 list들에 positional_encoding을 더함.
        for layer in self.encoder:
            embedded = layer(embedded + self.positional_encoding)

        # classification
        return self.classifier(embedded)