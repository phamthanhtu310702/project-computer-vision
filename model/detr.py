import torch
from transformers import AutoProcessor, Swinv2Config, Swinv2Model
import torch
from torch import nn

CONFIG = Swinv2Config()
PRE_TRAINED_MODEL = "microsoft/swinv2-tiny-patch4-window8-256"

class SwinDetr(nn.Module):
    """
    Demo Swin DETR implementation.
    """
    def __init__(self, num_classes, hidden_dim=384, nheads=8,
                 num_encoder_layers=6, num_decoder_layers=6):
        super().__init__()

        self.backbone = Swinv2Model(CONFIG)
        self.merge_backbone = nn.Sequential()
        # x.append(i) for i in self.backbone
        # x = x[:-2]
        #  y = nn.Sequential(*x[:-2])
        # create conversion layer
        self.conv = nn.Conv2d(384, hidden_dim, 1)

        # create a default PyTorch transformer
        self.transformer = nn.Transformer(
            hidden_dim, nheads, num_encoder_layers, num_decoder_layers
            )

        # prediction heads, one extra class for predicting non-empty slots
        # note that in baseline DETR linear_bbox layer is 3-layer MLP
        self.linear_class = nn.Linear(hidden_dim, num_classes + 1)
        self.linear_bbox = nn.Linear(hidden_dim, 4)

        # output positional encodings (object queries)
        self.query_pos = nn.Parameter(torch.rand(100, hidden_dim))

        # spatial positional encodings
        # note that in baseline DETR we use sine positional encodings
        self.row_embed = nn.Parameter(torch.rand(50, hidden_dim // 2))
        self.col_embed = nn.Parameter(torch.rand(50, hidden_dim // 2))

    def forward(self, inputs):
        # propagate inputs through Swin-v2 up to avg-pool layer
        module=[]
        for i in self.backbone.children():
             module.append(i)
        x = nn.Sequential(*module[:-2]) # 1 or 2 ??? read the paper(swin and swinv2) again
        x = x(**inputs)
        x = x.last_hidden_state
       # temp = f" Shapes \n Output of Backbone : {x.shape}"

        # convert from 2048 to 256 feature planes for the transformer
        h = self.conv(x)

        # temp += f" \n Output of Conv : {h.shape}"

        # construct positional encodings
        H, W = h.shape[-2:]

        # temp += f" \n H,W : {H, W}"
        # temp += f" \n Col Embed Alone : {self.col_embed[:W].shape}"
        # temp += f" \n Col Embed After : {self.col_embed[:W].unsqueeze(0).repeat(H, 1, 1).shape}"
        # temp += f" \n Row Embed Alone : {self.row_embed[:H].shape}"
        # temp += f" \n Row Embed After : {self.row_embed[:H].unsqueeze(1).repeat(1, W, 1).shape}"
        # temp += f" \n Cat Alone : {torch.cat([ self.col_embed[:W].unsqueeze(0).repeat(H, 1, 1),self.row_embed[:H].unsqueeze(1).repeat(1, W, 1),], dim=-1).shape}"

        pos = torch.cat([
            self.col_embed[:W].unsqueeze(0).repeat(H, 1, 1),
            self.row_embed[:H].unsqueeze(1).repeat(1, W, 1),
        ], dim=-1).flatten(0, 1).unsqueeze(1)

        # temp += f" \n Cat After : {pos.shape}"
        # temp += f" \n h.flatten(2).permute(2, 0, 1) : {h.flatten(2).permute(2, 0, 1).shape}"
        # temp += f" \n self.query_pos.unsqueeze(1) : {self.query_pos.unsqueeze(1).shape}"

        

        # propagate through the transformer
        h = self.transformer(pos + 0.1 * h.flatten(2).permute(2, 0, 1),
                             self.query_pos.unsqueeze(1)).transpose(0, 1)

        temp += f" \n h last : {h.shape}"

        print(temp)
        
        # finally project transformer outputs to class labels and bounding boxes
        return {'pred_logits': self.linear_class(h), 
                'pred_boxes': self.linear_bbox(h).sigmoid()}