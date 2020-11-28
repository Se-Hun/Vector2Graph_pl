import torch.nn as nn
import torch.nn.functional as F


class ImageEncoder(nn.Module):
    def __init__(self, num_labels, image_chanel_size=3):
        super(ImageEncoder, self).__init__()
        self.layer1 = nn.Conv2d(image_chanel_size, 64, kernel_size=(3, 3),
                                padding=(1, 1), stride=(1, 1))
        self.layer2 = nn.Conv2d(64, 128, kernel_size=(3, 3),
                                padding=(1, 1), stride=(1, 1))
        self.layer3 = nn.Conv2d(128, 256, kernel_size=(3, 3),
                                padding=(1, 1), stride=(1, 1))
        self.layer4 = nn.Conv2d(256, 256, kernel_size=(3, 3),
                                padding=(1, 1), stride=(1, 1))
        self.layer5 = nn.Conv2d(256, 512, kernel_size=(3, 3),
                                padding=(1, 1), stride=(1, 1))
        self.layer6 = nn.Conv2d(512, 512, kernel_size=(3, 3),
                                padding=(1, 1), stride=(1, 1))
        self.layer7 = nn.Sequential(
            nn.Linear(512 * 8 * 8, 1000),
            nn.Linear(1000, 100)
        )
        self.batch_norm1 = nn.BatchNorm2d(256)
        self.batch_norm2 = nn.BatchNorm2d(512)
        self.batch_norm3 = nn.BatchNorm2d(512)
        self.to_class = nn.Linear(100, num_labels)

    @classmethod
    def from_opt(cls, opt, embeddings=None):
        """Alternate constructor."""
        if embeddings is not None:
            raise ValueError("Cannot use embeddings with ImageEncoder.")
        # why is the model_opt.__dict__ check necessary?
        if "image_channel_size" not in opt.__dict__:
            image_channel_size = 3
        else:
            image_channel_size = opt.image_channel_size
        return cls(
            opt.enc_layers,
            opt.brnn,
            opt.enc_rnn_size,
            opt.dropout[0] if type(opt.dropout) is list else opt.dropout,
            image_channel_size
        )

    def load_pretrained_vectors(self, opt):
        """Pass in needed options only when modify function definition."""
        pass

    def forward(self, b_img_tensors, b_class_intent_id=None):
        """See :func:`onmt.encoders.encoder.EncoderBase.forward()`"""
        batch_size = b_img_tensors.size(0)
        # (batch_size, 64, imgH, imgW)
        # layer 1
        h = F.relu(self.layer1(b_img_tensors[:, :, :, :] - 0.5), True)
        # (batch_size, 64, imgH/2, imgW/2)
        h = F.max_pool2d(h, kernel_size=(2, 2), stride=(2, 2))
        # (batch_size, 128, imgH/2, imgW/2)
        # layer 2
        h = F.relu(self.layer2(h), True)
        # (batch_size, 128, imgH/2/2, imgW/2/2)
        h = F.max_pool2d(h, kernel_size=(2, 2), stride=(2, 2))
        #  (batch_size, 256, imgH/2/2, imgW/2/2)
        # layer 3
        # batch norm 1
        h = F.relu(self.batch_norm1(self.layer3(h)), True)
        # (batch_size, 256, imgH/2/2, imgW/2/2)
        # layer4
        h = F.relu(self.layer4(h), True)
        # (batch_size, 256, imgH/2/2/2, imgW/2/2)
        h = F.max_pool2d(h, kernel_size=(1, 2), stride=(1, 2))
        # (batch_size, 512, imgH/2/2/2, imgW/2/2)
        # layer 5
        # batch norm 2
        h = F.relu(self.batch_norm2(self.layer5(h)), True)
        # (batch_size, 512, imgH/2/2/2, imgW/2/2/2)
        h = F.max_pool2d(h, kernel_size=(2, 1), stride=(2, 1))
        # (batch_size, 512, imgH/2/2/2, imgW/2/2/2)
        h = F.relu(self.batch_norm3(self.layer6(h)), True)
        h = h.reshape(h.size(0), -1)
        h = self.layer7(h)  # --> [100, 100]
        last_hidden_state = h
        logit = self.to_class(h)

        return logit, last_hidden_state