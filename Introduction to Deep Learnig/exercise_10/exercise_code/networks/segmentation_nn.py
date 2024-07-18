"""SegmentationNN"""
import torch
import torch.nn as nn

class ConvLayer(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(ConvLayer, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.activation = nn.ReLU()
        self.norm = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.activation(x)
        return x



class SegmentationNN(nn.Module):

    def __init__(self, num_classes=23, hp=None):
        super().__init__()
        self.hp = hp
        #######################################################################
        #                             YOUR CODE                               #
        #######################################################################

        from torchvision.models import alexnet, mobilenet_v3_large, densenet201



        self.encoder = alexnet(pretrained = True).features
        self.encoder.requires_grad_(False)
        self.num_classes = num_classes
        self.hparams = hp
        self.batch_size = hp['batch_size']


        # input size: 240 x 240
        # output size: 240 x 240
        self.upsample = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bicubic'), # 12 x 12 
            ConvLayer(256, 128, 3, 1, 1), # 12 x 12
            nn.Upsample(scale_factor=2, mode='bicubic'), # 24 x 24
            ConvLayer(128, 64, 5, 1, 0), # 20 x 20 
            nn.Upsample(scale_factor=2, mode='bicubic'), # 40 x 40
            ConvLayer(64, 32, 3, 1, 1), # 40 x 40 
            nn.Upsample(scale_factor=2, mode='bicubic'), # 64 x 64
            ConvLayer(32, 16, 3, 1, 1), # 60 x 60
            nn.Upsample(scale_factor=3, mode='bicubic'), # 120 x 120
            nn.Conv2d(16, self.num_classes, 1, 1, 0) # 240 x 240
        )

        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################

    def forward(self, x):
        """
        Forward pass of the convolutional neural network. Should not be called
        manually but by calling a model instance directly.

        Inputs:
        - x: PyTorch input Variable
        """
        #######################################################################
        #                             YOUR CODE                               #
        #######################################################################
        
        x = self.encoder(x)
        x = self.upsample(x)
    
        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################

        return x

    # @property
    # def is_cuda(self):
    #     """
    #     Check if model parameters are allocated on the GPU.
    #     """
    #     return next(self.parameters()).is_cuda

    def save(self, path):
        """
        Save model with its parameters to the given path. Conventionally the
        path should end with "*.model".

        Inputs:
        - path: path string
        """
        print('Saving model... %s' % path)
        torch.save(self, path)

        
class DummySegmentationModel(nn.Module):

    def __init__(self, target_image):
        super().__init__()
        def _to_one_hot(y, num_classes):
            scatter_dim = len(y.size())
            y_tensor = y.view(*y.size(), -1)
            zeros = torch.zeros(*y.size(), num_classes, dtype=y.dtype)

            return zeros.scatter(scatter_dim, y_tensor, 1)

        target_image[target_image == -1] = 1

        self.prediction = _to_one_hot(target_image, 23).permute(2, 0, 1).unsqueeze(0)

    def forward(self, x):
        return self.prediction.float()

if __name__ == "__main__":
    from torchinfo import summary
    summary(SegmentationNN(), (1, 3, 240, 240), device="cpu")