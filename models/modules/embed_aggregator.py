import jittor as jt
import jittor.nn as nn

class EmbedAggregator(nn.Module):
    """
    Aggregate feature maps of neighboring time interval.
    """
    def __init__(self, channels, kernel_size=3):
        super(EmbedAggregator, self).__init__()
        self.embed_convs = nn.Sequential(nn.Conv(channels, channels, kernel_size, padding=(kernel_size-1)//2),
                                         nn.relu(),
                                         nn.Conv(channels, channels, kernel_size, padding=(kernel_size-1)//2),
                                         nn.relu())

    def execute(self, curr_x, prev_x):
        """
        1.Compute the cos similarity between current feature map and previous feature map. e.g. Ft, Ft-1
        2.Use the normalized(softmax)cos similarity to weightedly hidden state
        Args:
            curr_x: [1, C, H, W]
            prev_x: [1, C, H, W]
        Returns:
            weights: [1, 1, H, W]
        """
        curr_embed = self.embed_convs(curr_x)
        prev_embed = self.embed_convs(prev_x)

        curr_embed = curr_embed / (curr_embed.norm(p=2, dim=1, keepdim=True) + 1e-6)  # L2
        prev_embed = prev_embed / (prev_embed.norm(p=2, dim=1, keepdim=True) + 1e-6)

        weights = jt.sum(curr_embed*prev_embed, dim=1, keepdim=True)
        return weights

# Example usage
if __name__ == "__main__":
    jt.flags.use_cuda = False
    agg = EmbedAggregator(channels=64)
    x1 = jt.randn([1,64,32,32])
    x2 = jt.randn([1,64,32,32])
    w = agg(x1, x2)
    print(w.shape)  # should be [1,1,32,32]
