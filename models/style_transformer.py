import torch
import torch.nn as nn


class StyleTransformer(nn.Module):
    def __init__(self, args):
        super().__init__()
        style_dim, num_domains = args.style_dim, args.num_domains
        layers = []
        layers += [nn.Linear(style_dim, style_dim)]
        layers += [nn.ReLU()]
        for _ in range(3):
            layers += [nn.Linear(style_dim, style_dim)]
            layers += [nn.ReLU()]
        self.decomposer = nn.Sequential(*layers)

        self.domain_transformer = nn.ModuleList()
        for _ in range(num_domains):
            self.domain_transformer += [nn.Sequential(nn.Linear(style_dim, style_dim), nn.ReLU(),
                                                      nn.Linear(style_dim, style_dim), nn.ReLU(),
                                                      nn.Linear(style_dim, style_dim), nn.ReLU(),
                                                      nn.Linear(style_dim, style_dim))]

    def forward(self, s, y, return_all=False):
        s_unrelated = self.decomposer(s)
        s_related = s - s_unrelated
        out = []
        for layer in self.domain_transformer:
            out += [layer(s_related)]
        out = torch.stack(out, dim=1)  # (batch, num_domains, style_dim)
        idx = torch.LongTensor(range(y.size(0))).to(y.device)
        s_related_tilde = out[idx, y]  # (batch, style_dim)
        s_tilde = s_related_tilde + s_unrelated
        if return_all:
            return s_tilde, s_unrelated, s_related, s_related_tilde
        return s_tilde
