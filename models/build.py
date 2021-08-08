import copy
from munch import Munch
from models.generator import Generator
from models.discriminator import Discriminator
from models.style_transformer import StyleTransformer
from models.style_encoder import StyleEncoder
from models.wing import FAN


def build_model(args):
    generator = Generator(args)
    style_transformer = StyleTransformer(args)
    style_encoder = StyleEncoder(args)
    discriminator = Discriminator(args)
    generator_ema = copy.deepcopy(generator)
    style_transformer_ema = copy.deepcopy(style_transformer)
    style_encoder_ema = copy.deepcopy(style_encoder)

    nets = Munch(generator=generator,
                 style_transformer=style_transformer,
                 style_encoder=style_encoder,
                 discriminator=discriminator)
    nets_ema = Munch(generator=generator_ema,
                     style_transformer=style_transformer_ema,
                     style_encoder=style_encoder_ema)

    if args.w_hpf > 0:
        fan = FAN(fname_pretrained=args.wing_path).eval()
        nets.fan = fan
        nets_ema.fan = fan

    return nets, nets_ema
