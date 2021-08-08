import copy
from munch import Munch
from models.generator import Generator
from models.discriminator import Discriminator
from models.mapping_network import MappingNetwork
from models.style_encoder import StyleEncoder
from models.wing import FAN


def build_model(args):
    generator = Generator(args)
    mapping_network = MappingNetwork(args)
    style_encoder = StyleEncoder(args)
    discriminator = Discriminator(args)
    generator_ema = copy.deepcopy(generator)
    mapping_network_ema = copy.deepcopy(mapping_network)
    style_encoder_ema = copy.deepcopy(style_encoder)

    nets = Munch(generator=generator,
                 mapping_network=mapping_network,
                 style_encoder=style_encoder,
                 discriminator=discriminator)
    nets_ema = Munch(generator=generator_ema,
                     mapping_network=mapping_network_ema,
                     style_encoder=style_encoder_ema)

    if args.w_hpf > 0:
        fan = FAN(fname_pretrained=args.wing_path).eval()
        nets.fan = fan
        nets_ema.fan = fan

    return nets, nets_ema
