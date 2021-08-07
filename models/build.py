import copy
from munch import Munch
from models.generator import Generator
from models.discriminator import Discriminator
from models.mapping_network import MappingNetwork


def build_model(args):
    generator = Generator(args)
    discriminator = Discriminator(args)
    mapping_network = MappingNetwork(args)

    generator_ema = copy.deepcopy(generator)
    mapping_network_ema = copy.deepcopy(mapping_network)

    nets = Munch(generator=generator, discriminator=discriminator, mapping_network=mapping_network)
    nets_ema = Munch(generator=generator_ema, mapping_network=mapping_network_ema)

    return nets, nets_ema
