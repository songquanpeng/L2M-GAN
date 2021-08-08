import torch
import os
import copy
from utils.image import save_image
from data.loader import get_eval_loader
from tqdm import tqdm
from utils.file import make_path


@torch.no_grad()
def translate_using_latent(nets, args, x_src, y_trg_list, z_trg_list, filename):
    x_concat = [x_src]
    for y_trg in y_trg_list:
        for z_trg in z_trg_list:
            s_trg = nets.mapping_network(z_trg, y_trg)
            x_fake = nets.generator(x_src, s_trg)
            x_concat += [x_fake]
    x_concat = torch.cat(x_concat, dim=0)
    save_image(x_concat, x_src.size()[0], filename)


@torch.no_grad()
def translate_using_label(nets, args, sample_src, y_trg_list, filename):
    x_concat = [sample_src.x]
    masks = nets.fan.get_heatmap(sample_src.x) if args.w_hpf > 0 else None
    for y_trg in y_trg_list:
        s = nets.style_encoder(sample_src.x, sample_src.y)
        s_tilde = nets.style_transformer(s, y_trg)
        x_fake = nets.generator(sample_src.x, s_tilde, masks=masks)
        x_concat += [x_fake]
    x_concat = torch.cat(x_concat, dim=0)
    save_image(x_concat, sample_src.x.size()[0], filename)


@torch.no_grad()
def generate_samples(nets, args, path):
    args = copy.deepcopy(args)
    args.batch_size = args.eval_batch_size
    for src_idx, src_domain in enumerate(args.domains):
        loader = get_eval_loader(path=os.path.join(args.test_path, src_domain), **args)
        N = args.eval_batch_size
        target_domains = [domain for domain in args.domains]
        src_class_label = torch.tensor([src_idx] * N).to(args.device)
        for trg_idx, trg_domain in enumerate(target_domains):
            if trg_domain == src_domain:
                continue
            trg_class_label = torch.tensor([trg_idx] * N).to(args.device)
            save_path = os.path.join(path, f"{src_domain}2{trg_domain}")
            make_path(save_path)
            for i, query_image in enumerate(tqdm(loader, total=len(loader))):
                query_image = query_image.to(args.device)
                masks = nets.fan.get_heatmap(query_image) if args.w_hpf > 0 else None
                images = []
                for j in range(args.eval_repeat_num):
                    s = nets.style_encoder(query_image, src_class_label)
                    s_tilde = nets.style_transformer(s, trg_class_label)
                    generated_image = nets.generator(query_image, s_tilde, masks=masks)
                    images.append(generated_image)
                    for k in range(N):
                        filename = os.path.join(save_path, f"{i * args.eval_batch_size + k}_{j}.png")
                        save_image(generated_image[k], col_num=1, filename=filename)
