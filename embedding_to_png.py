import modules.scripts as scripts
import gradio as gr
import os
import torch
import io
from modules import devices, sd_models, sd_hijack
from PIL import PngImagePlugin
from modules.processing import Processed, process_images
from modules.shared import opts, cmd_opts, state
from modules.textual_inversion.image_embedding import (
    caption_image_overlay,
    insert_image_data_embed,
    extract_image_data_embed,
    embedding_to_b64,
    embedding_from_b64,
)


class Script(scripts.Script):

    def title(self):
        return "Embedding to Shareable PNG"

    def ui(self, is_img2img):
        embedding = gr.File(label="Source embedding to convert")
        embedding_token = gr.Textbox(label="Embedding token")
        destination_folder = gr.Textbox(label="Output directory", value="outputs")
        return [embedding, embedding_token, destination_folder]

    def run(self, p, embedding, embedding_token, destination_folder):
        print(embedding, embedding_token, destination_folder)
        assert os.path.exists(destination_folder)
        sd_hijack.model_hijack.embedding_db.load_textual_inversion_embeddings()

        p.width = 512
        p.height = 512

        embedding_token = embedding_token.replace('<', '').replace('>', '').strip()

        try:
            data = torch.load(embedding.name)
        except Exception as e:
            data = extract_image_data_embed(Image.open(embedding.name))

        assert data is not None

        original_name = os.path.splitext(os.path.basename(os.path.split(embedding.orig_name)[-1]))[0]

        if p.prompt == '':
            p.prompt = original_name

        if embedding_token == '':
            embedding_token = original_name

        # textual inversion embeddings
        if 'string_to_param' in data:
            param_dict = data['string_to_param']
            if hasattr(param_dict, '_parameters'):
                param_dict = getattr(param_dict, '_parameters')  # fix for torch 1.12.1 loading saved file from torch 1.11
            assert len(param_dict) == 1, 'embedding file has multiple terms in it'
            emb = next(iter(param_dict.items()))[1]
        # diffuser concepts
        elif type(data) == dict and type(next(iter(data.values()))) == torch.Tensor:
            assert len(data.keys()) == 1, 'embedding file has multiple terms in it'

            emb = next(iter(data.values()))
            if len(emb.shape) == 1:
                emb = emb.unsqueeze(0)
        else:
            raise Exception(f"Couldn't identify embedding as either textual inversion embedding nor diffuser concept.")

        checkpoint = sd_models.select_checkpoint()

        emb_data = {
            "string_to_token": {"*": 265},
            "string_to_param": {"*": emb.detach().to(devices.device, dtype=torch.float32)},
            "name": embedding_token,
            "step": data.get('step', 0),
            "sd_checkpoint": data.get('hash', None),
            "sd_checkpoint_name": data.get('sd_checkpoint_name', None),
        }

        data = emb_data

        processed = process_images(p)
        image = processed.images[0]

        title = ' '

        if 'name' in data:
            title = "<{}>".format(embedding_token)

        info = PngImagePlugin.PngInfo()
        data['name'] = embedding_token
        info.add_text("sd-ti-embedding", embedding_to_b64(data))

        try:
            vectorSize = list(data['string_to_param'].values())[0].shape[0]
        except Exception as e:
            vectorSize = None

        footer_left = checkpoint.model_name
        footer_mid = '[{}]'.format(checkpoint.hash)
        footer_right = ' '

        if vectorSize is not None:
            footer_right += '{}v'.format(vectorSize)

        if data.get('step', 0) > 0:
            footer_right += ' {}s'.format(data.get('step', 0))

        captioned_image = caption_image_overlay(image, title, footer_left, footer_mid, footer_right)
        captioned_image = insert_image_data_embed(captioned_image, data)

        captioned_image.save(os.path.join(destination_folder, embedding_token+'.png'), "PNG", pnginfo=info)

        processed.images += [captioned_image]

        return processed
