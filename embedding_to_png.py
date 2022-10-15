import modules.scripts as scripts
import gradio as gr
import os
import torch
from modules import sd_models, sd_hijack
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
        destination_folder = gr.Textbox(label="Output directory",value=r"outputs")
        return [embedding,embedding_token,destination_folder]

    def run(self, p, embedding,embedding_token,destination_folder):
        print(embedding,embedding_token,destination_folder)
        assert os.path.exists(destination_folder)
        sd_hijack.model_hijack.embedding_db.load_textual_inversion_embeddings()

        p.width=512
        p.height=512

        embedding_token = embedding_token.replace('<','').replace('>','').strip()

        try:
            data = torch.load(embedding.name)
        except Exception as e:
            data = extract_image_data_embed(Image.open(embedding.name))

        assert data is not None
        
        processed = process_images(p)
        image = processed.images[0]

        info = PngImagePlugin.PngInfo()
        
        info.add_text("sd-ti-embedding", embedding_to_b64(data))

        title = "<{}>".format(embedding_token)

        try:
            vectorSize = list(data['string_to_param'].values())[0].shape[0]
        except Exception as e:
            vectorSize = '?'

        checkpoint = sd_models.select_checkpoint()
        footer_left = checkpoint.model_name
        footer_mid = '[{}]'.format(checkpoint.hash)
        footer_right = '{}v {}s'.format(vectorSize, data.get('step', 0))

        captioned_image = caption_image_overlay(image, title, footer_left, footer_mid, footer_right)
        captioned_image = insert_image_data_embed(captioned_image, data)

        captioned_image.save(os.path.join(destination_folder,embedding_token+'.png'), "PNG", pnginfo=info)

        processed.images += [captioned_image]

        return processed
    
    