import gradio as gr
import torch
import argparse
from net_helper import net_helper
from PIL import Image
from clip_embeding import clip_embeding
from milvus_operator import text_image_vector


def image_search(text):
    if text is None:
        return None

    # clip编码
    imput_embeding = clip_embeding.embeding_text(text)
    imput_embeding = imput_embeding[0].detach().cpu().numpy()

    results = text_image_vector.search_data(imput_embeding)
    pil_images = [Image.open(result['path']) for result in results]
    return pil_images


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--share", action="store_true",
                        default=False, help="share gradio app")
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    app = gr.Blocks(theme='default', title="image",
                    css=".gradio-container, .gradio-container button {background-color: #009FCC} "
                        "footer {visibility: hidden}")
    with app:
        with gr.Tabs():
            with gr.TabItem("image search"):
                with gr.Row():
                    with gr.Column():
                        text = gr.TextArea(label="Text",
                                           placeholder="description",
                                           value="",)
                        btn = gr.Button(label="search")

                    with gr.Column():
                        with gr.Row():
                            output_images = [gr.outputs.Image(type="pil", label=None) for _ in range(16)]

                btn.click(image_search, inputs=[text], outputs=output_images, show_progress=True)

    ip_addr = net_helper.get_host_ip()
    app.queue(concurrency_count=3).launch(show_api=False, share=True, server_name=ip_addr, server_port=9099)
