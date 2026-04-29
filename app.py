# Copyright (c) Alibaba Cloud.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import os
import sys
import time
from pathlib import Path
import tempfile
import shutil
import cv2
import gradio as gr
import torch
from argparse import ArgumentParser
from modelscope import AutoModelForCausalLM, AutoTokenizer, GenerationConfig, snapshot_download

# Set to use only the A6000 (GPU 1)
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

# Available models with their paths and descriptions
AVAILABLE_MODELS = {
    'Qwen-VL-Int4 (4GB VRAM, Faster)': 'Qwen/Qwen-VL-Chat-Int4',
    'Qwen-VL-Chat (8GB+ VRAM, Better Quality)': 'Qwen/Qwen-VL-Chat',
    'Qwen1.5-VL-Chat (Latest, 8GB+ VRAM)': 'Qwen/Qwen1.5-VL-Chat',
}

# Default model
DEFAULT_MODEL = 'Qwen1.5-VL-Chat (Latest, 8GB+ VRAM)'

# Global variables
model = None
tokenizer = None
current_model_name = None
from argparse import ArgumentParser
from pathlib import Path

import copy
import gradio as gr
import os
import re
import secrets
import tempfile
import shutil
import cv2
from video_utils import _extract_frames
from modelscope import (
    AutoModelForCausalLM, AutoTokenizer, GenerationConfig
)
from huggingface_hub import snapshot_download

DEFAULT_CKPT_PATH = '4bit/Qwen-VL-Chat-Int4'
REVISION = 'v1.0.0'
BOX_TAG_PATTERN = r"<box>([\s\S]*?)</box>"
PUNCTUATION = "！？。＂＃＄％＆＇（）＊＋，－／：；＜＝＞＠［＼］＾＿｀｛｜｝～｟｠｢｣､、〃》「」『』【】〔〕〖〗〘〙〚〛〜〝〞〟〰〾〿–—‘’‛“”„‟…‧﹏."


def _get_args():
    parser = ArgumentParser()
    parser.add_argument("--cpu-only", action="store_true", help="Run demo with CPU only")
    parser.add_argument("--share", action="store_true", default=False,
                        help="Create a publicly shareable link for the interface.")
    parser.add_argument("--inbrowser", action="store_true", default=False,
                        help="Automatically launch the interface in a new tab on the default browser.")
    parser.add_argument("--server-port", type=int, default=8000,
                        help="Demo server port.")
    parser.add_argument("--server-name", type=str, default="127.0.0.1",
                        help="Demo server name.")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL,
                        help=f"Default model to use. Available: {', '.join(AVAILABLE_MODELS.keys())}")
    args = parser.parse_args()
    return args


def _load_model_tokenizer(model_name, cpu_only=False):
    global model, tokenizer, current_model_name
    
    if model_name == current_model_name and model is not None and tokenizer is not None:
        return model, tokenizer
        
    print(f"Loading model: {model_name}")
    start_time = time.time()
    
    # Clear CUDA cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # Get the model path from available models
    model_id = AVAILABLE_MODELS.get(model_name)
    if not model_id:
        raise ValueError(f"Model {model_name} not found in available models")
    
    try:
        # Download and load the model
        model_dir = snapshot_download(model_id)
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            model_dir, 
            trust_remote_code=True, 
            resume_download=True,
        )

        # Set device
        if cpu_only or not torch.cuda.is_available():
            device_map = "cpu"
        else:
            device_map = "auto"

        # Load model
        model = AutoModelForCausalLM.from_pretrained(
            model_dir,
            device_map=device_map,
            trust_remote_code=True,
            resume_download=True,
        ).eval()
        
        # Load generation config
        model.generation_config = GenerationConfig.from_pretrained(
            model_dir, 
            trust_remote_code=True, 
            resume_download=True,
        )
        
        current_model_name = model_name
        print(f"Model loaded in {time.time() - start_time:.2f} seconds")
        return model, tokenizer
        
    except Exception as e:
        print(f"Error loading model: {e}")
        raise


def _parse_text(text):
    lines = text.split("\n")
    lines = [line for line in lines if line != ""]
    count = 0
    for i, line in enumerate(lines):
        if "```" in line:
            count += 1
            items = line.split("`")
            if count % 2 == 1:
                lines[i] = f'<pre><code class="language-{items[-1]}">'
            else:
                lines[i] = f"<br></code></pre>"
        else:
            if i > 0:
                if count % 2 == 1:
                    line = line.replace("`", r"\`")
                    line = line.replace("<", "&lt;")
                    line = line.replace(">", "&gt;")
                    line = line.replace(" ", "&nbsp;")
                    line = line.replace("*", "&ast;")
                    line = line.replace("_", "&lowbar;")
                    line = line.replace("-", "&#45;")
                    line = line.replace(".", "&#46;")
                    line = line.replace("!", "&#33;")
                    line = line.replace("(", "&#40;")
                    line = line.replace(")", "&#41;")
                    line = line.replace("$", "&#36;")
                lines[i] = "<br>" + line
    text = "".join(lines)
    return text


def _launch_demo(args):
    uploaded_file_dir = os.environ.get("GRADIO_TEMP_DIR") or str(
        Path(tempfile.gettempdir()) / "gradio"
    )

    def predict(_chatbot, task_history):
        chat_query = _chatbot[-1][0]
        query = task_history[-1][0]
        print("User: " + _parse_text(query))
        history_cp = copy.deepcopy(task_history)
        full_response = ""

        history_filter = []
        pic_idx = 1
        pre = ""
        for i, (q, a) in enumerate(history_cp):
            if isinstance(q, (tuple, list)):
                q = f'Picture {pic_idx}: <img>{q[0]}</img>'
                pre += q + '\n'
                pic_idx += 1
            else:
                pre += q
                history_filter.append((pre, a))
                pre = ""
        history, message = history_filter[:-1], history_filter[-1][0]
        response, history = model.chat(tokenizer, message, history=history)
        image = tokenizer.draw_bbox_on_latest_picture(response, history)
        if image is not None:
            temp_dir = secrets.token_hex(20)
            temp_dir = Path(uploaded_file_dir) / temp_dir
            temp_dir.mkdir(exist_ok=True, parents=True)
            name = f"tmp{secrets.token_hex(5)}.jpg"
            filename = temp_dir / name
            image.save(str(filename))
            _chatbot[-1] = (_parse_text(chat_query), (str(filename),))
            chat_response = response.replace("<ref>", "")
            chat_response = chat_response.replace(r"</ref>", "")
            chat_response = re.sub(BOX_TAG_PATTERN, "", chat_response)
            if chat_response != "":
                _chatbot.append((None, chat_response))
        else:
            _chatbot[-1] = (_parse_text(chat_query), response)
        full_response = _parse_text(response)

        task_history[-1] = (query, full_response)
        print("Qwen-VL-Chat: " + _parse_text(full_response))
        task_history = task_history[-10:]
        return _chatbot

    def regenerate(_chatbot, task_history):
        if not task_history:
            return _chatbot
        item = task_history[-1]
        if item[1] is None:
            return _chatbot
        task_history[-1] = (item[0], None)
        chatbot_item = _chatbot.pop(-1)
        if chatbot_item[0] is None:
            _chatbot[-1] = (_chatbot[-1][0], None)
        else:
            _chatbot.append((chatbot_item[0], None))
        return predict(_chatbot, task_history)

    def add_text(history, task_history, text):
        task_text = text
        if len(text) >= 2 and text[-1] in PUNCTUATION and text[-2] not in PUNCTUATION:
            task_text = text[:-1]
        history = history + [(_parse_text(text), None)]
        task_history = task_history + [(task_text, None)]
        return history, task_history, ""

    def add_file(history, task_history, file):
        # Check if the file is a video
        video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.webm']
        file_ext = os.path.splitext(file.name)[1].lower()
        
        if file_ext in video_extensions:
            # Create a temporary directory for extracted frames
            temp_dir = tempfile.mkdtemp()
            # Extract frames from video
            frame_paths = _extract_frames(file.name, temp_dir)
            if frame_paths:
                # Use the first frame for now (you can modify to use more frames)
                file_path = frame_paths[0]
                history = history + [((file_path,), None)]
                task_history = task_history + [((file_path,), None)]
                return history, task_history
        
        # For non-video files or if frame extraction fails, use the file as is
        history = history + [((file.name,), None)]
        task_history = task_history + [((file.name,), None)]
        return history, task_history

    def reset_user_input():
        return gr.update(value="")

    def reset_state(task_history):
        task_history.clear()
        return []

    def load_model(model_name):
        try:
            global model, tokenizer
            model, tokenizer = _load_model_tokenizer(model_name, args.cpu_only)
            return f"Model loaded: {model_name}"
        except Exception as e:
            return f"Error loading model: {str(e)}"

    with gr.Blocks(title="Qwen-VL Chat") as demo:
        gr.Markdown(
            """\
            <center><font size=4>Qwen-VL Chat Demo</font></center>
            <center><font size=2>Select a model and click 'Load Model' to start chatting</font></center>
            """
        )
        
        with gr.Row():
            model_dropdown = gr.Dropdown(
                choices=list(AVAILABLE_MODELS.keys()),
                value=args.model,
                label="Select Model",
                interactive=True
            )
            load_btn = gr.Button("Load Model")
            model_status = gr.Textbox(value="", label="Status", interactive=False)
        
        load_btn.click(
            load_model,
            inputs=[model_dropdown],
            outputs=[model_status]
        )
        
        # Only show chat interface after model is loaded
        with gr.Group(visible=False) as chat_interface:
            gr.Markdown(
                """\
                <center><font size=3>This WebUI is based on Qwen-VL-Chat, developed by Alibaba Cloud. \
                (本WebUI基于Qwen-VL-Chat打造，实现聊天机器人功能。)</center>"""
            )
            gr.Markdown("""\
                <center><font size=4>Qwen-VL <a href="https://modelscope.cn/models/qwen/Qwen-VL/summary">🤖 </a> 
                | <a href="https://huggingface.co/Qwen/Qwen-VL">🤗</a>&nbsp ｜ 
                Qwen-VL-Chat <a href="https://modelscope.cn/models/qwen/Qwen-VL-Chat/summary">🤖 </a> | 
                <a href="https://huggingface.co/Qwen/Qwen-VL-Chat">🤗</a>&nbsp ｜ 
                &nbsp<a href="https://github.com/QwenLM/Qwen-VL">Github</a></center>"""
            )

            chatbot = gr.Chatbot(label='Qwen-VL-Chat', height=500, elem_classes=["control-height"])
            query = gr.Textbox(lines=2, label='Input')
            task_history = gr.State([])

            with gr.Row():
                addfile_btn = gr.UploadButton("📁 Upload (上传文件)", file_types=["image", "video/mp4", "video/avi", "video/quicktime"])
                submit_btn = gr.Button("🚀 Submit (发送)")
                regen_btn = gr.Button("🤔️ Regenerate (重试)")
                empty_bin = gr.Button("🧹 Clear History (清除历史)")

            submit_btn.click(add_text, [chatbot, task_history, query], [chatbot, task_history]).then(
                predict, [chatbot, task_history], [chatbot], show_progress=True
            )
            submit_btn.click(reset_user_input, [], [query])
            empty_bin.click(reset_state, [task_history], [chatbot], show_progress=True)
            regen_btn.click(regenerate, [chatbot, task_history], [chatbot], show_progress=True)
            addfile_btn.upload(add_file, [chatbot, task_history, addfile_btn], [chatbot, task_history], show_progress=True)

            gr.Markdown("""\
                <font size=2>Note: This demo is governed by the original license of Qwen-VL. \
                We strongly advise users not to knowingly generate or allow others to knowingly generate harmful content, \
                including hate speech, violence, pornography, deception, etc. \
                (注：本演示受Qwen-VL的许可协议限制。我们强烈建议，用户不应传播及不应允许他人传播以下内容，\
                包括但不限于仇恨言论、暴力、色情、欺诈相关的有害信息。)"""
            )

        # Load the default model at startup
        demo.load(
            fn=load_model,
            inputs=[model_dropdown],
            outputs=[model_status],
            queue=False
        )
        
        # Toggle chat interface visibility based on model loading
        def toggle_chat_interface(model_status):
            if model_status and "loaded" in model_status.lower():
                return gr.Group(visible=True)
            return gr.Group(visible=False)
        
        demo.load(
            fn=toggle_chat_interface,
            inputs=[model_status],
            outputs=[chat_interface],
            queue=False
        )
    
    # Enable sharing and set server configuration
    demo.queue(max_size=100).launch(
        share=True,  # Always enable sharing to handle localhost access issues
        server_name=args.server_name,
        server_port=args.server_port,
        inbrowser=not args.no_auto_launch,
        max_threads=200,
        debug=True
    )


def main():
    args = _get_args()
    _launch_demo(args)


if __name__ == '__main__':
    main()