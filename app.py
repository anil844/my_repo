import gradio as gr
from transformers import AutoProcessor, SeamlessM4Tv2Model
import torchaudio
import numpy as np
import torch

# Initialize processor and model
processor = AutoProcessor.from_pretrained("facebook/seamless-m4t-v2-large")
model = SeamlessM4Tv2Model.from_pretrained("facebook/seamless-m4t-v2-large")

# Language dictionary for selection
lang_dict = {
    'English': 'eng',
    'Russian': 'rus',
    'Spanish': 'spa',
    'Bangla': 'ben',
    'Tamil': 'tam',
    'Gujarati': 'guj',
    'Punjabi': 'pan',
    'Hindi': 'hin',
    'Telugu': 'tel',
    'Kannada': 'kan'
    # Add more languages as needed
}

def translate_audio_real_time(source_lang, target_lang, audio):
    audio = np.array(audio)
    audio = torch.from_numpy(audio).unsqueeze(0)  # Add batch dimension
    audio_inputs = processor(audios=audio, return_tensors="pt")
    translated_audio = model.generate(**audio_inputs, tgt_lang=lang_dict[target_lang])[0].cpu().numpy().squeeze()
    return (16000, translated_audio)

# Gradio interface for real-time speech-to-speech translation
with gr.Blocks() as demo:
    gr.Markdown("# Real-Time Speech-to-Speech Translation")

    with gr.Row():
        source_lang = gr.Dropdown(label="Select Source Language", choices=list(lang_dict.keys()))
        target_lang = gr.Dropdown(label="Select Target Language", choices=list(lang_dict.keys()))

    with gr.Tab("Real-Time Audio"):
        audio_input = gr.Audio(type="numpy", streaming=True, label="Speak to Translate")
        audio_output = gr.Audio(label="Translated Audio")
        audio_input.stream(translate_audio_real_time, inputs=[source_lang, target_lang, audio_input], outputs=audio_output)

demo.launch()
