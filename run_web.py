import gradio as gr
import gpt_audio
import os
from pydub import AudioSegment


def save_audio(jobid, numpy_audio, sample_rate):
    # 检查音频声道数
    channels = 1 if numpy_audio.ndim == 1 else numpy_audio.shape[1]

    # 将Numpy数组转换为音频
    audio_segment = AudioSegment(
        numpy_audio.tobytes(),
        frame_rate=sample_rate,
        sample_width=numpy_audio.dtype.itemsize,
        channels=channels
    )
    # 文件名
    format = "mp3"
    filename = f"audio_{jobid}.{format}"
    # 保存路径
    audio_output_path = os.path.join("output", "audio", filename)
    if not os.path.exists(os.path.dirname(audio_output_path)):
        os.makedirs(os.path.dirname(audio_output_path))

    # 保存
    audio_segment.export(audio_output_path, format=format, bitrate="128k")
    return audio_output_path


def fn_gpt_audio(audio_numpy, prompt, response_format):
    sample_rate, numpy_audio = audio_numpy
    job_id = gpt_audio.generate_jobid()
    audio_filepath = save_audio(job_id, numpy_audio, sample_rate)

    if response_format == "srt":
        srt_filename = str(job_id) + ".srt"
        srt_output = os.path.join("output", "srt", srt_filename)
        if not os.path.exists(os.path.dirname(srt_output)):
            os.makedirs(os.path.dirname(srt_output))

        # 增加prompt格式要求
        # 读取srt_format.md
        with open("srt_format.md", "r", encoding="utf-8") as f:
            srt_format = f.read()
        prompt_modified = prompt + srt_format

        print(f"即将发送给Whisper，内容如下：")
        print(f"audio_filepath: {audio_filepath}")
        print(f"audio_大小：{os.path.getsize(audio_filepath)}")
        print(f"prompt: {prompt_modified}")
        print(f"response_format: {response_format}")

        audio_filepath = str(audio_filepath)

        transcript = gpt_audio.transcription(audio_filepath, prompt_modified, response_format)
        gpt_audio.write_srt(transcript, srt_output)

        # 返回srt_output的绝对路径
        srt_path = os.path.abspath(srt_output)

        return transcript, srt_path

    else:
        transcript = gpt_audio.transcription(audio_filepath, prompt, response_format)
        srt_path= ""
        return transcript, srt_path


def show_srt_filename(output_format):
    if output_format == "srt":
        return True
    else:
        return False


web_gpt_audio = gr.Interface(
    fn=fn_gpt_audio,
    inputs=[
        gr.Audio(sources="upload", type="numpy", label="上传音频文件"),
        gr.Textbox(lines=2, placeholder="稿件原文/主题/关键词等", label="输入与音频有关的提示文本"),
        gr.Radio(["srt", "text", "json", "verbose_json"], label="选择输出格式"),
    ],
    outputs=[
        gr.Textbox(label="转写结果"),
        "file"
    ],
    title="WhatOnEarth - 音频转写",
    description="使用 Whisper 模型：转写+打轴+下载"
)


web_gpt_audio.launch()

