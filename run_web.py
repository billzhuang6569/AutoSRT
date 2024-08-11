import gradio as gr
import tools
import os
from pydub import AudioSegment
import main
import base64
import conf

# 获取当前Python文件的绝对路径
py_abspath = os.path.abspath(__file__)
py_dir = os.path.dirname(py_abspath)
os.chdir(py_dir)

def save_audio(jobid, numpy_audio, sample_rate):
    channels = 1 if numpy_audio.ndim == 1 else numpy_audio.shape[1]
    audio_segment = AudioSegment(
        numpy_audio.tobytes(),
        frame_rate=sample_rate,
        sample_width=numpy_audio.dtype.itemsize,
        channels=channels
    )
    format = "mp3"
    filename = f"audio_{jobid}.{format}"
    audio_output_path = os.path.join("output", "audio", filename)
    if not os.path.exists(os.path.dirname(audio_output_path)):
        os.makedirs(os.path.dirname(audio_output_path))
    audio_segment.export(audio_output_path, format=format, bitrate="128k")
    return audio_output_path

def format_requirements_to_html(format_requirements):
    html = """
    <div style='margin-bottom: 10px; font-size: 14px; color: #888;'>当前已生效的字幕格式</div>
    <div style='display: flex; flex-wrap: wrap; gap: 20px;'>
    """
    for key, value in format_requirements.items():
        html += f"""
        <div style='
            background: linear-gradient(135deg, #6e8efb, #a777e3);
            background-size: 400% 400%;
            animation: gradient 15s ease infinite;
            border-radius: 15px;
            padding: 15px;
            color: white;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            width: 200px;
        '>
            <h3 style='margin: 0; font-size: 14px; font-weight: bold; color: #e0e0e0;'>{key}</h3>
            <p style='margin: 10px 0 0; font-size: 16px; font-weight: 300;'>{value}</p>
        </div>
        """
    html += """
    </div>
    <style>
    @keyframes gradient {
        0% {background-position: 0% 50%;}
        50% {background-position: 100% 50%;}
        100% {background-position: 0% 50%;}
    }
    </style>
    """
    return html

def fn_gpt_audio(audio_numpy, user_prompt, format_requirements):
    try:
        sample_rate, numpy_audio = audio_numpy
    except Exception as e:
        return None, "请等待上传完成"

    job_id = tools.generate_jobid()
    audio_filepath = save_audio(job_id, numpy_audio, sample_rate)

    srt_filepath = main.main(audio_filepath, 
                    user_prompt, 
                    "verbose_json", 
                    timestamp_granularities="word", 
                    format_requirements=format_requirements
                        )
    return srt_filepath, f"SRT 文件已生成：{os.path.basename(srt_filepath)}"

def add_format_requirement(format_requirements, subtitle_format, requirement):
    new_requirements = format_requirements.copy()
    new_requirements[subtitle_format] = requirement
    return new_requirements

def remove_last_format_requirement(format_requirements):
    new_requirements = format_requirements.copy()
    if new_requirements:
        new_requirements.popitem()
    return new_requirements


# 读取并编码logo
def get_image_base64(image_path):
    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
    return f"data:image/png;base64,{encoded_string}"

logo_path = os.path.join(py_dir, "logo.png")
logo_base64 = get_image_base64(logo_path)


with gr.Blocks() as demo:
    gr.Markdown(
        f"""
        <div style="display: flex; align-items: center; justify-content: space-between; padding: 20px; background-color: #f0f0f0; border-radius: 10px;">
            <div style="flex: 1;">
                <h1 style="margin: 0; color: #333;">WhatOnEarth一探究竟 - AutoSRT</h1>
                <p style="margin: 10px 0; color: #666;">
                    🎙️ 使用先进的人工智能，识别音频转写<br>
                    📝 自由添加字幕格式规范，按照生产要求输出SRT<br>
                    🔍 自由添加参考文本、名词，让翻译准确无误<br>
                    💯 完全免费可商用，上手即用
                    🤖 WOE团队借助人工智能开发
                </p>
            </div>
            <a href="https://www.woe.show" target="_blank">
                <img src="{logo_base64}" alt="WhatOnEarth Logo" style="width: 200px; height: 200px; object-fit: contain;">
            </a>
        </div>
        """
    )
    
    with gr.Column():
        gr.Markdown("## STEP 1 上传音频")
        with gr.Row():
            audio_input = gr.Audio(type="numpy", label="上传音频文件")
            prompt_input = gr.Textbox(label="提示词", placeholder="文章原文/关键词/使用场景", lines=5)
    
    with gr.Column():
        gr.Markdown("## STEP 2 字幕格式")
        format_requirements = gr.State({
            "每行最多字数": 20, 
            "标点符号": "不能出现任何标点符号（逗号、句号），只允许出现上引号【“】、下引号【”】、书名号【《》】，顿号【、】由空格【 】替代"
        })
        format_display = gr.HTML()

        with gr.Column():
            gr.Markdown("### 添加格式要求")
            with gr.Row():
                subtitle_format = gr.Textbox(label="字幕格式")
                requirement = gr.Textbox(label="有何要求")
            with gr.Row():
                add_btn = gr.Button("添加格式要求")
                remove_btn = gr.Button("删除最后一个格式要求")
    
    with gr.Column():
        gr.Markdown("## STEP 3 生成结果")
        output_file = gr.File(label="下载 SRT 文件")
        output_text = gr.Textbox(label="处理结果")
    
    submit_btn = gr.Button("提交")

    def update_format_display(format_requirements):
        return format_requirements_to_html(format_requirements)

    demo.load(update_format_display, inputs=[format_requirements], outputs=[format_display])

    add_btn.click(
        lambda x, y, z: (add_format_requirement(x, y, z), update_format_display(add_format_requirement(x, y, z))),
        inputs=[format_requirements, subtitle_format, requirement],
        outputs=[format_requirements, format_display]
    )

    remove_btn.click(
        lambda x: (remove_last_format_requirement(x), update_format_display(remove_last_format_requirement(x))),
        inputs=[format_requirements],
        outputs=[format_requirements, format_display]
    )

    submit_btn.click(
        fn_gpt_audio,
        inputs=[audio_input, prompt_input, format_requirements],
        outputs=[output_file, output_text]
    )

if __name__ == "__main__":
    gradio_port = conf.gradio_port
    demo.launch(server_port=gradio_port)
