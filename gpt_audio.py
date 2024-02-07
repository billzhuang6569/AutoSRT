from openai import OpenAI
from dotenv import load_dotenv
import os
import random
import string
import datetime

# 加载环境变量
load_dotenv()
openai_api_key = os.getenv('openai_api_key')



def generate_jobid():
    jobid = ''.join(random.sample(string.ascii_letters + string.digits, 6))
    date = datetime.datetime.now().strftime('%Y%m%d')
    result = date + '_job' + jobid
    return result


# 创建音频转录
# audio_filepath: 音频文件路径
# language: 语言，可空，使用ISO-639-1标准，中文为zh，英文为en，日文为ja，韩文为ko-KR
# https://en.wikipedia.org/wiki/List_of_ISO_639_language_codes
# prompt: 提示，可空，用于指导模型风格或继续之前音频片段的可选文本。提示应与音频语言匹配。
# response_format: 响应格式，可空。json/text/srt/vtt/verbose_json
def transcription(audio_filepath, prompt, response_format):
    client = OpenAI(api_key=openai_api_key)
    audio_file = open(audio_filepath, 'rb')
    transcript = client.audio.transcriptions.create(
        model='whisper-1',
        file=audio_file,
        prompt=prompt,
        response_format=response_format
    )
    return transcript

# 写入SRT文件
def write_srt(transcript, srt_filepath):
    try:
        with open(srt_filepath, 'w', encoding='utf-8') as file:
            file.write(transcript)
    except IOError as e:
        print(f"写入文件时出错: {e}")


def audio_to_srt(audio_filepath, prompt, response_format, srt_filepath):
    a = transcription(audio_filepath, prompt, response_format)
    b = write_srt(a, srt_filepath)
    return b



def audio_to_list(audio_filepath, prompt, list_filepath):
    # 输出目录list_folder
    list_folder = os.dirname(list_filepath)
    if not os.path.exists(list_folder):
        os.makedirs(list_folder)
    
    # 获取list_filepath的文件名
    list_filename = os.path.basename(list_filepath)
    filename_without_ext = os.path.splitext(list_filename)[0]

    # 确定SRT输出目录
    srt_filename = filename_without_ext + ".srt"
    srt_output = os.path.join(list_folder, srt_filename)

    # 获取转写，转换为SRT
    transcript = transcription(audio_filepath, prompt, response_format='srt')
    write_srt(transcript, srt_output)

    # 读取SRT文件，转换为list
    with open(srt_output, 'r', encoding='utf-8') as file:
        srt = file.readlines()