# tools.py
from openai import OpenAI
# from dotenv import load_dotenv
import conf
import json
import datetime
import ast
import random
import string
import re

openai_api_key = conf.openai_api_key
openai_api_url = conf.openai_api_url

client = OpenAI(api_key=openai_api_key, base_url=openai_api_url)

# 创建默认的音频转录
# audio_filepath: 音频文件路径
# language: 语言，可空，使用ISO-639-1标准，中文为zh，英文为en，日文为ja，韩文为ko-KR
# https://en.wikipedia.org/wiki/List_of_ISO_639_language_codes
# prompt: 提示，可空，用于指导模型风格或继续之前音频片段的可选文本。提示应与音频语言匹配。
# response_format: 响应格式，可空。json/text/srt/vtt/verbose_json
# timestamp_granularities: 时间戳粒度，可空。word/segment
def transcribe_audio(audio_filepath, prompt, response_format, timestamp_granularities):
    audio_file = open(audio_filepath, 'rb')
    transcript = client.audio.transcriptions.create(
        model='whisper-1',
        file=audio_file,
        prompt=prompt,
        response_format=response_format,
        timestamp_granularities=[timestamp_granularities]
    )
    return transcript

# 生成示例字幕
# user_prompt: 用户输入的提示词
# format_requirements: 格式要求
def generate_whisper_prompt(user_prompt, format_requirements):
    # 提取关键词
    extract_keywords_prompt = f"""
    From the following text, extract key terms that might be prone to transcription errors. 
    These could include specialized terminology, proper nouns, brand names, place names, or any uncommon words.
    
    Text: {user_prompt}
    
    Return ONLY a JSON array of extracted terms, without any additional text or explanation. For example:
    ["term1", "term2", "term3"]
    """
    keywords_response = client.chat.completions.create(
        model="gpt-4-turbo",
        messages=[
            {"role": "system", "content": "You are an assistant that extracts keywords and returns them in a strict JSON array format."},
            {"role": "user", "content": extract_keywords_prompt}
        ]
    )

    response_content = keywords_response.choices[0].message.content.strip()

    # 使用正则表达式提取JSON数组
    json_match = re.search(r'\[.*\]', response_content, re.DOTALL)
    if json_match:
        try:
            keywords = json.loads(json_match.group())
        except json.JSONDecodeError:
            print("Error: Unable to parse JSON array from response")
            return None
    else:
        print("Error: No JSON array found in response")
        return None

    # 生成示例字幕
    generate_subtitle_prompt = f"""
    Create a sample subtitle of no more than 300 characters that includes the following keywords: {', '.join(keywords)}
    
    The subtitle should follow these format requirements:
    {json.dumps(format_requirements, ensure_ascii=False, indent=2)}
    
    Return ONLY the subtitle text, without any additional explanation or formatting.
    """

    subtitle_response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are an assistant that creates sample subtitles. Respond only with the subtitle text."},
            {"role": "user", "content": generate_subtitle_prompt}
        ]
    )

    sample_subtitle = subtitle_response.choices[0].message.content.strip()

    if len(sample_subtitle) > 300:
        sample_subtitle = sample_subtitle[:300]

    return sample_subtitle



# 使用GPT处理转录结果，返回SRT格式字幕
# transcription_result: 转录结果
# user_prompt: 用户输入的提示词
# format_requirements: 格式要求 
def process_with_gpt(transcription_result, user_prompt, format_requirements):
    # 提取纯文本
    plain_text = transcription_result.text

    # 将格式要求转换为易读的字符串
    formatted_requirements = json.dumps(format_requirements, ensure_ascii=False, indent=2)

    # 第一步：处理纯文本，进行分行和断句
    text_processing_prompt = f"""
    请根据以下规则处理给定的文本：
    

    1. 符合字幕阅读习惯地，将文本分割成多个字幕行，每行理想长度为5-15个汉字。
    2. 断句规则:
        a. 优先在完整句子结束时断开字幕。
        b. 当一行长度过长需要断开时：
            - 考虑前后两句的字数均匀，无需严格均匀，但尽量均匀。
            - 在自然停顿处或从句之间断开。若需要从”的“”了“”和“处断开，则将"的"、"了"、"和"、"是"等虚词放在前半句。
        c. 优先考虑句子完整性和自然断句，而非严格的字符限制。
    3. 避免出现一行字幕只有一个词语，避免前一行的末尾出现了下一行的第一个词语
    4. 遵循以下格式要求：
    {formatted_requirements}

    原文：
    {plain_text}

    请输出处理后的纯文本，每行一个字幕，不要添加行号或时间戳。
    不包含任何解释或额外格式化。不使用```或其他格式标记。
    """

    text_processing_response = client.chat.completions.create(
        model="gpt-4-turbo",
        messages=[
            {"role": "system", "content": "你是一个专业的字幕编辑，擅长将长文本分割成简洁、易读的字幕行。"},
            {"role": "user", "content": text_processing_prompt}
        ]
    )

    processed_text = text_processing_response.choices[0].message.content.strip()

    # 第二步：根据处理后的文本和时间戳信息创建 SRT 字幕
    srt_creation_prompt = f"""
    请使用以下处理过的文本和原始转录中的时间戳信息创建 SRT 格式的字幕：

    处理后的文本（每行一个字幕）：
    {processed_text}

    原始转录（包含时间戳信息）：
    {transcription_result}

    创建 SRT 字幕时，请遵循以下规则：
    1. 使用原始转录中的精确时间戳。
    2. 每个字幕的开始时间应为该行第一个词的开始时间，结束时间应为最后一个词的结束时间。
    3. 按照 SRT 格式严格编号和格式化时间戳。
    4. 确保字幕内容与处理后的文本完全一致。

    请直接输出纯粹的SRT格式的纯文本，不需要任何额外解释。不包含任何解释或额外格式化。不使用```或其他格式标记。

    """

    srt_creation_response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "你是一个精通 SRT 字幕格式的专家，能够准确地将文本和时间戳信息转换为 SRT 格式的字幕。请直接输出纯粹的SRT格式的纯文本，不需要任何额外解释。不包含任何解释或额外格式化。不使用```或其他格式标记。"},
            {"role": "user", "content": srt_creation_prompt}
        ]
    )

    return srt_creation_response.choices[0].message.content.strip()


# 验证SRT格式字幕
# srt_content: SRT格式字幕
# format_requirements: 格式要求 
def validate_srt(srt_content):
    print("Validating SRT format...")
    lines = srt_content.split('\n')
    
    subtitle_pattern = re.compile(r'^\d+$')
    time_pattern = re.compile(r'^\d{2}:\d{2}:\d{2},\d{3} --> \d{2}:\d{2}:\d{2},\d{3}$')
    
    i = 0
    while i < len(lines):
        # 检查字幕序号
        if not subtitle_pattern.match(lines[i].strip()):
            print(f"Invalid SRT: Line {i+1} is not a valid subtitle number")
            return False, f"Line {i+1} is not a valid subtitle number"
        i += 1
        
        # 检查时间码
        if i >= len(lines) or not time_pattern.match(lines[i].strip()):
            print(f"Invalid SRT: Line {i+1} does not match the required timestamp format")
            return False, f"Line {i+1} does not match the required timestamp format"
        i += 1
        
        # 检查字幕文本（可能跨多行）
        if i >= len(lines):
            print(f"Invalid SRT: Missing subtitle text after timestamp")
            return False, f"Missing subtitle text after timestamp"
        
        while i < len(lines) and lines[i].strip() != '':
            i += 1
        
        # 检查空行（除非是最后一个字幕）
        if i < len(lines):
            if lines[i].strip() != '':
                print(f"Invalid SRT: Expected empty line at line {i+1}")
                return False, f"Expected empty line at line {i+1}"
            i += 1

    print("SRT format validation successful.")
    return True, "SRT format is valid"

def save_srt(srt_content, output_filepath):
    with open(output_filepath, 'w', encoding='utf-8') as f:
        f.write(srt_content)


def generate_jobid():
    jobid = ''.join(random.sample(string.ascii_letters + string.digits, 6))
    date = datetime.datetime.now().strftime('%Y%m%d')
    result = date + '_job' + jobid
    return result