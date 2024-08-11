# main.py

from tools import generate_whisper_prompt, transcribe_audio, process_with_gpt, validate_srt, save_srt
import os

def main(audio_filepath, user_prompt, response_format, timestamp_granularities, format_requirements):

    # 生成Whisper Prompt
    print(f"Processing audio file: {audio_filepath}")
    whisper_prompt = generate_whisper_prompt(user_prompt, format_requirements)
    
    # 音频转录
    print(f"Transcribing audio file: {audio_filepath}")
    transcription_result = transcribe_audio(audio_filepath, whisper_prompt, response_format, timestamp_granularities)
    
    # GPT处理
    print(f"Processing with GPT")
    srt_content = process_with_gpt(transcription_result, user_prompt, format_requirements)
    
    # 验证SRT
    print(f"Validating SRT: {srt_content}")
    is_valid, message = validate_srt(srt_content)
    
    if not is_valid:
        print(f"SRT validation failed: {message}")
        # 这里可以添加重试逻辑或错误处理
        return None
    
    # 保存SRT文件
    print(f"Saving SRT file: {srt_content}")
    output_filepath = os.path.splitext(audio_filepath)[0] + ".srt"
    save_srt(srt_content, output_filepath)
    
    print(f"SRT file has been created: {output_filepath}")
    
    return output_filepath

# 示例用法
if __name__ == "__main__":
    audio_filepath = "audio_20240207_job32rMAU.mp3"
    user_prompt = "腾讯，人工智能，行业大模型，京东，数科"
    response_format = "verbose_json"
    timestamp_granularities = "word"
    format_requirements = {
        "每行最多字数": 20,
        "标点符号": r"不能出现任何标点符号（逗号、句号），只允许出现上引号【”】、下引号【“】、书名号【《》】，顿号【、】由空格【 】替代",
        "语言": "简体中文为主，可能掺杂少量英文"
    }
    
    result = main(audio_filepath, user_prompt, response_format, timestamp_granularities, format_requirements)
    if result:
        print(f"Processing completed. SRT file saved at: {result}")
    else:
        print("Processing failed.")

