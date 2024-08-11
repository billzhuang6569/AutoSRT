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

