import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
import sys
import json

import openai
import tiktoken
import pydub
from pydub import AudioSegment
from pydub.silence import split_on_silence
from tiktoken.core import Encoding

# ディレクトリを作成
def mk_dir(dir_path: list or str):
    if isinstance(dir_path, str):
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
    else:
        for i in dir_path:
            if not os.path.exists(i):
                os.makedirs(i)


# 音声ファイルからテキストに変換
def audio_to_text(audio_path: str, model="whisper-1", language="ja") -> str:
    with open(audio_path, "rb") as f:
        transcript = openai.Audio.transcribe(model, f,
            prompt = "こんにちは。今日は、いいお天気ですね。")
    
    return transcript["text"]


# 音声を適切な長さに分割
def split_audio(input_audio, max_length: int, config: dict) -> list:
    # 無音部分をカットして分割
    chunks = split_on_silence(
        input_audio,
        min_silence_len = config["min_silence_len"],
        silence_thresh = config["silence_thresh"],
        keep_silence = config["keep_silence"],
    )

    num_of_chunks = len(chunks)

    # 音声データを保存
    def output_mp3(chunk, output):
        chunk.export(output, format="mp3")
    
    # 分割したチャンクをmax_lengthごとに結合
    current_chunk = None
    file_path_list = []
    file_name = os.path.splitext(config["input_audio_path"])[0].split("/")[-1]

    for i, c in enumerate(chunks):
        if current_chunk is None:
            current_chunk = c
            continue
        temp_chunk = current_chunk + c
        outFilePath = f'{config["temp_dir"]}/{file_name}_split_{i + 1}.mp3'
        if len(temp_chunk) > max_length:
            output_mp3(current_chunk, outFilePath)
            current_chunk = c
        else:
            if i == len(chunks) - 1:
                output_mp3(temp_chunk, outFilePath)
            else:
                current_chunk += c
        
        file_path_list.append(outFilePath)
    
    return file_path_list 


# 書き起こしの作成・保存
def mk_transcription(config: dict) -> str:
    src_file = config["input_audio_path"]
    file_parts = os.path.splitext(src_file)
    file_name = file_parts[0].split("/")[-1]

    # mp3以外のファイルはmp3に変換
    if file_parts[-1] != ".mp3":
        raw_audio = AudioSegment.from_file(src_file, format = file_parts[-1])
        src_file = config["temp_dir"] + file_name + ".mp3"
        raw_audio.export(src_file, format="mp3")
    
    # ファイルサイズチェック
    audio = AudioSegment.from_file(src_file, format="mp3")
    total_length = len(audio)
    total_size = os.path.getsize(src_file)
    max_file_size = 25000000 * 0.9   # Whisper APIは25MB制限, 10%ほど余裕を残す
    max_length = total_length * (max_file_size / total_size)     # ファイルサイズと時間から、分割する最大時間を取得する
    
    
    # 書き起こしの作成
    if total_size > max_file_size:
        audio_file_list = split_audio(audio, max_length, config)
        transcription_list = []
        for i in audio_file_list:
            trans = audio_to_text(i)
            transcription_list.append(transcription)
        transcription = "".join(transcription_list)

    else:
        transcription = audio_to_text(src_file)

    # 書き起こしの保存
    with open(config["result_dir"] + file_name + "_transcription.txt", "w") as f:
        f.write(transcription)

    return transcription


# 文章を最大長以下で分割
def split_text(text: str, max_length: int) -> list:
    # textの長さを取得する
    text_length = len(text)

    # 各セグメントの長さを計算する
    segment_length = max_length
    num_segments = text_length // segment_length
    if text_length % segment_length != 0:
        num_segments += 1

    # 各セグメントの開始位置と終了位置を計算する
    start = 0
    segments = []
    for i in range(num_segments):
        end = start + segment_length

        # 現在のセグメントがtextの範囲内に収まるように調整する
        if end > text_length:
            end = text_length
        if start == end:
            break

        # 現在のセグメントを取得する
        segment = text[start:end]

        # セグメントの最後の文字が全角文字であれば、それ以前までをセグメントに含める
        if len(segment) >= 2 and segment[-1] != "。":
            while len(segment) > 0 and segment[-1] != "。":
                segment = segment[:-1]
            end = start + len(segment)

        segments.append(segment)

        # 次のセグメントの開始位置を更新する
        start = end

        # 次のセグメントの長さを調整する
        remaining_length = text_length - start
        if remaining_length < max_length:
            segment_length = remaining_length

    return segments


# chatGPT APIの使用
def use_chatGPT(target_text :str, prompt_path: str, model) -> str:
    with open(prompt_path) as f:
        prompt = f.read()
    response = openai.ChatCompletion.create(
    model = model,
    messages = [
        {"role": "system", "content":"あなたは優秀な社員です。"},
        {"role": "assistant", "content": target_text},
        {"role": "user", "content": prompt}
    ])

    return response["choices"][0]["message"]["content"]



# 入力文章から議事録を作成
def mk_minutes(trans: str, config: dict):
    file_name = os.path.splitext(config["input_audio_path"])[0].split("/")[-1]

    # モデルごとの大体の最大トークン数
    model_max_token_length = {
        "gpt-3.5-turbo": 4000,
        "gpt-4": 8000,
        "gpt-4-32k": 32000
    }

    model = config["model"]
    max_token_length = model_max_token_length[model] - 700  # 入力以外のトークン分を引く 

    # トークン数のカウント
    encoding: Encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
    tokens = encoding.encode(trans)
    tokens_count = len(tokens)


    if tokens_count < max_token_length:
        gijiroku = use_chatGPT(trans, config["prompt_gijiroku"], model=model)

    elif model == "gpt-4" and config["gpt-4-32k"] == "True":
        # modelをgpt-4-32kに更新
        model = "gpt-4-32k"
        max_token_length = model_max_token_length[model] - 700
        
        if tokens_count < max_token_length:
            gijiroku = use_chatGPT(trans, config["prompt_gijiroku"], model=model)
        
        else:
            text_list = split_text(trans, int(max_token_length*0.8))     # tokenと文字数のずれを考慮し、0.8掛けて余裕を持たせる
            gijiroku_list = []
            for text in text_list:
                gijiroku = use_chatGPT(text, config["prompt_split_summary"], model=model)
                gijiroku_list.append(gijiroku)

            gijiroku_all = "\n\n\n".join(gijiroku_list)
            gijiroku = use_chatGPT(gijiroku_all , config["prompt_compile"], model=model)
            
            # 分割した議事録を保存
            with open(config["temp_dir"] + file_name + "_split_gijiroku.txt", "w") as f:
                f.write(gijiroku_all)
    
    else:
        text_list = split_text(trans, int(max_token_length*0.8))     # tokenと文字数のずれを考慮し、0.8掛けて余裕を持たせる
        gijiroku_list = []
        for text in text_list:
            gijiroku = use_chatGPT(text, config["prompt_split_summary"], model=model)
            gijiroku_list.append(gijiroku)

        gijiroku_all = "\n\n\n".join(gijiroku_list)
        gijiroku = use_chatGPT(gijiroku_all , config["prompt_compile"], model=model)
    
        # 分割した議事録を保存
        with open(config["temp_dir"] + file_name + "_split_gijiroku.txt", "w") as f:
            f.write(gijiroku_all)
    
    # 最終的な議事録を保存
    with open(config["result_dir"] + file_name + "_gijiroku.txt", "w") as f:
        f.write(gijiroku)
    

def main():
    # 設定ファイルの読み込み
    with open("./config.json", "r") as f:
        config = json.load(f)
    
    # API Keyの設定
    openai.api_key = config["openai_api_token"]
    
    # ディレクトリ設定
    mk_dir([config["result_dir"], config["temp_dir"]])

    # 書き起こしの作成
    transcription = mk_transcription(config)


    # 議事録の作成・保存
    mk_minutes(transcription, config)

if __name__ == '__main__':
    main()
