import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
import sys
import json
import shutil
from typing import Any

import openai
import tiktoken
import pydub
from pydub import AudioSegment
from pydub.silence import split_on_silence
from tiktoken.core import Encoding
from moviepy.editor import VideoFileClip


def mk_dir(dir_path: list or str):
    """"
    必要なディレクトリを作成する

    Parameters
    -----------
    dir_path : list or int
        作成するディレクトリのpath
    """
    if isinstance(dir_path, str):
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
    else:
        for i in dir_path:
            if not os.path.exists(i):
                os.makedirs(i)


def convert_to_mp3(input_file: str) -> str:
    """
    入力ファイルを.mp3に変換する

    Parameters
    ----------
    input_file : str
        入力ファイルのpath

    Returns
    ----------
    converted_file : str
        変換後のファイルのpath 
    """

    file_parts = os.path.splitext(input_file)
    file_extension = file_parts[-1].lower()
    file_name = file_parts[0].split("/")[-1]

    # 動画ファイルの場合、音声を抽出
    if file_extension in [".mp4", ".mov", ".avi", ".wmv", ".mpg", ".mkv", ".flv", ".asf", ".wmv"]:
        video = VideoFileClip(input_file)
        input_file = "./temp/" + file_name + ".wav"
        video.audio.write_audiofile(input_file)

        file_parts = os.path.splitext(input_file)
        file_extension = file_parts[-1].lower()
    
    # mp3以外のファイルはmp3に変換
    if file_extension != ".mp3":
        raw_audio = AudioSegment.from_file(input_file, format = file_parts[-1])
        converted_file = "./temp/" + file_name + ".mp3"
        raw_audio.export(converted_file, format="mp3")
    else:
        converted_file = input_file

    return converted_file

    
# 音声を適切な長さに分割
def split_audio(input_audio: AudioSegment, max_length: int, config: dict) -> list:
    """
    音声ファイルを最大長以下に分割する

    Parameters
    ----------
    input_audio : AudioSegment
        pydubで読み込んだ音声データ
    max_length : int
        動画を分割する際の最大長 [s]
    config : dict
        設定ファイル。split_on_silenceのパラメータを設定するために使用。

    Returns
    ----------
    file_path_list : list
        分割した音声ファイルのpathを要素とするlist
    """


    # 無音部分をカットして分割
    chunks = split_on_silence(
        input_audio,
        min_silence_len = config["min_silence_len"],
        silence_thresh = config["silence_thresh"],
        keep_silence = config["keep_silence"],
    )

    # 音声データを保存
    def output_mp3(chunk, output):
        chunk.export(output, format="mp3")
    
    # 分割したチャンクをmax_lengthごとに結合
    current_chunk = chunks[0]
    file_path_list = []
    file_name = os.path.splitext(config["input_audio_path"])[0].split("/")[-1]

    for i, c in enumerate(chunks[1:], 1):  # 1からスタート
        temp_chunk = current_chunk + c
        outFilePath = f'./temp/{file_name}_split_{i + 1}.mp3'
        if len(temp_chunk) > max_length:
            output_mp3(current_chunk, outFilePath)
            file_path_list.append(outFilePath)
            current_chunk = c
        else:
            current_chunk += c

        # 最後のチャンクの処理
        if i == len(chunks) - 1:
            output_mp3(current_chunk, outFilePath)
            file_path_list.append(outFilePath)
    
    return file_path_list 


def audio_to_text(audio_path: str, model="whisper-1", language="ja", response_format = "json") -> Any:
    """
    whisper apiを用いて音声データをテキストに変換

    Parameters
    ----------
    audio_path : str
        音声ファイルのpath
    model : str
        whisper api で使用するmodelの種類
    language : str
        書き起こし言語
    response_format : str
        出力ファイルの形式(json, text, srt, verbose_json, vtt)

    Returns
    ----------
    transcript : Any
        whisper API による出力
    """

    with open(audio_path, "rb") as f:
        transcript = openai.Audio.transcribe(
            model = model, 
            file = f,
            language = language,
            response_format = response_format,
            prompt = "こんにちは。今日は、いいお天気ですね。")
    
    return transcript


def trans_for_subtitles(input_audio_path: str) -> (Any, list):
    """
    字幕作成用の書き起こしを出力する

    Parameters
    ----------
    input_audio_path : str
        音声ファイルのpath

    Returns
    ----------
    output : Any
        whisper API による出力
    extracted_data : list
        text(文章), start(開始時間[s]), duration(継続時間[s])をキーに持つdictが各要素のlist
    """

    output = audio_to_text(input_audio_path, response_format="verbose_json")
    segments = output["segments"]

    extracted_data = []
    for segment in segments:
        text = segment["text"]
        start = segment["start"]
        duration = segment["end"] - segment["start"]
        extracted_data.append({"text": text, "start": start, "duration": duration})

    return output, extracted_data


def check_file_size(file_path :str) -> (AudioSegment, bool, int):
    """
    入力ファイルがwhisper API 制限の25MBに収まっているか判断する

    Parameters
    ----------
    file_path : str
        入力ファイルのpath
    
    Returns
    ----------
    audio : AudioSegment
        pydubで読み込んだ音声データ
    need_split : bool
        ファイル分割が必要かどうか(Trueであれば必要)
    max_length : int
        分割する際の最大時間 [s]
    """

    audio = AudioSegment.from_file(file_path, format="mp3")
    total_length = len(audio)
    total_size = os.path.getsize(file_path)
    max_file_size = 25000000 * 0.9   # Whisper APIは25MB制限, 10%ほど余裕を残す
    max_length = total_length * (max_file_size / total_size)     # ファイルサイズと時間から、分割する最大時間を取得する

    need_split = total_size > max_file_size # ファイル分割が必要かどうか

    return audio, need_split, max_length



def mk_transcription(config: dict) -> str:
    """
    音声データを書き起こし保存する
    config["mode"] == "subtitles"の場合は時間付きの書き起こし文章を結果ファイルに保存する

    Parameters
    ----------
    config : dict
        設定ファイル
    
    Returns
    ----------
    transcription : str
        書き起こした文章
    """

    input_file = config["input_audio_path"]
    file_parts = os.path.splitext(input_file)
    file_name = file_parts[0].split("/")[-1]

    src_file = convert_to_mp3(input_file)

    audio, need_split, max_length = check_file_size(src_file)

    # 音声分割が必要な場合
    if need_split:
        audio_file_list = split_audio(audio, max_length, config)
        extracted_data_list = []
        duration_list = [] # 分割した音声ファイルの長さを格納
        transcription_list = []

        for i in audio_file_list:
            output, extracted_data = trans_for_subtitles(i)
            duration_list.append(output["duration"])
            extracted_data_list.append(extracted_data)
            transcription_list.append(output["text"])
        
        transcription = "".join(transcription_list) # 書き起こしの文章

        # 各音声ファイルの開始時間を計算
        offsets = [0] + [sum(duration_list[:i+1]) for i in range(len(duration_list)-1)]

        all_extracted_data = [] # 字幕用の書き起こし文章
        for offset, each_data in zip(offsets, extracted_data_list):
            for item in each_data:
                item["start"] += offset
                all_extracted_data.append(item)
            
    # 音声分割が不必要な場合
    else:
        output, all_extracted_data = trans_for_subtitles(src_file)
        transcription = output["text"]
    
    # 書き起こしの文章のみの場合
    if config["mode"] == "minutes":
        with open("./results/" + file_name + "_transcription.txt", "w", encoding="utf-8") as f:
            f.write(transcription)
    
    # 字幕用の書き起こしの場合
    if config["mode"] == "subtitles":
        with open("./results/" + file_name + "_subtitles.json", "w", encoding="utf-8") as f:
            json.dump(all_extracted_data, f, ensure_ascii=False)
 
    return transcription


def split_text(text: str, max_length: int) -> list:
    """
    文章を最大長以下で大体長さが等しくなるように分割する

    Parameters
    ----------
    text : str
        分割対象となるテキスト
    max_length : int
        分割する際のテキストの最大長
    
    Returns
    ----------
    segments : list
        分割した文章を要素とするlist

    """
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
def use_chatGPT(target_text :str, prompt_path: str, model : str) -> str:
    """
    Chat GPT APIを利用する

    Parameters
    ----------
    target_text : str
        Chat GPTに与える情報であるテキスト
    prompt_path : str
        Chat GPT に与えるプロンプトが記述されたテキストファイルのpath
    model : str
        Chat GPT API で使用するmodel ("gpt-3.5-turbo", "gpt-4", "gpt-4-32k")

    Returns
    ----------
    output : str
        Chat GPT APIが返す出力の内容
    """
    with open(prompt_path) as f:
        prompt = f.read()
    response = openai.ChatCompletion.create(
    model = model,
    messages = [
        {"role": "system", "content":"あなたは優秀な社員です。"},
        {"role": "assistant", "content": target_text},
        {"role": "user", "content": prompt}
    ])
    output = response["choices"][0]["message"]["content"]

    return output



# 入力文章から議事録を作成
def mk_minutes(trans: str, config: dict):
    """
    書き起こした文章から議事録を作成・保存する
    制限トークンを超える場合は、文章の分割->各文章の要約->各要約文から議事録を作成 というSTEPを踏む

    Parameters
    ----------
    trans : str
        書き起こしの文章
    config : dict
        設定ファイル(使用するChat GPT のモデルやプロンプトの情報を利用する)
    """

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
            with open("./temp/" + file_name + "_split_gijiroku.txt", "w") as f:
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
        with open("./temp/" + file_name + "_split_gijiroku.txt", "w") as f:
            f.write(gijiroku_all)
    
    # 最終的な議事録を保存
    with open("./results/" + file_name + "_gijiroku.txt", "w") as f:
        f.write(gijiroku)
    

def main():
    # 設定ファイルの読み込み
    with open("./config_debug.json", "r") as f:
        config = json.load(f)
    
    # API Keyの設定
    openai.api_key = config["openai_api_token"]
    
    # ディレクトリ設定
    mk_dir(["./results/", "./temp/"])
    
    # 書き起こしの作成
    transcription = mk_transcription(config)

    if config["mode"] == "minutes":
        # 議事録の作成・保存
        mk_minutes(transcription, config) 

    # tempディレクトリの削除
    shutil.rmtree( "./temp/")


if __name__ == '__main__':
    main()
