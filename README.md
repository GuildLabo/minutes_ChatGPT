# 概要
ChatGPT APIとWhisper APIを活用し、音声データから書き起こしや議事録を作成する

# ディレクトリ / ファイルの概要
- mk_gijiroku.py：会議の音声データから議事録や書き起こしを作成するpyファイル
- requirements.txt：必要なライブラリ
- config.json：設定ファイル
- prompt：chatGPTに入力するプロンプトのtextファイルが格納されているディレクトリ
    - gijiroku.txt：入力された文章から議事録を作成するプロンプトを書き込むためのファイル（入力文章がchatGPTのトークン制限に収まる場合に使うプロンプト）
    - split_summary.txt：入力文章がchatGPTの制限を超える場合、※の処理を行う。その処理のうち②部分を担うプロンプトを書き込むためのファイル
    - compile.txt：入力文章がchatGPTの制限を超える場合、※処理を行う。その処理のうち③部分を行うためのプロンプトを書き込むためのファイル
    - ※入力文章がChatGPTのトークン制限を超える場合の処理
        - ①文章を分割→②それぞれの文章を要約→③要約された文章ををまとめ、議事録を作成する」
- results：結果が格納されるファイル(自動生成される)
    - "元データのファイル名"_transcription.txt：会議の書き起こしデータ
    - "元データのファイル名"_gijiroku.txt：会議の議事録

# 使い方
1. requirements.textからライブラリをインストールする
2. FFmpegをインストールする
3. config.jsonで、Open API のtokenを設定する
4. config.jsonのinput_audio_pathに入力となる音声データのパスを入れる
5. config.jsonのmodelを選択する ("gpt-3.5-turbo". "gpt-4", "gpt-4-32k")
6. config.jsonのmodeを選択する ("minutes", "subtitles")
7. prpmptの中のgijiroku.txtとcompile.txtでchatGPTに入力するプロンプトを決定する（とりあえず変更せずに回してみて、出力がおかしな場合に直すとかでいいと思います）
8. mk_gijiroku.pyを動かす


# config.jsonのキーの説明
- openai_api_token
    - OpenAI APIのトークン

- **input_audio_path**
    - 入力ファイルのパス (適宜変更)
    - 音声データでも動画データでもどちらでもOK
   
- 以下の3つは、音声データの容量が大きすぎる際に音声データを分割する処理を行うが、その処理のためのパラメータ（適宜変更）
    - min_silence_len
        - 無音部分と判断する最小の長さ (ミリ秒単位)。デフォルトは 1000。

    - keep_silence
        - 無音部分として取り出した音声データの前後に、どの程度の無音を残すか (ミリ秒単位)。デフォルトは 100。

    - silence_thresh
        - 無音部分と判断する閾値。振幅がこの値以下の音を無音部分とみなす。デフォルトは -16 dBFS。

- model
    - chatGPTで使用するモデル（以下の３種類から選択）
        - gpt-3.5-turbo
        - gpt-4
        - gpt-4-32k

- gpt-4-32k
    - gpt-4-32kの使用許可の設定（”True" or "False"）
        - Trueの場合：書き起こしの文章がgpt-4のトークン数(約8000)を超えた際に自動でgpt-4-32kを使用する
        - Falseの場合：書き起こしの文章がgpt-4のトークン数(約8000)を超えた場合でもgpt-4を使用し、文章を分割することで議事録を作成する

- mode
    - 出力として「議事録」か「字幕用の書き起こし」のどちらかを選択する ("minutes" or "subtitles")
    - minutesの場合の出力
        - 書き起こしの文章が格納された.txtファイル
        - 議事録が格納された .txtファイル
    - subtitlesの場合の出力
        - "text:文章", "start:開始時間[s]", "duration:持続時間[s]"をキーとするdictを各要素とするlistが格納された.jsonファイル
