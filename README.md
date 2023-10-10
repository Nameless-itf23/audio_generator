# 音声生成AI
自作の音声生成AIです。
Transformerを用いて、既存の音楽を無制限に延長します。
## 概要
メルスペクトログラムの一定区間を1単語とみなすLLMによって次の区間の推定を行い、自身を入力として再帰的にこれを行うことで実現しています。
## 参考
https://github.com/ghmagazine/vit_book/blob/main/ch2/vit.py (MIT License)