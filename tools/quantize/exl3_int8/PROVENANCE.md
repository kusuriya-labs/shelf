# 由来

`jyohukuchan/rocm_exl3_forme` の commit `b922ccbd558237b13cc2e3bb25b87bc7cb2cbfe3` から変換器、実行に必要な Python パッケージ、ROCm 拡張のソースおよび calibration データを取り込んだ。元のライセンスは [EXLLAMAV3-LICENSE.txt](EXLLAMAV3-LICENSE.txt) に保存する。

example、test、評価、文書、画像および ROCm のビルド対象外の量子化 kernel 生成単位は含めない。共通ヘッダーと Python の import が必要とするモジュールは残す。変換の入口で拡張の multiplier を検証し、mul1 codebook 以外を拒否する。calibration の乱数 seed は環境変数から指定できる。ビルドは gfx1201 と mul24 に固定する。
