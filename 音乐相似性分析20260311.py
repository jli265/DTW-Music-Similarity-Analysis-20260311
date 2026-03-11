# https://gemini.google.com/app/5b9640d3b9fd0239
import librosa
import numpy as np
import os
import warnings
import pandas as pd
from sklearn.preprocessing import minmax_scale

# 忽略警告
warnings.filterwarnings('ignore')

def list_all_files(rootdir,is_print=False):
    _files = []
    # 列出文件夹下所有的目录与文件
    list_file = os.listdir(rootdir)
    for i in range(0, len(list_file)):
        # 构造路径
        if is_print:
            print('rootdir: ', rootdir)
            print('list_file[i]: ', list_file[i])
        # path = os.path.join(rootdir,list_file[i])
        path = rootdir + '/' + list_file[i] if rootdir != './' else rootdir + list_file[i]
        if is_print:
           print('path: ', path)
        # 判断路径是否是一个文件目录或者文件
        # 如果是文件目录，继续递归
        if os.path.isdir(path):
            _files.extend(list_all_files(path))
        if os.path.isfile(path):
            _files.append(path)
    return _files

def extract_features_cached(file_list, duration=30):
    """
    预提取特征并缓存
    """
    feature_cache = {}
    print(f"--- 1. 提取特征 (时长: {duration}s) ---")

    for file_path in file_list:
        if not os.path.exists(file_path):
            print(f"⚠️ 跳过不存在的文件: {file_path}")
            continue
        try:
            y, sr = librosa.load(file_path, sr=22050, duration=duration)
            # 使用 CQT 提取色度特征
            chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=8192)
            # 归一化
            chroma = minmax_scale(chroma, axis=1)
            feature_cache[file_path] = chroma
            print(f"✅ 处理完成: {os.path.basename(file_path)}")
        except Exception as e:
            print(f"❌ 错误 {file_path}: {e}")

    return feature_cache

def run_dtw_analysis(feature_cache):
    """
    执行 N x N DTW 计算
    """
    files = list(feature_cache.keys())
    n = len(files)
    matrix = np.eye(n)

    print(f"\n--- 2. 计算 {n}x{n} DTW 相似度 ---")

    for i in range(n):
        for j in range(i + 1, n):
            f1, f2 = files[i], files[j]
            seq1, seq2 = feature_cache[f1], feature_cache[f2]

            # 计算 DTW 距离
            D, wp = librosa.sequence.dtw(X=seq1, Y=seq2, metric='cosine')
            avg_cost = D[-1, -1] / len(wp)
            score = max(0.0, 1.0 - avg_cost)

            matrix[i][j] = score
            matrix[j][i] = score

    return matrix, files

def save_matrix_to_excel(matrix, file_paths, output_name="music_similarity.xlsx"):
    """
    使用 Pandas 将矩阵保存为 Excel
    """
    # 提取文件名作为行列标签
    file_names = [os.path.basename(f) for f in file_paths]

    # 创建 DataFrame
    df = pd.DataFrame(matrix, index=file_names, columns=file_names)

    # 转换为百分比格式（可选，方便在 Excel 中直接查看）
    df_styled = df.applymap(lambda x: f"{x * 100:.2f}%")

    try:
        df_styled.to_excel(output_name)
        print(f"\n💾 成功！结果已保存至: {os.path.abspath(output_name)}")
    except Exception as e:
        print(f"❌ 保存 Excel 失败: {e}")

# ==========================================
# 主程序
# ==========================================
if __name__ == "__main__":
    # 1. 填入你的 MP3 文件路径列表
    my_songs = list_all_files('./音乐文件夹')
    print('MP3分析数量:',len(my_songs))

    # 2. 提取特征
    cache = extract_features_cached(my_songs, duration=30)

    if len(cache) > 1:
        # 3. 计算 DTW
        sim_matrix, valid_files = run_dtw_analysis(cache)

        # 4. 保存到本地 Excel
        save_matrix_to_excel(sim_matrix, valid_files, "./结果/歌曲相似度分析结果.xlsx")
    else:
        print("❌ 有效文件不足，无法分析。")