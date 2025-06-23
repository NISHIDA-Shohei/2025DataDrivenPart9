import pandas as pd
import numpy as np
import os
from itertools import combinations
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.font_manager as fm
from matplotlib.colors import ListedColormap, BoundaryNorm
import networkx as nx

# 日本語フォントの設定
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Hiragino Sans', 'Yu Gothic', 'Meiryo', 'Takao', 'IPAexGothic', 'IPAPGothic', 'VL PGothic', 'Noto Sans CJK JP']
plt.rcParams['axes.unicode_minus'] = False

def preprocess_data(file_path):
    """CSVを読み込み、前処理を行う"""
    if not os.path.exists(file_path):
        return None
    df = pd.read_csv(file_path)
    df['datetime'] = pd.to_datetime(df['date'].astype(str) + ' ' + df['time'].astype(str), format='%Y%m%d %H:%M:%S', errors='coerce')
    df.dropna(subset=['datetime'], inplace=True)
    return df

def calculate_d_metric(df_a, df_b, window_minutes):
    """2つのデータフレームからd(a,b)を計算する"""
    # 同じdatetimeでデータをマージ
    merged_df = pd.merge(df_a, df_b, on='datetime', suffixes=('_a', '_b'))

    if merged_df.empty:
        return pd.DataFrame()

    # 計算に必要なカラムを作成
    # (a - b)^2
    merged_df['diff_sq'] = (merged_df['steps_a'] - merged_df['steps_b'])**2
    # a^2 + b^2
    merged_df['sum_sq'] = merged_df['steps_a']**2 + merged_df['steps_b']**2
    
    # 探索窓（rolling window）で合計を計算
    rolling_diff_sq = merged_df['diff_sq'].rolling(window=window_minutes, min_periods=window_minutes).sum()
    rolling_sum_sq = merged_df['sum_sq'].rolling(window=window_minutes, min_periods=window_minutes).sum()
    
    # d(a,b) を計算
    # 分母が0の場合を避ける
    d_metric = rolling_diff_sq / rolling_sum_sq.replace(0, np.nan)
    
    merged_df['d_metric'] = d_metric
    merged_df['denominator'] = rolling_sum_sq
    
    return merged_df[['datetime', 'd_metric', 'denominator']].dropna()

def calculate_relationship_score(df, denominator_threshold, d_metric_threshold, continuous_minutes):
    """
    人間関係スコアを計算する
    - 分母が denominator_threshold 以上
    - d(a,b) が d_metric_threshold 以下
    - これが continuous_minutes 以上連続した期間の合計時間（分）をスコアとする
    """
    if df.empty:
        return 0

    df_sorted = df.sort_values('datetime').reset_index(drop=True)
    
    # 条件を満たす行を特定
    condition = (df_sorted['denominator'] >= denominator_threshold) & (df_sorted['d_metric'] <= d_metric_threshold)
    df_sorted['condition_met'] = condition
    
    # 連続ブロックを識別
    df_sorted['block'] = (df_sorted['condition_met'] != df_sorted['condition_met'].shift()).cumsum()
    
    # 条件を満たすブロックだけを抽出
    score_blocks = df_sorted[df_sorted['condition_met']]
    
    if score_blocks.empty:
        return 0
        
    # 各ブロックのサイズ（連続時間）を計算
    block_sizes = score_blocks.groupby('block').size()
    
    # 連続時間が continuous_minutes 以上のブロックの合計時間（分）をスコアとする
    score = block_sizes[block_sizes >= continuous_minutes].sum()
    
    return score

def generate_matrix_for_day_type(all_user_data, day_type):
    """指定された曜日の種類（平日/休日）でデータをフィルタリングし、マトリクスを生成する"""
    
    day_type_japanese = "平日" if day_type == "weekday" else "休日"
    print(f"\n===== Processing All Groups ({day_type_japanese}) =====")

    # 曜日に基づいてデータをフィルタリング
    filtered_data = {}
    for user, df in all_user_data.items():
        if day_type == 'weekday':
            # 月曜日=0, 日曜日=6
            filtered_df = df[df['datetime'].dt.dayofweek < 5].copy()
        else: # weekend
            filtered_df = df[df['datetime'].dt.dayofweek >= 5].copy()
        
        if not filtered_df.empty:
            filtered_data[user] = filtered_df

    if len(filtered_data) < 2:
        print(f"十分なデータがないため、{day_type_japanese}の処理をスキップします。")
        return
        
    # パラメータ設定
    w = 60
    denominator_threshold = 5500
    d_metric_threshold = 0.05
    continuous_minutes = 15
    
    scores = {}
    print(f"Calculating scores for {day_type_japanese}... (This may take a while)")
    
    # ユーザーペアでループ
    for user_a, user_b in combinations(sorted(filtered_data.keys()), 2):
        df_a = filtered_data[user_a]
        df_b = filtered_data[user_b]
        
        # d(a,b)を計算
        d_metric_df = calculate_d_metric(df_a, df_b, window_minutes=w)
        
        # スコアを計算
        score = calculate_relationship_score(
            d_metric_df,
            denominator_threshold,
            d_metric_threshold,
            continuous_minutes
        )
        if score > 0:
            scores[(user_a, user_b)] = score
        
    # 結果をマトリクスで準備
    user_list = sorted(all_user_data.keys())
    score_matrix = pd.DataFrame(0, index=user_list, columns=user_list)

    for (user_a, user_b), score in scores.items():
        score_matrix.loc[user_a, user_b] = score
        score_matrix.loc[user_b, user_a] = score
    
    score_matrix = score_matrix.astype(int)

    print(f"\n{day_type_japanese}の人間関係スコア（マトリクス）:")
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        print(score_matrix)
    
    # ヒートマップを生成して保存
    plt.figure(figsize=(22, 20))
    
    # 3色の離散的なカラーマップを定義
    colors = ['#f0f0f0', '#a2d5f2', '#0077b6'] # 0, 1-100, 101+
    cmap = ListedColormap(colors)
    
    # スコアの最大値に基づいて色の境界と正規化を決定
    max_val = max(102, score_matrix.max().max()) # 最小でも102を確保
    boundaries = [0, 1, 101, max_val]
    norm = BoundaryNorm(boundaries, cmap.N)

    sns.heatmap(score_matrix, annot=True, fmt="d", cmap=cmap, norm=norm, 
                linewidths=.1, annot_kws={"size": 5})

    plt.title(f'全グループにおける人間関係スコア ({day_type_japanese})', fontsize=20)
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    filename = f"score_matrix_all_{day_type}.png"
    plt.savefig(filename, dpi=300)
    print(f"\nスコア行列のヒートマップを {filename} として保存しました。")


def generate_matrix_for_time_period(all_user_data, time_period):
    """指定された時間帯（夜勤/通常）でデータをフィルタリングし、マトリクスを生成する"""
    
    time_period_japanese = "夜勤時間帯" if time_period == "night_shift" else "通常時間帯"
    print(f"\n===== Processing All Groups ({time_period_japanese}) =====")

    # 時間帯に基づいてデータをフィルタリング
    filtered_data = {}
    for user, df in all_user_data.items():
        if time_period == 'night_shift':
            # 夜勤時間帯: 22時〜翌5時
            filtered_df = df[
                (df['datetime'].dt.hour >= 22) | (df['datetime'].dt.hour <= 5)
            ].copy()
        else: # regular_hours
            # 通常時間帯: 6時〜21時
            filtered_df = df[
                (df['datetime'].dt.hour >= 6) & (df['datetime'].dt.hour <= 21)
            ].copy()
        
        if not filtered_df.empty:
            filtered_data[user] = filtered_df

    if len(filtered_data) < 2:
        print(f"十分なデータがないため、{time_period_japanese}の処理をスキップします。")
        return
        
    # パラメータ設定
    w = 60
    denominator_threshold = 5500
    d_metric_threshold = 0.05
    continuous_minutes = 15
    
    scores = {}
    print(f"Calculating scores for {time_period_japanese}... (This may take a while)")
    
    # ユーザーペアでループ
    for user_a, user_b in combinations(sorted(filtered_data.keys()), 2):
        df_a = filtered_data[user_a]
        df_b = filtered_data[user_b]
        
        # d(a,b)を計算
        d_metric_df = calculate_d_metric(df_a, df_b, window_minutes=w)
        
        # スコアを計算
        score = calculate_relationship_score(
            d_metric_df,
            denominator_threshold,
            d_metric_threshold,
            continuous_minutes
        )
        if score > 0:
            scores[(user_a, user_b)] = score
        
    # 結果をマトリクスで準備
    user_list = sorted(all_user_data.keys())
    score_matrix = pd.DataFrame(0, index=user_list, columns=user_list)

    for (user_a, user_b), score in scores.items():
        score_matrix.loc[user_a, user_b] = score
        score_matrix.loc[user_b, user_a] = score
    
    score_matrix = score_matrix.astype(int)

    print(f"\n{time_period_japanese}の人間関係スコア（マトリクス）:")
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        print(score_matrix)
    
    # ヒートマップを生成して保存
    plt.figure(figsize=(22, 20))
    
    # 3色の離散的なカラーマップを定義
    colors = ['#f0f0f0', '#a2d5f2', '#0077b6'] # 0, 1-100, 101+
    cmap = ListedColormap(colors)
    
    # スコアの最大値に基づいて色の境界と正規化を決定
    max_val = max(102, score_matrix.max().max()) # 最小でも102を確保
    boundaries = [0, 1, 101, max_val]
    norm = BoundaryNorm(boundaries, cmap.N)

    sns.heatmap(score_matrix, annot=True, fmt="d", cmap=cmap, norm=norm, 
                linewidths=.1, annot_kws={"size": 5})

    plt.title(f'全グループにおける人間関係スコア ({time_period_japanese})', fontsize=20)
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    filename = f"score_matrix_all_{time_period}.png"
    plt.savefig(filename, dpi=300)
    print(f"\nスコア行列のヒートマップを {filename} として保存しました。")


def create_network_graph(all_user_data, title="人間関係ネットワーク"):
    """人間関係スコアをネットワークグラフで可視化する"""
    
    # パラメータ設定
    w = 60
    denominator_threshold = 5500
    d_metric_threshold = 0.05
    continuous_minutes = 15
    
    print(f"\n===== {title}の計算中 =====")
    
    scores = {}
    print("Calculating relationship scores... (This may take a while)")
    
    # ユーザーペアでループ
    for user_a, user_b in combinations(sorted(all_user_data.keys()), 2):
        df_a = all_user_data[user_a]
        df_b = all_user_data[user_b]
        
        # d(a,b)を計算
        d_metric_df = calculate_d_metric(df_a, df_b, window_minutes=w)
        
        # スコアを計算
        score = calculate_relationship_score(
            d_metric_df,
            denominator_threshold,
            d_metric_threshold,
            continuous_minutes
        )
        if score > 0:
            scores[(user_a, user_b)] = score
    
    if not scores:
        print("関係性スコアが見つかりませんでした。")
        return
    
    # NetworkXグラフを作成
    G = nx.Graph()
    
    # ノードを追加（全ユーザー）
    for user in sorted(all_user_data.keys()):
        G.add_node(user)
    
    # エッジを追加（スコアが0より大きい関係のみ）
    for (user_a, user_b), score in scores.items():
        G.add_edge(user_a, user_b, weight=score)
    
    # グループ別の色を定義
    group_colors = {
        'a': '#ff6b6b',  # 赤系
        'b': '#4ecdc4',  # 青緑系
        'c': '#45b7d1'   # 青系
    }
    
    # ノードの色を設定
    node_colors = []
    for node in G.nodes():
        group = node[0]  # ユーザー名の最初の文字（a, b, c）
        node_colors.append(group_colors[group])
    
    # エッジの太さをスコアに基づいて設定
    edge_weights = [G[u][v]['weight'] for u, v in G.edges()]
    max_weight = max(edge_weights) if edge_weights else 1
    edge_widths = [w / max_weight * 5 + 0.5 for w in edge_weights]  # 0.5〜5.5の範囲
    
    # グラフの描画
    plt.figure(figsize=(20, 16))
    
    # カスタムレイアウトでグループ別に配置
    pos = {}
    
    # グループ別の配置位置を定義
    group_positions = {
        'a': (-2, 1),    # グループA: 左上
        'b': (0, -1),    # グループB: 中央下
        'c': (2, 1)      # グループC: 右上
    }
    
    # 各グループ内でのノード配置
    for group in ['a', 'b', 'c']:
        base_x, base_y = group_positions[group]
        group_nodes = [node for node in G.nodes() if node.startswith(group)]
        group_nodes.sort()
        
        # グループ内で円形に配置
        n_nodes = len(group_nodes)
        for i, node in enumerate(group_nodes):
            angle = 2 * np.pi * i / n_nodes
            radius = 0.8
            x = base_x + radius * np.cos(angle)
            y = base_y + radius * np.sin(angle)
            pos[node] = (x, y)
    
    # ノードを描画
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=800, alpha=0.8)
    
    # エッジを描画
    nx.draw_networkx_edges(G, pos, width=edge_widths, alpha=0.6, edge_color='gray')
    
    # ノードラベルを描画
    nx.draw_networkx_labels(G, pos, font_size=10, font_weight='bold')
    
    # エッジラベル（スコア）を描画
    edge_labels = {(u, v): f'{G[u][v]["weight"]}' for u, v in G.edges()}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8)
    
    # 凡例を作成
    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=group_colors['a'], 
                  markersize=15, label='グループA'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=group_colors['b'], 
                  markersize=15, label='グループB'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=group_colors['c'], 
                  markersize=15, label='グループC')
    ]
    
    plt.legend(handles=legend_elements, loc='upper left', fontsize=12)
    
    plt.title(title, fontsize=20, fontweight='bold')
    plt.axis('off')
    plt.tight_layout()
    
    # ファイル名を生成
    filename = f"network_graph_{title.replace(' ', '_').replace('（', '').replace('）', '')}.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"\nネットワークグラフを {filename} として保存しました。")


def create_network_graph_with_steps(all_user_data, office_hours_stats, title="人間関係ネットワーク（通常時間帯・歩数付き）"):
    """通常時間帯の人間関係スコアをネットワークグラフで可視化し、ノードに歩数情報を表示する"""
    
    # パラメータ設定
    w = 60
    denominator_threshold = 5500
    d_metric_threshold = 0.05
    continuous_minutes = 15
    
    print(f"\n===== {title}の計算中 =====")
    
    # 通常時間帯（9時〜21時）のデータをフィルタリング
    regular_hours_data = {}
    for user, df in all_user_data.items():
        regular_df = df[
            (df['datetime'].dt.hour >= 9) & (df['datetime'].dt.hour <= 21)
        ].copy()
        if not regular_df.empty:
            regular_hours_data[user] = regular_df
    
    if len(regular_hours_data) < 2:
        print("十分なデータがないため、処理をスキップします。")
        return
    
    scores = {}
    print("Calculating relationship scores... (This may take a while)")
    
    # ユーザーペアでループ
    for user_a, user_b in combinations(sorted(regular_hours_data.keys()), 2):
        df_a = regular_hours_data[user_a]
        df_b = regular_hours_data[user_b]
        
        # d(a,b)を計算
        d_metric_df = calculate_d_metric(df_a, df_b, window_minutes=w)
        
        # スコアを計算
        score = calculate_relationship_score(
            d_metric_df,
            denominator_threshold,
            d_metric_threshold,
            continuous_minutes
        )
        if score > 0:
            scores[(user_a, user_b)] = score
    
    if not scores:
        print("関係性スコアが見つかりませんでした。")
        return
    
    # NetworkXグラフを作成
    G = nx.Graph()
    
    # ノードを追加（全ユーザー）
    for user in sorted(all_user_data.keys()):
        G.add_node(user)
    
    # エッジを追加（スコアが0より大きい関係のみ）
    for (user_a, user_b), score in scores.items():
        G.add_edge(user_a, user_b, weight=score)
    
    # グループ別の色を定義
    group_colors = {
        'a': '#ff6b6b',  # 赤系
        'b': '#4ecdc4',  # 青緑系
        'c': '#45b7d1'   # 青系
    }
    
    # ノードの色を設定
    node_colors = []
    for node in G.nodes():
        group = node[0]  # ユーザー名の最初の文字（a, b, c）
        node_colors.append(group_colors[group])
    
    # エッジの太さをスコアに基づいて設定
    edge_weights = [G[u][v]['weight'] for u, v in G.edges()]
    max_weight = max(edge_weights) if edge_weights else 1
    edge_widths = [w / max_weight * 5 + 0.5 for w in edge_weights]  # 0.5〜5.5の範囲
    
    # カスタムレイアウトでグループ別に配置
    pos = {}
    
    # グループ別の配置位置を定義
    group_positions = {
        'a': (-2, 1),    # グループA: 左上
        'b': (0, -1),    # グループB: 中央下
        'c': (2, 1)      # グループC: 右上
    }
    
    # 各グループ内でのノード配置
    for group in ['a', 'b', 'c']:
        base_x, base_y = group_positions[group]
        group_nodes = [node for node in G.nodes() if node.startswith(group)]
        group_nodes.sort()
        
        # グループ内で円形に配置
        n_nodes = len(group_nodes)
        for i, node in enumerate(group_nodes):
            angle = 2 * np.pi * i / n_nodes
            radius = 0.8
            x = base_x + radius * np.cos(angle)
            y = base_y + radius * np.sin(angle)
            pos[node] = (x, y)
    
    # グラフの描画
    plt.figure(figsize=(22, 18))
    
    # ノードを描画
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=1000, alpha=0.8)
    
    # エッジを描画
    nx.draw_networkx_edges(G, pos, width=edge_widths, alpha=0.6, edge_color='gray')
    
    # ノードラベルを描画（ユーザー名 + 歩数）
    node_labels = {}
    for node in G.nodes():
        avg_steps = office_hours_stats[node]['avg_daily_steps']
        node_labels[node] = f"{node}\n{avg_steps:.0f}歩"
    
    nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=8, font_weight='bold')
    
    # エッジラベル（スコア）を描画
    edge_labels = {(u, v): f'{G[u][v]["weight"]}' for u, v in G.edges()}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8)
    
    # 凡例を作成
    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=group_colors['a'], 
                  markersize=15, label='グループA'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=group_colors['b'], 
                  markersize=15, label='グループB'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=group_colors['c'], 
                  markersize=15, label='グループC')
    ]
    
    plt.legend(handles=legend_elements, loc='upper left', fontsize=12)
    
    plt.title(title, fontsize=20, fontweight='bold')
    plt.axis('off')
    plt.tight_layout()
    
    # ファイル名を生成
    filename = f"network_graph_regular_hours_with_steps.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"\n歩数付きネットワークグラフを {filename} として保存しました。")


def main():
    """全グループのデータを読み込み、オフィスアワー（平日6時〜21時）の1日平均歩数を計算して表示する"""
    group_prefixes = ['a', 'b', 'c']
    user_files = {}
    for prefix in group_prefixes:
        for i in range(1, 11):
            user = f"{prefix}{i:02d}"
            user_files[user] = f"{user}.csv"

    # データの読み込みと前処理
    all_data = {}
    print("Loading all user data...")
    for user, file in user_files.items():
        df = preprocess_data(file)
        if df is not None:
            all_data[user] = df
    
    if len(all_data) == 0:
        print("データファイルが1つも見つかりませんでした。処理を終了します。")
        return

    print("\n===== オフィスアワー（平日9時〜21時）の1日平均歩数 =====")
    
    # オフィスアワーのデータをフィルタリングして1日平均歩数を計算
    office_hours_stats = {}
    
    for user, df in all_data.items():
        # 平日かつ9時〜21時のデータをフィルタリング
        office_hours_df = df[
            (df['datetime'].dt.dayofweek < 5) &  # 平日（月〜金）
            (df['datetime'].dt.hour >= 9) & (df['datetime'].dt.hour <= 21)  # 9時〜21時
        ].copy()
        
        if not office_hours_df.empty:
            # 日付ごとにグループ化して1日の総歩数を計算
            daily_steps = office_hours_df.groupby(office_hours_df['datetime'].dt.date)['steps'].sum()
            
            # 1日の平均歩数を計算
            avg_daily_steps = daily_steps.mean()
            total_days = len(daily_steps)
            
            office_hours_stats[user] = {
                'avg_daily_steps': avg_daily_steps,
                'total_days': total_days,
                'min_daily_steps': daily_steps.min(),
                'max_daily_steps': daily_steps.max()
            }
        else:
            office_hours_stats[user] = {
                'avg_daily_steps': 0,
                'total_days': 0,
                'min_daily_steps': 0,
                'max_daily_steps': 0
            }
    
    # 結果を表示
    print("\nユーザー別オフィスアワー1日平均歩数:")
    print("ユーザー | 平均歩数/日 | データ日数 | 最小歩数/日 | 最大歩数/日")
    print("-" * 60)
    
    for user in sorted(office_hours_stats.keys()):
        stats = office_hours_stats[user]
        print(f"{user:6} | {stats['avg_daily_steps']:10.1f} | {stats['total_days']:8} | {stats['min_daily_steps']:10.0f} | {stats['max_daily_steps']:10.0f}")

    # グループ別の統計を計算
    group_stats = {'A': [], 'B': [], 'C': []}
    for user, stats in office_hours_stats.items():
        group = user[0].upper()
        group_stats[group].append(stats['avg_daily_steps'])
    
    print("\n===== グループ別オフィスアワー平均歩数 =====")
    print("グループ | 平均歩数/日 | メンバー数 | 最小歩数/日 | 最大歩数/日 | 標準偏差")
    print("-" * 70)
    
    for group in ['A', 'B', 'C']:
        if group_stats[group]:
            avg_steps = np.mean(group_stats[group])
            min_steps = np.min(group_stats[group])
            max_steps = np.max(group_stats[group])
            std_steps = np.std(group_stats[group])
            member_count = len(group_stats[group])
            
            print(f"{group:7} | {avg_steps:10.1f} | {member_count:8} | {min_steps:10.0f} | {max_steps:10.0f} | {std_steps:8.1f}")
    
    # グループ間の比較
    print("\n===== グループ間比較 =====")
    group_avgs = {group: np.mean(steps) for group, steps in group_stats.items() if steps}
    
    if len(group_avgs) > 1:
        # 最も活動的なグループ
        most_active = max(group_avgs.items(), key=lambda x: x[1])
        least_active = min(group_avgs.items(), key=lambda x: x[1])
        
        print(f"最も活動的なグループ: {most_active[0]} ({most_active[1]:.1f}歩/日)")
        print(f"最も非活動的なグループ: {least_active[0]} ({least_active[1]:.1f}歩/日)")
        print(f"グループ間の差: {most_active[1] - least_active[1]:.1f}歩/日")
        
        # グループ内のばらつき
        group_variations = {group: np.std(steps) for group, steps in group_stats.items() if steps}
        most_varied = max(group_variations.items(), key=lambda x: x[1])
        least_varied = min(group_variations.items(), key=lambda x: x[1])
        
        print(f"最もばらつきの大きいグループ: {most_varied[0]} (標準偏差: {most_varied[1]:.1f})")
        print(f"最もばらつきの小さいグループ: {least_varied[0]} (標準偏差: {least_varied[1]:.1f})")

    print("\n全ての処理が完了しました。")

    # ネットワークグラフの生成（全データ）
    create_network_graph(all_data)
    
    # 時間帯別ネットワークグラフの生成
    print("\n===== 時間帯別ネットワークグラフの生成 =====")
    
    # 通常時間帯（9時〜21時）のネットワークグラフ
    regular_hours_data = {}
    for user, df in all_data.items():
        regular_df = df[
            (df['datetime'].dt.hour >= 9) & (df['datetime'].dt.hour <= 21)
        ].copy()
        if not regular_df.empty:
            regular_hours_data[user] = regular_df
    
    if len(regular_hours_data) >= 2:
        create_network_graph(regular_hours_data, "人間関係ネットワーク（通常時間帯）")
    
    # 夜勤時間帯（22時〜翌5時）のネットワークグラフ
    night_shift_data = {}
    for user, df in all_data.items():
        night_df = df[
            (df['datetime'].dt.hour >= 22) | (df['datetime'].dt.hour <= 5)
        ].copy()
        if not night_df.empty:
            night_shift_data[user] = night_df
    
    if len(night_shift_data) >= 2:
        create_network_graph(night_shift_data, "人間関係ネットワーク（夜勤時間帯）")
    
    # 休日（土日）のネットワークグラフ
    weekend_data = {}
    for user, df in all_data.items():
        weekend_df = df[df['datetime'].dt.dayofweek >= 5].copy()  # 土日
        if not weekend_df.empty:
            weekend_data[user] = weekend_df
    
    if len(weekend_data) >= 2:
        create_network_graph(weekend_data, "人間関係ネットワーク（休日）")

    # 通常時間帯の人間関係スコアをネットワークグラフで可視化し、ノードに歩数情報を表示する
    create_network_graph_with_steps(all_data, office_hours_stats)

if __name__ == "__main__":
    main() 
