import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity
import warnings
warnings.filterwarnings('ignore')

# 用户评分数据，包含用户ID、电影ID、评分和时间戳
reting_df = None

# 电影基本信息，包含电影ID、标题和类型
movie_df = None

# 用户标签数据，包含用户对电影的标签信息
tags_df = None

# 电影外部链接数据，包含电影在IMDb和TMDb上的ID
links_df = None

# 用户-电影评分矩阵，行为用户，列为电影，值为评分
user_movie_matrix = None

# 加载所有数据文件
def loadData():
    global reting_df, movie_df, tags_df, links_df, user_movie_matrix
    
    reting_df = pd.read_csv('data/ratings.csv')
    movie_df = pd.read_csv('data/movies.csv')
    tags_df = pd.read_csv('data/tags.csv')
    links_df = pd.read_csv('data/links.csv')
    print("数据加载成功！")
    printData()
    initData()

# 打印数据集基本信息
def printData():
    global reting_df, movie_df, tags_df, links_df
    
    print("\n数据集基本信息：")
    print(f"评分数据：{len(reting_df)} 条记录")
    print(f"电影数据：{len(movie_df)} 条记录")
    print(f"标签数据：{len(tags_df)} 条记录")
    print(f"链接数据：{len(links_df)} 条记录")
    
    # 转换时间戳
    reting_df['timestamp'] = pd.to_datetime(reting_df['timestamp'], unit='s')
    tags_df['timestamp'] = pd.to_datetime(tags_df['timestamp'], unit='s')

# 数据预处理
def initData():
    global movie_df, reting_df, user_movie_matrix
    
    # 处理电影类型
    movie_df['genres'] = movie_df['genres'].str.split('|')
    
    # 构建用户-电影评分矩阵
    user_movie_matrix = reting_df.pivot(
        index='userId',
        columns='movieId',
        values='rating'
    ).fillna(0)
    
    print("\n用户-电影评分矩阵形状：", user_movie_matrix.shape)
    print("评分矩阵示例（前5行5列）：")
    print(user_movie_matrix.iloc[:5, :5])

# 获取所有可用的电影类型
def getAllGenres():
    global movie_df
    
    all_genres = set()
    for genres in movie_df['genres']:
        all_genres.update(genres)
    return sorted(list(all_genres))

# 获取用户推荐
def getUserRecommend(user_id, preferred_genres=None, time_period=None, n_recommendations=5):
    global user_movie_matrix, movie_df
    
    if user_id not in user_movie_matrix.index:
        return f"用户 {user_id} 不存在"
        
    # 计算用户相似度
    user_similarity = cosine_similarity(user_movie_matrix)
    user_similarity_df = pd.DataFrame(
        user_similarity,
        index=user_movie_matrix.index,
        columns=user_movie_matrix.index
    )
    
    # 获取目标用户的相似用户
    similar_users = user_similarity_df[user_id].sort_values(ascending=False)[1:6]
    
    # 获取目标用户未评分的电影
    user_ratings = user_movie_matrix.loc[user_id]
    unwatched_movies = user_ratings[user_ratings == 0].index
    
    # 根据用户偏好筛选电影
    filtered_movies = movie_df[movie_df['movieId'].isin(unwatched_movies)]
    
    if preferred_genres:
        # 筛选包含偏好类型的电影
        genre_mask = filtered_movies['genres'].apply(
            lambda x: any(genre in x for genre in preferred_genres)
        )
        filtered_movies = filtered_movies[genre_mask]
        
    if time_period:
        start_year, end_year = time_period
        # 从电影标题中提取年份
        filtered_movies['year'] = filtered_movies['title'].str.extract(r'\((\d{4})\)').astype(float)
        filtered_movies = filtered_movies[
            (filtered_movies['year'] >= start_year) & 
            (filtered_movies['year'] <= end_year)
        ]
        
    # 计算推荐分数
    recommendations = []
    for _, movie in filtered_movies.iterrows():
        movie_id = movie['movieId']
        # 获取相似用户对该电影的评分
        similar_user_ratings = user_movie_matrix.loc[similar_users.index, movie_id]
        # 计算加权平均评分
        weighted_rating = (similar_user_ratings * similar_users.values).sum() / similar_users.values.sum()
        
        # 如果用户指定了偏好类型，增加相关类型的权重
        if preferred_genres:
            genre_overlap = len(set(movie['genres']) & set(preferred_genres))
            weighted_rating *= (1 + 0.1 * genre_overlap)
            
        recommendations.append((movie_id, weighted_rating))
        
    # 排序并获取前N个推荐
    recommendations.sort(key=lambda x: x[1], reverse=True)
    top_recommendations = recommendations[:n_recommendations]
    
    # 获取电影详细信息
    recommended_movies = []
    for movie_id, score in top_recommendations:
        movie_info = movie_df[movie_df['movieId'] == movie_id].iloc[0]
        recommended_movies.append({
            'movie_id': movie_id,
            'title': movie_info['title'],
            'genres': '|'.join(movie_info['genres']),
            'predicted_rating': round(score, 2)
        })
        
    return recommended_movies

# 获取用户输入
def getUserInp():
    print("\n=== 电影推荐系统 ===")
    
    # 获取用户ID
    user_id = int(input("\n请输入您的用户ID (1-610): "))
    while not (1 <= user_id <= 610):
        print("用户ID必须在1-610之间")
        user_id = int(input("\n请输入您的用户ID (1-610): "))
    
    # 加载数据并获取电影类型偏好
    loadData()
    all_genres = getAllGenres()
    
    print("\n可用的电影类型：")
    for i, genre in enumerate(all_genres, 1):
        print(f"{i}. {genre}")
    
    preferred_genres = []
    genre_choice = input("\n请输入您感兴趣的电影类型编号（多个类型用逗号分隔，例如：1,2,3，直接回车跳过）: ")
    if genre_choice.strip():
        # 清理输入，移除空格
        genre_choice = genre_choice.replace(" ", "")
        # 分割并转换为数字
        genre_indices = [int(x) for x in genre_choice.split(',') if x]
        # 转换为0基索引并获取类型
        preferred_genres = [all_genres[x-1] for x in genre_indices]
        print(f"\n您选择的电影类型：{', '.join(preferred_genres)}")
    
    # 获取时间范围
    time_period = None
    year_range = input("\n请输入您感兴趣的电影年份范围（1900-2024）: ")
    if year_range.strip():
        start_year, end_year = map(int, year_range.split('-'))
        if 1900 <= start_year <= end_year <= 2024:
            time_period = (start_year, end_year)
    
    return user_id, preferred_genres, time_period

# 可视化用户评分分布
def visualize_rating_distribution():

    plt.figure(figsize=(10, 6))
    sns.histplot(data=reting_df, x='rating', bins=20)
    plt.title('用户评分分布')
    plt.xlabel('评分')
    plt.ylabel('频次')
    plt.savefig('rating_distribution.png')
    plt.close()

# 可视化电影类型分布
def visualize_genre_distribution():
    # 统计每种类型的电影数量
    genre_counts = {}
    for genres in movie_df['genres']:
        for genre in genres:
            genre_counts[genre] = genre_counts.get(genre, 0) + 1
    
    # 绘制条形图
    plt.figure(figsize=(12, 6))
    genres = list(genre_counts.keys())
    counts = list(genre_counts.values())
    plt.bar(genres, counts)
    plt.xticks(rotation=45, ha='right')
    plt.title('电影类型分布')
    plt.xlabel('类型')
    plt.ylabel('电影数量')
    plt.tight_layout()
    plt.savefig('genre_distribution.png')
    plt.close()

# 可视化推荐电影的评分
def visualize_recommendations(recommendations):
    plt.figure(figsize=(10, 6))
    titles = [movie['title'][:20] + '...' if len(movie['title']) > 20 else movie['title'] 
             for movie in recommendations]
    ratings = [movie['predicted_rating'] for movie in recommendations]
    
    plt.bar(titles, ratings)
    plt.xticks(rotation=45, ha='right')
    plt.title('推荐电影预测评分')
    plt.xlabel('电影标题')
    plt.ylabel('预测评分')
    plt.tight_layout()
    plt.savefig('recommendations_ratings.png')
    plt.close()

# 可视化用户相似度热力图
def visualize_user_similarity(user_id):
    # 获取前10个最相似的用户
    user_similarity = cosine_similarity(user_movie_matrix)
    user_similarity_df = pd.DataFrame(
        user_similarity,
        index=user_movie_matrix.index,
        columns=user_movie_matrix.index
    )
    
    similar_users = user_similarity_df[user_id].sort_values(ascending=False)[:10]
    
    plt.figure(figsize=(10, 6))
    sns.heatmap(user_similarity_df.loc[similar_users.index, similar_users.index], 
                annot=True, cmap='YlOrRd')
    plt.title(f'用户 {user_id} 的相似用户热力图')
    plt.tight_layout()
    plt.savefig('user_similarity.png')
    plt.close()

def main():
    # 获取用户输入
    user_id, preferred_genres, time_period = getUserInp()
    
    # 生成推荐
    if preferred_genres:
        print(f"偏好类型: {', '.join(preferred_genres)}")
    if time_period:
        print(f"时间范围: {time_period[0]}-{time_period[1]}")
        
    recommendations = getUserRecommend(
        user_id,
        preferred_genres=preferred_genres,
        time_period=time_period
    )
    
    # 生成可视化图表
    visualize_rating_distribution()
    visualize_genre_distribution()
    visualize_recommendations(recommendations)
    visualize_user_similarity(user_id)
    
    print("\n推荐电影列表：")
    for i, movie in enumerate(recommendations, 1):
        print(f"{i}. {movie['title']}")
        print(f"   类型: {movie['genres']}")
        print(f"   预测评分: {movie['predicted_rating']}")
        print()

if __name__ == "__main__":
    main()