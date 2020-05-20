import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
import codecs
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from scipy import interpolate
# listType = df_1['product_parent'].unique()

FileNameStr_hairdryer = 'Problem_C_Data/hair_dryer.tsv'
FileNameStr_microwave = 'Problem_C_Data/microwave.tsv'
FileNameStr_pacifier = 'Problem_C_Data/pacifier.tsv'


def prepare(FileNameStr):
    df = pd.read_csv(FileNameStr, sep='\t')
    review_df = df.loc[:,
                ['review_id', 'product_parent', 'product_title', 'star_rating', 'helpful_votes', 'total_votes',
                 'vine', 'verified_purchase', 'review_body', 'review_date']]
    review_df = review_df.dropna(subset=['review_id', 'product_parent', 'product_title', 'star_rating', 'helpful_votes',
                                         'total_votes', 'vine', 'verified_purchase', 'review_body', 'review_date'],
                                 how='any')
    review_df['review_date'] = pd.to_datetime(review_df['review_date'], format='%m/%d/%Y')
    review_df = review_df.sort_values(by='review_date', ascending=True, na_position='first')
    review_df = review_df.reset_index(drop=True)
    return review_df


def num_of_five_stars(subreviewsDf):
    # 选出分为5的
    querySer = subreviewsDf.loc[:, 'star_rating'] == 5
    sub1Df = subreviewsDf.loc[querySer, :]
    sub1Df.head()
    sub1Df.shape[0]
    # 按产品进行分类
    sub1Df1 = pd.DataFrame(sub1Df.groupby(["product_title"], sort=True)["star_rating"].size()).reset_index()
    # 列重命名
    NameDict = {'star_rating': 'five_stars'}
    sub1Df1.rename(columns=NameDict, inplace=True)
    # 按好评数降序排列
    sub1Df1 = sub1Df1.sort_values(by='five_stars', ascending=False, na_position='first')
    # 取前5结果
    sub1Df1 = sub1Df1.head(5)
    return sub1Df1


#总体星级分类
def percentage_of_each_rating(subreviewsDf):
    # 按产品进行分类
    sub2Df1 = pd.DataFrame(subreviewsDf.groupby(["star_rating"], sort=True)["product_title"].size()).reset_index()
    # 列重命名
    NameDict = {'product_title': 'total_reviews'}
    sub2Df1.rename(columns=NameDict, inplace=True)
    a = sub2Df1.loc[:, 'total_reviews'].sum()
    # 定义新列
    aDf = pd.DataFrame()
    aDf['percentage'] = sub2Df1['total_reviews'] / a
    sub2Df1 = pd.concat([sub2Df1, aDf], axis=1)
    return sub2Df1


def top_rating(subreviewsDf,str):
    querySer = subreviewsDf.loc[:, 'product_title'] == str
    sub2Df = subreviewsDf.loc[querySer, :]
    # 按产品进行分类
    sub2Df2 = pd.DataFrame(sub2Df.groupby(["star_rating"], sort=True)["product_title"].size()).reset_index()
    # 列重命名
    NameDict = {'product_title': 'total_reviews'}
    sub2Df2.rename(columns=NameDict, inplace=True)
    a = sub2Df2.loc[:, 'total_reviews'].sum()
    # 定义新列
    aDf = pd.DataFrame()
    aDf['percentage'] = sub2Df2['total_reviews'] / a
    sub2Df2 = pd.concat([sub2Df2, aDf], axis=1)
    print(sub2Df2)
    return sub2Df2


review_df_hairdryer = prepare(FileNameStr_hairdryer)
review_df_microwave = prepare(FileNameStr_microwave)
review_df_pacifier = prepare(FileNameStr_pacifier)


def to_percent(temp, position):
    return '%1.0f'%(10*temp) + '%'


# #1、创建画布
# plt.figure(figsize=(20, 8), dpi=200)
#
# # 2、绘制图像
# x= range(1, 6, 1)
# plt.plot(x, top_rating(review_df_hairdryer,'remington ac2015 t|studio salon collection pearl ceramic hair dryer, deep purple')['percentage']*10,
#          linewidth='2', label="remington ac2015 t|studio salon collection pearl ceramic hair dryer, deep purple")
# plt.plot(x, top_rating(review_df_microwave,'danby 0.7 cu.ft. countertop microwave')['percentage']*10,
#          linewidth = '2', label = "danby 0.7 cu.ft. countertop microwave")
# plt.plot(x, top_rating(review_df_pacifier,'philips avent bpa free soothie pacifier, 0-3 months, 2 pack, packaging may vary')['percentage']*10,
#          linewidth = '2', label = "philips avent bpa free soothie pacifier, 0-3 months, 2 pack, packaging may vary")
# plt.gca().yaxis.set_major_formatter(FuncFormatter(to_percent))
# plt.xticks(x)
# plt.yticks(range(1,11))
# plt.legend(loc = 'upper left')
# plt.title('Rating distribution of top rating goods')
# # 3、显示图像
# plt.show()
# top_rating(review_df_hairdryer,'remington ac2015 t|studio salon collection pearl ceramic hair dryer, deep purple').to_csv('new_data/top_of_hairdryer')
# top_rating(review_df_microwave,'danby 0.7 cu.ft. countertop microwave').to_csv('new_data/top_of_microwave')
# top_rating(review_df_pacifier,'philips avent bpa free soothie pacifier, 0-3 months, 2 pack, packaging may vary').to_csv('new_data/top_of_pacifier')
def plot_comment(df_temp):
    sub1Df2 = pd.DataFrame(df_temp.groupby(["review_date"],sort=True)["star_rating"].size()).reset_index()
    NameDict = {'star_rating': 'total_votes'}
    sub1Df2.rename(columns=NameDict, inplace=True)
    return sub1Df2


def get_specific_day_stars(df,date,num):
    if num == 0:
        querySer = df.loc[:, 'review_date'] == date
    elif num < 0:
        querySer = df.loc[:, 'review_date'] < date
    else:
        querySer = df.loc[:, 'review_date'] > date
    sub2Df = df.loc[querySer, :]
    # 按产品进行分类
    sub2Df2 = pd.DataFrame(sub2Df.groupby(["star_rating"], sort=True)["product_title"].size()).reset_index()
    # 列重命名
    NameDict = {'product_title': 'total_reviews'}
    sub2Df2.rename(columns=NameDict, inplace=True)
    a = sub2Df2.loc[:, 'total_reviews'].sum()
    # 定义新列
    aDf = pd.DataFrame()
    aDf['percentage'] = sub2Df2['total_reviews'] / a
    sub2Df2 = pd.concat([sub2Df2, aDf], axis=1)
    return sub2Df2


def get_specific_Day_stars_full(df,date,num):
    if num == 0:
        querySer = df.loc[:, 'review_date'] == date
    elif num < 0:
        querySer = df.loc[:, 'review_date'] < date
    else:
        querySer = df.loc[:, 'review_date'] > date
    sub2Df = df.loc[querySer, :]
    return sub2Df

# plt.plot(plot_comment(review_df_hairdryer).loc[:,'review_date'],
#          plot_comment(review_df_hairdryer).loc[:,'star_rating'],
#          label = 'Hairdryer')
# plt.plot(plot_comment(review_df_microwave).loc[:,'review_date'],
#          plot_comment(review_df_microwave).loc[:,'star_rating'],
#          label = 'Microwave')
# plt.plot(plot_comment(review_df_pacifier).loc[:,'review_date'],
#          plot_comment(review_df_pacifier).loc[:,'star_rating'],
#          label = 'Pacifier')
# plt.legend(loc = 'upper left')
# #x轴文本
# plt.xlabel('Time')
# #y轴文本
# plt.ylabel('Total_votes')
# #标题
# plt.title('The number of votes change over time')
# plt.show()
def get_the_day(df):
    df = plot_comment(df).sort_values(by='total_votes',ascending=False,na_position='first')
    return df
def draw_pie(df,num):
    df = get_specific_day_stars(df,'2013-07-06',num)
    print(df)
    label_list = ["One star", "Two stars", "Three stars", "Four stars", "Five stars"]    # 各部分标签
    size = df['percentage']    # 各部分大小
    # color = ["r", "g", "b"]     # 各部分颜色
    # explode = [0.05, 0, 0]   # 各部分突出值
    # plt.figure(figsize=(20, 15), dpi=400)
    plt.pie(size,labels=label_list, labeldistance=1.1, autopct="%1.1f%%", shadow=True, startangle=90, pctdistance=0.6)
    plt.axis("equal")    # 设置横轴和纵轴大小相等，这样饼才是圆的
    plt.title("Frequency of star_rating")
    plt.legend(loc='upper right')
    plt.show()


x=['Before the peak', 'Peak', 'After the peak']
plt.plot(x,[7.07,7.19,7.74],label='Hair_dryer',marker='D',ms=10,linewidth=2)
plt.plot(x,[4.76,5.00,6.15],label='Microwave',marker='d',ms=10,linewidth=2)
plt.plot(x,[7.93,8.03,8.19],label='Pcifier',marker='*',ms=10,linewidth=2)
plt.legend(loc='lower right')
plt.title('The trend of favorable rating')
plt.gca().yaxis.set_major_formatter(FuncFormatter(to_percent))
plt.show()