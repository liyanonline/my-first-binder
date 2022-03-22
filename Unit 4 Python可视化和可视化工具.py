#ch10_p_1.py

#导入相关的库
import numpy as np
import pandas as pd
import datetime
import sqlalchemy
from sqlalchemy import Float,Integer,NVARCHAR,DATE

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

# zhfont1 = mpl.font_manager.FontProperties(fname='/usr/share/fonts/truetype/arphic/ukai.ttc') 
#读入csv中的数据，以“;”作为分割添加字段名
df = pd.read_csv('dianying.csv',delimiter=';',encoding='utf-8',names=['电影名称','上线时间','下线时间','公司','导演','主演','类型','票房','城市'])

#提取有效的列
dyxx = df.loc[:,['电影名称','上线时间','下线时间','票房','城市','类型']]

#输出初始行数，通过比较，理解去重操作
print('原始数据行数：')
print(len(dyxx))

#去重处理
dyxx = dyxx.drop_duplicates().reset_index().drop('index',axis=1)

#输出去重后的行数
print('去重后的行数：')
print(len(dyxx))

#去除票房列多余的符号“）”且转为float类型，规范处理
dyxx['票房'] = dyxx['票房'].str.split('）').str[1].astype(float)

#将时间列转换datetime时间类型，规范处理
dyxx['上线时间'] = pd.to_datetime(dyxx['上线时间'])
dyxx['下线时间'] = pd.to_datetime(dyxx['下线时间'])

#计算上映总天数
dyxx['上映天数'] = dyxx['下线时间']-dyxx['上线时间']+datetime.timedelta(days=1)

#按电影名称计算总票房，最多的上映天数
tjjg = dyxx.groupby('电影名称')['票房','上映天数'].agg({'票房':'sum','上映天数':'max'})

#将票房加上单位“万元”
dw = pd.Series(('万元*'*len(tjjg)).split('*'))
del dw[len(tjjg)] #删除最后的空元素
#将票房强制转换成str，再运用cat拼接
tjjg['票房'] = tjjg['票房'].astype('str').str.cat(others=dw)
#增加电影名称列
tjjg['电影名称'] =tjjg.index

#增加天数（整数类型）列
tjjg['天数']=0
for i in range(len(tjjg)):
    tjjg.iloc[i,3]=tjjg.iloc[i,1].days  #提取timedelta的天数

    
    
# #准备写入数据库
# def mapp(df):
#     dtypedict = {}
#     for i ,j in zip(df.columns,df.dtypes):  #列名与类型的映射
#         if 'object' in str(j):
#             dtypedict.update({i:NVARCHAR(255)})
#         if 'int' in str(j):
#             dtypedict.update({i:Integer()})
#         if 'float' in str(j):
#             dtypedict.update({i:Float(precision=2,asdecimal=True)})
#         if 'datetime64' in str(j):
#             dtypedict.update({i:DATE()})
#     return dtypedict

# del tjjg['上映天数']   #删除上映天数列
# dty = mapp(tjjg)      #生成写入的数据类型
# #创建引擎
# engine = sqlalchemy.create_engine("mysql+mysqldb://root:123123@localhost:3306/dyxx?charset=utf8")

# #写入数据
# tjjg.to_sql(name='tjjg',con=engine,index=False,dtype=dty,if_exists='replace')

# engine.dispose() #关闭引擎




#数据清洗并获取有效数据
def get_data(path):
    #读取数据
    film_data = pd.read_csv(path,delimiter=';', encoding='utf8', names=['电影名称','上线时间','下线时间','公司','导演','主演','类型','票房','城市'])
    # 去重
    film_data = film_data.drop_duplicates().reset_index().drop('index', axis=1)
    # 选择需要的列,并去空
    film_data=film_data[['电影名称','导演','类型','票房']].dropna()
    #对电影类型进行处理
    film_data['类型'] = film_data['类型'].str.strip()
    film_data['类型'] = film_data['类型'].str[0:2] #取前2个字符代表类型
    # 获取票房列数据，去除")"，并转换成浮点数
    film_data['票房'] = film_data['票房'].str.split('）', expand=True)[1].astype(np.float64)
    print(film_data) #调试用的，查看中间处理结果
    return film_data
#取得清洗后的数据，注意文件的位置
data = get_data('dianying.csv')
#对电影的票房进行求和
film_box_office = data.groupby(data['电影名称'])['票房'].sum()
#将统计结果的Series格式转换为DataFrame，并按降序排序
film_box_office = film_box_office.reset_index().sort_values(by='票房',ascending=False)
#取票房前5名的
film_box_office_5 = film_box_office.head()
#输出统计结果
                                                                       

#对电影类型进行计数
film_type = data.groupby(data['类型'])['电影名称'].count().reset_index()
#将“电影名称”列改为“小计”
film_type.rename(columns = {'电影名称':'小计'},inplace = True)
print(film_type) #调试用的，查看中间处理结果
#对导演的票房进行统计，并按降序排列
director_box_office = data.groupby(['导演'])['票房'].sum().reset_index().sort_values(by='票房',ascending=False)
director_box_office_5 = director_box_office.head()
print(director_box_office_5) #调试用的，查看中间处理结果
#对导演所导电影类型进行统计
director = data.groupby(['导演','类型'])['票房'].count().reset_index()
#将“票房”改为“小计”
director.rename(columns = {'票房':'小计'},inplace = True)
print(director) #调试用的，查看中间处理结果
# 画图
fig = plt.figure(figsize=(15,15)) #创建画布
ax_1 = fig.add_subplot(2,2,1) #添加子图
ax_2 = fig.add_subplot(2,2,2)
ax_3 = fig.add_subplot(2,2,3)
ax_4 = fig.add_subplot(2,2,4)

#票房前五的电影
# ax_1.set_title("票房总计",fontproperties=zhfont1)
# #ax_1.set_xlabel('电影名称',fontproperties=zhfont1)
# # ax_1.set_ylabel('万元',fontproperties=zhfont1)
# # ax_1.set_xticklabels(film_box_office_5['电影名称'],fontproperties=zhfont1,rotation=15) #文字显示旋转15度
ax_1.bar(film_box_office_5['电影名称'],film_box_office_5['票房'])
# #电影类型统计
# # ax_2.set_title("电影类型统计",fontproperties=zhfont1)
# ax_2.pie(film_type['小计'],labels=film_type['类型'],textprops={'fontsize': 12, 'color': 'black','fontproperties':zhfont1},autopct='%1.2f%%',shadow=True)
ax_2.pie(film_type['小计'],labels=film_type['类型'],textprops={'fontsize': 12, 'color': 'black'},autopct='%1.2f%%',shadow=True)


# #导演票房前五的统计
# # ax_3.set_title("导演票房总计",fontproperties=zhfont1)
# # ax_3.set_xlabel('导演',fontproperties=zhfont1)
# ax_3.set_ylabel('万元',fontproperties=zhfont1)
# # ax_3.set_xticklabels(director_box_office_5['导演'],fontproperties=zhfont1)
ax_3.bar(director_box_office_5 ['导演'], director_box_office_5 ['票房'])
# #导演与类型统计
# ax_4.set_title("导演与电影类型统计",fontproperties=zhfont1)
# ax_4.set_xticklabels(director['导演'],rotation=90,fontproperties=zhfont1)#文字显示旋转90度
# ax_4.set_yticklabels(director['类型'],fontproperties=zhfont1)
# #点的大小由分类统计的数量决定
ax_4.scatter(director['导演'],director['类型'],s=director['小计']*50,edgecolors="red")
plt.show() #显示图形


