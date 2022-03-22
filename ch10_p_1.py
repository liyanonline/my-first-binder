#ch10_p_1.py

#导入相关的库
import pandas as pd
import datetime
import sqlalchemy
from sqlalchemy import Float,Integer,NVARCHAR,DATE

#读入csv中的数据，以“;”作为分割添加字段名
df = pd.read_csv('dianying.csv',delimiter=';',encoding='utf-8',names=['电影名称','上线时间','下线时间','公司','导演','主演','类型','票>房','城市'])

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

#准备写入数据库
def mapp(df):
    dtypedict = {}
    for i ,j in zip(df.columns,df.dtypes):  #列名与类型的映射
        if 'object' in str(j):
            dtypedict.update({i:NVARCHAR(255)})
        if 'int' in str(j):
            dtypedict.update({i:Integer()})
        if 'float' in str(j):
            dtypedict.update({i:Float(precision=2,asdecimal=True)})
        if 'datetime64' in str(j):
            dtypedict.update({i:DATE()})
    return dtypedict

del tjjg['上映天数']   #删除上映天数列
dty = mapp(tjjg)      #生成写入的数据类型
#创建引擎
engine = sqlalchemy.create_engine("mysql+mysqldb://root:123123@localhost:3306/dyxx?charset=utf8")

#写入数据
tjjg.to_sql(name='tjjg',con=engine,index=False,dtype=dty,if_exists='replace')

engine.dispose() #关闭引擎
