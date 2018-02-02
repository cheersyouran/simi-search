# Stock History Similarity Search

## 目录
1. [全局查找](#1-全局查找)
2. [基于3级快速傅里叶变换的快速查找(标准化)](#2-基于3级快速傅里叶变换的快速查找(标准化))
3. [基于3级快速傅里叶变换的回归测试(标准化)](#3-基于3级快速傅里叶变换的回归测试(标准化))
4. [基于小波降噪及3级快速傅里叶变换的快速查找](#4-基于小波降噪及3级快速傅里叶变换的快速查找)
5. [基于小波降噪及3级快速傅里叶变换的回归测试](#5-基于小波降噪及3级快速傅里叶变换的回归测试)
6. [基于3级快速傅里叶变换的快速查找(盈利率)](#6-基于3级快速傅里叶变换的快速查找)
7. [基于3级快速傅里叶变换的回归测试(盈利率)](#7-基于3级快速傅里叶变换的回归测试)

---

### 1-全局查找

**[查找股票]:**  000001.SZ    
**[起止时间]:** 2016-01-01 起 30个交易日    
**[数据集]:** 中证800五年的日行情数据
**[预处理方式]:** 数据标准化 
**[相似衡量]：** 欧氏距离
**[查询时间]:** 662s

<img src="https://raw.githubusercontent.com/cheersyouran/simi-search/master/pic/000001.SZ-30-20160101-euclidean-all.jpg" width = "60%" height = "60%" alt="1" align=center />

**[查找股票]:**  000001.SZ    
**[起止时间]:** 2016-01-01 起 30个交易日    
**[数据集]:** 中证800五年的日行情数据
**[预处理方式]:** 数据标准化 
**[相似衡量]：** 皮尔森系数
**[查询时间]:** 662s

<img src="https://raw.githubusercontent.com/cheersyouran/simi-search/master/pic/000001.SZ-30-20160101-person-all.jpg" width = "60%" height = "60%" alt="1" align=center />

---

### 2-基于3级快速傅里叶变换的快速查找

**[查找股票]:** 000001.SZ   
**[起止时间]:** 2016-01-01 起 30个交易日   
**[数据集]:** 中证800五年的日行情数据   
**[预处理方式]:** 数据标准化、3级快速傅里叶变换
**[相似衡量]：** 欧氏距离
**[查询时间]:** 20s

<img src="https://raw.githubusercontent.com/cheersyouran/simi-search/master/pic/000001.SZ-30-20160101-euclidean-speed-fft3.jpg" width = "60%" height = "60%" alt="2" align=center />

------

### 3-基于3级快速傅里叶变换的回归测试

**[查找股票]:** 000001.SZ   
**[起止时间]:** 2017-02-24 至 250个交易日   
**[数据集]:** 中证800五年的日行情数据  
**[相似衡量]：** 欧氏距离  

<img src="https://raw.githubusercontent.com/cheersyouran/simi-search/master/pic/RegressionTest(250D)-000001.SZ-30-201702024-euclidean-speed-fft3.jpg" width = "60%" height = "60%" alt="2" align=center />

------

### 4-基于小波降噪及3级快速傅里叶变换的快速查找(标准化)

**[查找股票]:** 000001.SZ   
**[起止时间]:** 2017-02-24 至 250个交易日   
**[数据集]:** 中证800五年的日行情数据  
**[预处理方式]:** 数据标准化、3级快速傅里叶变换
**[相似衡量]：** 欧氏距离  
**[查询时间]:** 20s

<img src="https://raw.githubusercontent.com/cheersyouran/simi-search/master/pic/000001.SZ-30-20160101-euclidean-speed-wave-fft3.jpg" width = "60%" height = "60%" alt="2" align=center />

------

### 5-基于小波降噪及3级快速傅里叶变换的回归测试(标准化)

**[查找股票]:** 000001.SZ   
**[起止时间]:** 2017-02-24 至 250个交易日   
**[查找数据集]:** 中证800五年的日行情数据  
**[相似衡量]：** 欧氏距离  

<img src="https://raw.githubusercontent.com/cheersyouran/simi-search/master/pic/RegressionTest(250D)-000001.SZ-30-201702024-euclidean-speed-wave-fft3.jpg" width = "60%" height = "60%" alt="2" align=center />

------

### 6-基于3级快速傅里叶变换的快速查找(盈利率)

**[查找股票]:** 000001.SZ   
**[起止时间]:** 2016-01-01 起 30个交易日   
**[数据集]:** 中证800五年的日行情数据   
**[预处理方式]:** 盈利率、3级快速傅里叶变换
**[相似衡量]：** 欧氏距离
**[查询时间]:** 20s

<img src="https://raw.githubusercontent.com/cheersyouran/simi-search/master/pic/000001.SZ-30-20160101-euclidean-speed-value-ratio-fft3.jpg" width = "60%" height = "60%" alt="2" align=center />

------

### 7-基于3级快速傅里叶变换的回归测试(盈利率)

**[查找股票]:** 000001.SZ   
**[起止时间]:** 2017-02-24 至 250个交易日   
**[数据集]:** 中证800五年的日行情数据  
**[相似衡量]：** 欧氏距离  

<img src="https://raw.githubusercontent.com/cheersyouran/simi-search/master/pic/RegressionTest(250D)-000001.SZ-30-201702024-euclidean-speed-value-ratio-fft3.jpg" width = "60%" height = "60%" alt="2" align=center />
