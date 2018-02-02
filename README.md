# Stock History Similarity Search

## 目录
1. [全局查找](#1-全局查找)
2. [3级快速傅里叶变换的快速查找](#2-3级快速傅里叶变换的快速查找)
3. [3级快速傅里叶变换的回归测试](#3-3级快速傅里叶变换的回归测试)
4. [小波降噪及3级快速傅里叶变换的快速查找](#4-基于欧式距离和3级快速傅里叶变换的回归测试)
5. [小波降噪及3级快速傅里叶变换的回归测试](#5-基于欧式距离和3级快速傅里叶变换的回归测试)

---

### 1-全局查找

**[查找股票]:**  000001.SZ    
**[起止时间]:** 2016-01-01 起 30个交易日    
**[数据集]:** 中证800五年的日行情数据
**[预处理方式]:** 数据标准化 
**[相似衡量]：** 欧氏距离
**[查询时间]:** 662s

<img src="https://raw.githubusercontent.com/cheersyouran/simi-search/master/pic/000001.SZ-30-20160101-euclidean-all.jpg" width = "60%" height = "60%" alt="1" align=center />

---

### 2-3级快速傅里叶变换的快速查找

**[查找股票]:** 000001.SZ   
**[起止时间]:** 2016-01-01 起 30个交易日   
**[数据集]:** 中证800五年的日行情数据   
**[预处理方式]:** 3级快速傅里叶变换
**[相似衡量]：** 欧氏距离
**[查询时间]:** 20s

<img src="https://raw.githubusercontent.com/cheersyouran/simi-search/master/pic/000001.SZ-30-20160101-euclidean-speed-fft3.jpg" width = "60%" height = "60%" alt="2" align=center />

------
### 3-3级快速傅里叶变换的回归测试

**[查找股票]:** 000001.SZ   
**[起止时间]:** 2017-02-024 至 250个交易日   
**[查找数据集]:** 中证800五年的日行情数据  
**[相似衡量]：** 欧氏距离  

<img src="https://raw.githubusercontent.com/cheersyouran/simi-search/master/pic/RegressionTest(250D)-000001.SZ-30-201702024-euclidean-speed-fft3.jpg" width = "60%" height = "60%" alt="2" align=center />
