# Stock History Similarity Search

13, Feb 多CPU上大规模多空交易

---

### 全局查找

**[查找股票]:**  000001.SZ    
**[起止时间]:** 2016-01-01 起 30个交易日     
**[数据集]:** 中证800五年的日行情数据    
**[预处理方式]:** 数据标准化    
**[相似衡量]：** 欧氏距离    
**[查询时间]:** 662s    

<img src="https://raw.githubusercontent.com/cheersyouran/simi-search/master/pic/000001.SZ-30-20160101-euclidean-all.jpg" width = "60%" height = "60%" alt="1" align=center />

---

### 基于3级快速傅里叶变换的快速查找

**[查找股票]:** 000001.SZ   
**[起止时间]:** 2016-01-01 起 30个交易日   
**[数据集]:** 中证800五年的日行情数据   
**[预处理方式]:** 数据标准化、3级快速傅里叶变换    
**[相似衡量]：** 欧氏距离    
**[查询时间]:** 20s   

<img src="https://raw.githubusercontent.com/cheersyouran/simi-search/master/pic/000001.SZ-30-20160101-euclidean-speed-fft3.jpg" width = "60%" height = "60%" alt="2" align=center />

------

### 基于小波降噪及3级快速傅里叶变换的快速查找(标准化)

**[查找股票]:** 000001.SZ   
**[起止时间]:** 2017-02-24 至 250个交易日   
**[数据集]:** 中证800五年的日行情数据  
**[预处理方式]:** 数据标准化、3级快速傅里叶变换    
**[相似衡量]：** 欧氏距离    
**[查询时间]:** 20s   

<img src="https://raw.githubusercontent.com/cheersyouran/simi-search/master/pic/000001.SZ-30-20160101-euclidean-speed-wave-fft3.jpg" width = "60%" height = "60%" alt="2" align=center />

------

### 基于3级快速傅里叶变换的快速查找(盈利率)

**[查找股票]:** 000001.SZ   
**[起止时间]:** 2016-01-01 起 30个交易日   
**[数据集]:** 中证800五年的日行情数据   
**[预处理方式]:** 盈利率、3级快速傅里叶变换    
**[相似衡量]：** 欧氏距离    
**[查询时间]:** 20s   

<img src="https://raw.githubusercontent.com/cheersyouran/simi-search/master/pic/000001.SZ-30-20160101-euclidean-speed-value-ratio-fft3.jpg" width = "60%" height = "60%" alt="2" align=center />

------

### 基于3级快速傅里叶变换的回归测试(盈利率)

**[查找股票]:** 000001.SZ   
**[起止时间]:** 2017-02-24 至 250个交易日    
**[数据集]:** 中证800五年的日行情数据    
**[相似衡量]：** 欧氏距离    

<img src="https://raw.githubusercontent.com/cheersyouran/simi-search/master/pic/RegressionTest(250D)-000001.SZ-30-201702024-euclidean-speed-value-ratio-fft3.jpg" width = "60%" height = "60%" alt="2" align=center />



