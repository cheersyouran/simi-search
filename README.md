# Stock History Similarity Search

## 目录
1. [基于欧几里得距离的查找结果](#1-基于欧几里得距离的查找结果)
2. [基于欧几里得距离和标准差的快速查找结果](#2-基于欧几里得距离和标准差的快速查找结果)
3. [基于欧几里得距离和小波变换的快速查找结果](#3-基于欧几里得距离和小波变换的快速查找结果)
4. [基于欧几里得距离和1级快速傅里叶变换的快速查找结果](#4-基于欧几里得距离和1级快速傅里叶变换的快速查找结果)
5. [基于欧几里得距离和3级快速傅里叶变换的快速查找结果](#5-基于欧几里得距离和3级快速傅里叶变换的快速查找结果)

---

### 1-基于欧几里得距离的查找结果

**[查找股票]:**  000001.SZ    
**[起止时间]:** 2016-01-01 至 30个交易日    
**[查找数据集]:** 中证800五年的日行情数据    
**[查询时间]:** 662s

<img src="https://raw.githubusercontent.com/cheersyouran/simi-search/master/pic/000001.SZ-30-20160101-euclidean-all.jpg" width = "60%" height = "60%" alt="1" align=center />

---

### 2-基于欧几里得距离和标准差的快速查找结果

**[查找股票]:** 000001.SZ   
**[起止时间]:** 2016-01-01 至 30个交易日   
**[查找数据集]:** 中证800五年的日行情数据 (head 1000)    
**[快速查找方法]:** 基于标准差的预处理   
**[查询时间]:** 90s

<img src="https://raw.githubusercontent.com/cheersyouran/simi-search/master/pic/000001.SZ-30-20160101-euclidean-part.jpg" width = "60%" height = "60%" alt="2" align=center />

---

### 3-基于欧几里得距离和小波变换的快速查找结果

**[查找股票]:** 000001.SZ   
**[起止时间]:** 2016-01-01 至 30个交易日   
**[查找数据集]:** 中证800五年的日行情数据 (head 1000)    
**[快速查找方法]:** 基于小波变换的预处理    
**[查询时间]:** 90s

<img src="https://raw.githubusercontent.com/cheersyouran/simi-search/master/pic/000001.SZ-30-20160101-euclidean-part-wavelet.jpg" width = "60%" height = "60%" alt="2" align=center />

---

### 4-基于欧几里得距离和1级快速傅里叶变换的快速查找结果

**[查找股票]:** 000001.SZ   
**[起止时间]:** 2016-01-01 至 30个交易日   
**[查找数据集]:** 中证800五年的日行情数据 (head 500)   
**[快速查找方法]:** 基于快速傅里叶变换的预处理-1级    
**[查询时间]:** 45s

<img src="https://raw.githubusercontent.com/cheersyouran/simi-search/master/pic/000001.SZ-30-20160101-euclidean-part-fft.jpg" width = "60%" height = "60%" alt="2" align=center />

---

### 5-基于欧几里得距离和3级快速傅里叶变换的快速查找结果

**[查找股票]:** 000001.SZ   
**[起止时间]:** 2016-01-01 至 30个交易日   
**[查找数据集]:** 中证800五年的日行情数据 (head 200)   
**[快速查找方法]:** 基于快速傅里叶变换的预处理-3级    
**[查询时间]:** 20s

<img src="https://raw.githubusercontent.com/cheersyouran/simi-search/master/pic/000001.SZ-30-20160101-euclidean-part-fft-3.jpg" width = "60%" height = "60%" alt="2" align=center />
------
