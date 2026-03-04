# 由贝叶斯公式推导到 BM25（PRP → BIM → BM25）

本文按经典信息检索（IR）推导链路说明：先由贝叶斯得到**概率排序原理**（PRP），在一组假设下得到 **Binary Independence Model**（BIM）的对数似然比打分形式（对应 RSJ 权重/IDF），最后通过经验化但可解释的 **TF 饱和**与**长度归一**得到 BM25。

## 1. 目标：按相关后验概率排序（PRP）

给定查询 $q$ 与文档 $d$，令相关变量 $R \in \{0,1\}$，目标是按

$$
P(R=1 \mid d,q)
$$

对文档排序。

由贝叶斯公式：

$$
P(R=1\mid d,q)=\frac{P(d\mid R=1,q)\,P(R=1\mid q)}{P(d\mid q)}
$$

对固定的查询 $q$，$P(R=1\mid q)$ 与 $P(d\mid q)$ 对排序可视为常数（或与 $d$ 无关），因此排序等价于按 $P(d\mid R=1,q)$ 或更常见的**对数后验赔率**排序：

$$
\log\frac{P(R=1\mid d,q)}{P(R=0\mid d,q)}
=\log\frac{P(d\mid R=1,q)}{P(d\mid R=0,q)}
\log\frac{P(R=1\mid q)}{P(R=0\mid q)}
$$

最后一项对排序是常数，所以关键是对数似然比：

$$
\log\frac{P(d\mid R=1,q)}{P(d\mid R=0,q)}
$$

## 2. BIM：二元独立模型（由似然比得到加权求和）

用二元变量表示文档：对词表中每个词 $t$，令

$$
x_t(d)\in\{0,1\}
$$

表示词 $t$ 是否在文档 $d$ 中出现。

做两个标准假设：

1. **词独立性假设**：在给定相关性 $R$ 与查询 $q$ 后，各词出现相互独立
2. **只关心查询词**：非查询词的贡献并入常数项（对排序无影响）

对查询词 $t\in q$，定义：

$$
p_t = P(x_t=1 \mid R=1,q),\qquad u_t = P(x_t=1 \mid R=0,q)
$$

在独立性假设下：

$$
P(d\mid R,q)=\prod_{t\in q} P(x_t(d)\mid R,q)
$$

且对每个词：

$$
P(x_t(d)\mid R,q)=
\begin{cases}
p_t,& R=1,\,x_t(d)=1\\
1-p_t,& R=1,\,x_t(d)=0\\
u_t,& R=0,\,x_t(d)=1\\
1-u_t,& R=0,\,x_t(d)=0
\end{cases}
$$

把它代入对数似然比并化简，可得（忽略对排序无关的常数项）：

$$
\log\frac{P(d\mid R=1,q)}{P(d\mid R=0,q)}
\doteq
\sum_{t\in q} x_t(d)\cdot \log\frac{p_t(1-u_t)}{u_t(1-p_t)}
$$

因此 BIM 的打分形式是：

$$
\mathrm{score}_{BIM}(d,q)=\sum_{t\in q} x_t(d)\cdot w_t
$$

其中权重：

$$
w_t=\log\frac{p_t(1-u_t)}{u_t(1-p_t)}
$$

## 3. RSJ 权重与 IDF（由概率估计得到）

记：

- 语料总文档数：$N$
- 含词 $t$ 的文档数：$n_t$
-（若有相关反馈）相关文档数：$R$
-（若有相关反馈）相关文档中含词 $t$ 的文档数：$r_t$

常用频率估计为：

$$
p_t \approx \frac{r_t}{R},\qquad u_t \approx \frac{n_t-r_t}{N-R}
$$

代入 BIM 权重并加上 $0.5$ 平滑，得到 RSJ（Robertson–Spärck Jones）权重：

$$
w_t=
\log\frac{(r_t+0.5)\,(N-n_t-R+r_t+0.5)}{(R-r_t+0.5)\,(n_t-r_t+0.5)}
$$

在**无相关反馈**的常见场景下，可取 $R=0,r_t=0$，则上式退化为 BM25 常用的 IDF 形式：

$$
\mathrm{idf}(t)=\log\frac{N-n_t+0.5}{n_t+0.5}
$$

直观解释：越稀有的词（$n_t$ 越小）越能区分相关与不相关，因此权重更大。

## 4. 从二元出现到词频：TF 饱和

BIM 只用 $x_t(d)\in\{0,1\}$，但实践中查询词在文档中出现次数越多通常越相关，同时边际收益递减。因此用词频 $f_{t,d}$ 替换二元出现，并引入饱和函数。

BM25 使用：

$$
\mathrm{tf\_sat}(t,d)=\frac{(k_1+1) f_{t,d}}{k_1+f_{t,d}}
$$

其中 $k_1>0$ 控制饱和速度：$k_1$ 越大越接近线性 TF，越小饱和越快。

## 5. 文档长度归一

长文档天然更容易出现更多词、词频更高，需要长度归一。BM25 采用平均长度归一项：

$$
K(d)=k_1\left(1-b+b\frac{|d|}{\mathrm{avgdl}}\right)
$$

其中：

- $|d|$ 为文档长度
- $\mathrm{avgdl}$ 为语料平均文档长度
- $b\in[0,1]$ 控制长度归一强度（$b=1$ 强归一，$b=0$ 不归一）

于是 TF×长度项写为：

$$
\frac{(k_1+1) f_{t,d}}{f_{t,d}+K(d)}
=
\frac{(k_1+1) f_{t,d}}{f_{t,d}+k_1\left(1-b+b\frac{|d|}{\mathrm{avgdl}}\right)}
$$

## 6. 得到 BM25（最终公式）

把 IDF（来自 BIM/RSJ）与 TF×长度项结合，得到常用 BM25：

$$
\mathrm{score}_{BM25}(d,q)=\sum_{t\in q}\mathrm{idf}(t)\cdot
\frac{(k_1+1) f_{t,d}}{f_{t,d}+k_1\left(1-b+b\frac{|d|}{\mathrm{avgdl}}\right)}
$$

如需考虑查询端词频 $f_{t,q}$（查询很短时经常忽略），可加上查询饱和项：

$$
\frac{(k_3+1) f_{t,q}}{k_3+f_{t,q}}
$$

得到更一般的形式：

$$
\mathrm{score}_{BM25}(d,q)=\sum_{t\in q}\mathrm{idf}(t)\cdot
\frac{(k_1+1) f_{t,d}}{f_{t,d}+k_1\left(1-b+b\frac{|d|}{\mathrm{avgdl}}\right)}\cdot
\frac{(k_3+1) f_{t,q}}{k_3+f_{t,q}}
$$

## 7. 推导链路小结

- 贝叶斯公式 $\Rightarrow$ PRP：按 $P(R=1\mid d,q)$（或 log-odds）排序
- PRP + 词独立性 + 二元出现 $\Rightarrow$ BIM：对数似然比变成“出现词的加权求和”
- 估计 $p_t,u_t$ 并平滑 $\Rightarrow$ RSJ 权重；无反馈下 $\Rightarrow$ $\log\frac{N-n_t+0.5}{n_t+0.5}$（IDF）
- 加入 TF 饱和与长度归一 $\Rightarrow$ BM25

