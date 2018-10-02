#Authors: Aayush Patial (apatial) , Rajat Narang (rnarang) , Sanya Kathuria (skathur2)
# coding: utf-8

# In[1]:


# import data
import pandas as pd
seeds = pd.read_csv(‘/Data/seeds.csv')
raw_dataset = seeds[['area_A', 'kernel_width']]

# Part a)
# normalize and standardize data
normalized_dataset = (raw_dataset - raw_dataset.min()) / (raw_dataset.max() - raw_dataset.min())
standardized_dataset = (raw_dataset - raw_dataset.mean()) / raw_dataset.std()


# In[2]:


# print range raw_dataset
print(raw_dataset.min())
print(raw_dataset.max())


# In[3]:


# print range standardized_dataset
print(standardized_dataset.min())
print(standardized_dataset.max())


# In[4]:


# print range normalized_dataset
print(normalized_dataset.min())
print(normalized_dataset.max())


# In[5]:


# Part b) i) Scatter Plot
import matplotlib.pyplot as plt
raw_dataset.plot(x='area_A', y='kernel_width', kind='scatter')


# In[6]:


normalized_dataset.plot(x='area_A', y='kernel_width', kind='scatter')


# In[7]:


standardized_dataset.plot(x='area_A', y='kernel_width', kind='scatter')


# In[8]:


# import libraries
import numpy as np
from scipy.spatial.distance import mahalanobis, cosine, canberra, euclidean, cityblock, minkowski, chebyshev
import scipy as sp

# Part b) ii) mean of area_A and kernel_width values
P_raw = (raw_dataset['area_A'].mean(), raw_dataset['kernel_width'].mean())
P_raw = np.array(P_raw)


# In[9]:


P_raw


# In[10]:


P_norm = (normalized_dataset['area_A'].mean(), normalized_dataset['kernel_width'].mean())
P_norm = np.array(P_norm)


# In[11]:


P_norm


# In[12]:


P_std = (standardized_dataset['area_A'].mean(), standardized_dataset['kernel_width'].mean())
P_std = np.array(P_std)


# In[13]:


P_std


# In[14]:


# Part b) iii)
# Euclidean Distance, raw_dataset
ed_r = raw_dataset.apply(lambda x: euclidean(x, P_raw), axis=1)
ed_r.head()


# In[15]:


# Euclidean Distance, normalized_dataset
ed_n = normalized_dataset.apply(lambda x: euclidean(x, P_norm), axis=1)
ed_n.head()


# In[16]:


# Euclidean Distance, standardized_dataset
ed_s = standardized_dataset.apply(lambda x: euclidean(x, P_std), axis=1)
ed_s.head()


# In[17]:


# Mahalanobis Distance, raw_dataset
Sx = raw_dataset.cov().values
Sx = sp.linalg.inv(Sx)
ma_r = raw_dataset.apply(lambda x: mahalanobis(x, P_raw, Sx), axis=1)
ma_r.head()


# In[18]:


# Mahalanobis Distance, normalized_dataset
Sx = normalized_dataset.cov().values
Sx = sp.linalg.inv(Sx)
ma_n = normalized_dataset.apply(lambda x: mahalanobis(x, P_norm, Sx), axis=1)
ma_n.head()


# In[19]:


# Mahalanobis Distance, standardized_dataset
Sx = standardized_dataset.cov().values
Sx = sp.linalg.inv(Sx)
ma_s = standardized_dataset.apply(lambda x: mahalanobis(x, P_std, Sx), axis=1)
ma_s.head()


# In[20]:


# City Block Metric, raw_dataset
cb_r = raw_dataset.apply(lambda x: cityblock(x, P_raw), axis=1)
cb_r.head()


# In[21]:


# City Block Metric, normalized_dataset
cb_n = normalized_dataset.apply(lambda x: cityblock(x, P_norm), axis=1)
cb_n.head()


# In[22]:


# City Block Metric, standardized_dataset
cb_s = standardized_dataset.apply(lambda x: cityblock(x, P_std), axis=1)
cb_s.head()


# In[23]:


# Minkowski metric, raw_dataset
r = 3
mi_raw = raw_dataset.apply(lambda x: minkowski(x, P_raw, r), axis=1)
mi_raw.head()


# In[24]:


# Minkowski metric, normalized_dataset
mi_norm = normalized_dataset.apply(lambda x: minkowski(x, P_norm, r), axis=1)
mi_norm.head()


# In[25]:


# Minkowski metric, standardized_dataset
mi_std = standardized_dataset.apply(lambda x: minkowski(x, P_std, r), axis=1)
mi_std.head()


# In[26]:


# Chebyshev distance, raw_dataset
ch_r = raw_dataset.apply(lambda x: chebyshev(x, P_raw), axis=1)
ch_r.head()


# In[27]:


# Chebyshev distance, normalized_dataset
ch_n = normalized_dataset.apply(lambda x: chebyshev(x, P_norm), axis=1)
ch_n.head()


# In[28]:


# Chebyshev distance, standardized_dataset
ch_s = standardized_dataset.apply(lambda x: chebyshev(x, P_std), axis=1)
ch_s.head()


# In[29]:


# Cosine Distance, raw_dataset
cos_r = raw_dataset.apply(lambda x: cosine(x, P_raw), axis=1)
cos_r.head()


# In[30]:


# Cosine Distance, normalized_dataset
cos_n = normalized_dataset.apply(lambda x: cosine(x, P_norm), axis=1)
cos_n.head()


# In[31]:


# Cosine Distance, standardized_dataset
cos_s = standardized_dataset.apply(lambda x: cosine(x, P_std), axis=1)
cos_s.head()


# In[32]:


# Canberra Distance, raw_dataset
ca_r = raw_dataset.apply(lambda x: canberra(x, P_raw), axis=1)
ca_r.head()


# In[33]:


# Canberra Distance, normalized_dataset
ca_n = normalized_dataset.apply(lambda x: canberra(x, P_norm), axis=1)
ca_n.head()


# In[34]:


# Canberra Distance, standardized_dataset
ca_s = standardized_dataset.apply(lambda x: canberra(x, P_std), axis=1)
ca_s.head()


# In[35]:


# Part b) iv)

# Closest 10 by euclidean distance on raw_dataset
closest_10_ed_r = ed_r.nsmallest(10)
closest_indices_ed_r = [i for i in closest_10_ed_r.index]
raw_dataset.iloc[closest_indices_ed_r][['area_A', 'kernel_width']]


# In[36]:


# Closest 10 by euclidean distance on normalized_dataset
closest_10_ed_n = ed_n.nsmallest(10)
closest_indices_ed_n = [i for i in closest_10_ed_n.index]
normalized_dataset.iloc[closest_indices_ed_n][['area_A', 'kernel_width']]


# In[37]:


# Closest 10 by euclidean distance on standardized_dataset
closest_10_ed_s = ed_s.nsmallest(10)
closest_indices_ed_s = [i for i in closest_10_ed_s.index]
standardized_dataset.iloc[closest_indices_ed_s][['area_A', 'kernel_width']]


# In[38]:


# Closest 10 by Mahalanobis distance on raw_dataset
closest_10_ma_r = ma_r.nsmallest(10)
closest_indices_ma_r = [i for i in closest_10_ma_r.index]
raw_dataset.iloc[closest_indices_ma_r][['area_A', 'kernel_width']]


# In[39]:


# Closest 10 by Mahalanobis distance on normalized_dataset
closest_10_ma_n = ma_n.nsmallest(10)
closest_indices_ma_n = [i for i in closest_10_ma_n.index]
normalized_dataset.iloc[closest_indices_ma_n][['area_A', 'kernel_width']]


# In[40]:


# Closest 10 by Mahalanobis distance on standardized_dataset
closest_10_ma_s = ma_s.nsmallest(10)
closest_indices_ma_s = [i for i in closest_10_ma_s.index]
standardized_dataset.iloc[closest_indices_ma_s][['area_A', 'kernel_width']]


# In[41]:


# Closest 10 points by City Block Metric on on raw_dataset
closest_10_cb_r = cb_r.nsmallest(10)
closest_indices_cb_r = [i for i in closest_10_cb_r.index]
raw_dataset.iloc[closest_indices_cb_r][['area_A', 'kernel_width']]


# In[42]:


# Closest 10 points by City Block Metric on on normalized_dataset
closest_10_cb_n = cb_n.nsmallest(10)
closest_indices_cb_n = [i for i in closest_10_cb_n.index]
normalized_dataset.iloc[closest_indices_cb_n][['area_A', 'kernel_width']]


# In[43]:


# Closest 10 points by City Block Metric on on standardized_dataset
closest_10_cb_s = cb_s.nsmallest(10)
closest_indices_cb_s = [i for i in closest_10_cb_s.index]
standardized_dataset.iloc[closest_indices_cb_s][['area_A', 'kernel_width']]


# In[44]:


# Closest 10 points by Minkowski metric on raw_dataset
closest_10_mi_raw = mi_raw.nsmallest(10)
closest_indices_mi_raw = [i for i in closest_10_mi_raw.index]
raw_dataset.iloc[closest_indices_mi_raw][['area_A', 'kernel_width']]


# In[45]:


# Closest 10 points by Minkowski metric on normalized_dataset
closest_10_mi_norm = mi_norm.nsmallest(10)
closest_indices_mi_norm = [i for i in closest_10_mi_norm.index]
normalized_dataset.iloc[closest_indices_mi_norm][['area_A', 'kernel_width']]


# In[46]:


# Closest 10 points by Minkowski metric on standardized_dataset
closest_10_mi_std = mi_std.nsmallest(10)
closest_indices_mi_std = [i for i in closest_10_mi_std.index]
standardized_dataset.iloc[closest_indices_mi_std][['area_A', 'kernel_width']]


# In[47]:


# Closest 10 points by Chebyshev distance on raw_dataset
closest_10_ch_r = ch_r.nsmallest(10)
closest_indices_ch_r = [i for i in closest_10_ch_r.index]
raw_dataset.iloc[closest_indices_ch_r][['area_A', 'kernel_width']]


# In[48]:


# Closest 10 points by Chebyshev distance on normalized_dataset
closest_10_ch_n = ch_n.nsmallest(10)
closest_indices_ch_n = [i for i in closest_10_ch_n.index]
normalized_dataset.iloc[closest_indices_ch_n][['area_A', 'kernel_width']]


# In[49]:


# Closest 10 points by Chebyshev distance on standardized_dataset
closest_10_ch_s = ch_s.nsmallest(10)
closest_indices_ch_s = [i for i in closest_10_ch_s.index]
standardized_dataset.iloc[closest_indices_ch_s][['area_A', 'kernel_width']]


# In[50]:


# Closest 10 points by Cosine distance on raw_dataset
closest_10_cos_r = cos_r.nsmallest(10)
closest_indices_cos_r = [i for i in closest_10_cos_r.index]
raw_dataset.iloc[closest_indices_cos_r][['area_A', 'kernel_width']]


# In[51]:


# Closest 10 points by Cosine distance on normalized_dataset
closest_10_cos_n = cos_n.nsmallest(10)
closest_indices_cos_n = [i for i in closest_10_cos_n.index]
normalized_dataset.iloc[closest_indices_cos_n][['area_A', 'kernel_width']]


# In[52]:


# Closest 10 points by Cosine distance on standardized_dataset
closest_10_cos_s = cos_s.nsmallest(10)
closest_indices_cos_s = [i for i in closest_10_cos_s.index]
standardized_dataset.iloc[closest_indices_cos_s][['area_A', 'kernel_width']]


# In[53]:


# Closest 10 points by Canberra distance on raw_dataset
closest_10_ca_r = ca_r.nsmallest(10)
closest_indices_ca_r = [i for i in closest_10_ca_r.index]
raw_dataset.iloc[closest_indices_ca_r][['area_A', 'kernel_width']]


# In[54]:


# Closest 10 points by Canberra distance on normalized_dataset
closest_10_ca_n = ca_n.nsmallest(10)
closest_indices_ca_n = [i for i in closest_10_ca_n.index]
normalized_dataset.iloc[closest_indices_ca_n][['area_A', 'kernel_width']]


# In[55]:


# Closest 10 points by Canberra distance on standardized_dataset
closest_10_ca_s = ca_s.nsmallest(10)
closest_indices_ca_s = [i for i in closest_10_ca_s.index]
standardized_dataset.iloc[closest_indices_ca_s][['area_A', 'kernel_width']]


# In[56]:


# Part b) vi): Plote all points, mark P(the mean point) and highlight neighbors

# Euclidean Distance, raw_dataset
x_raw, y_raw = raw_dataset['area_A'], raw_dataset['kernel_width']
closest10_x_raw_ed_r = raw_dataset.iloc[closest_10_ed_r.index]['area_A']
closest10_y_raw_ed_r = raw_dataset.iloc[closest_10_ed_r.index]['kernel_width']
plt.scatter(x_raw, y_raw)
plt.scatter(closest10_x_raw_ed_r, closest10_y_raw_ed_r, color='yellow')
plt.scatter(P_raw[0], P_raw[1], color='red', marker='x')
plt.xlabel('area_A')
plt.ylabel('kernel_width')


# In[57]:


# Euclidean Distance, normalized_dataset
x_norm, y_norm = normalized_dataset['area_A'], normalized_dataset['kernel_width']
closest10_x_norm_ed_n = normalized_dataset.iloc[closest_10_ed_n.index]['area_A']
closest10_y_norm_ed_n = normalized_dataset.iloc[closest_10_ed_n.index]['kernel_width']
plt.scatter(x_norm, y_norm)
plt.scatter(closest10_x_norm_ed_n, closest10_y_norm_ed_n, color='yellow')
plt.scatter(P_norm[0], P_norm[1], color='red', marker='x')
plt.xlabel('area_A')
plt.ylabel('kernel_width')


# In[58]:


# Euclidean Distance, standardized_dataset
x_std, y_std = standardized_dataset['area_A'], standardized_dataset['kernel_width']
closest10_x_std_ed_s = standardized_dataset.iloc[closest_10_ed_s.index]['area_A']
closest10_y_std_ed_s = standardized_dataset.iloc[closest_10_ed_s.index]['kernel_width']
plt.scatter(x_std, y_std)
plt.scatter(closest10_x_std_ed_s, closest10_y_std_ed_s, color='yellow')
plt.scatter(P_std[0], P_std[1], color='red', marker='x')
plt.xlabel('area_A')
plt.ylabel('kernel_width')


# In[59]:


# Mahalanobis Distance, raw_dataset
closest10_x_raw_ma_r = raw_dataset.iloc[closest_10_ma_r.index]['area_A']
closest10_y_raw_ma_r = raw_dataset.iloc[closest_10_ma_r.index]['kernel_width']
plt.scatter(x_raw, y_raw)
plt.scatter(closest10_x_raw_ma_r, closest10_y_raw_ma_r, color='yellow')
plt.scatter(P_raw[0], P_raw[1], color='red', marker='x')
plt.xlabel('area_A')
plt.ylabel('kernel_width')


# In[60]:


# Mahalanobis Distance, normalized_dataset
closest10_x_norm_ma_n = normalized_dataset.iloc[closest_10_ma_n.index]['area_A']
closest10_y_norm_ma_n = normalized_dataset.iloc[closest_10_ma_n.index]['kernel_width']
plt.scatter(x_norm, y_norm)
plt.scatter(closest10_x_norm_ma_n, closest10_y_norm_ma_n, color='yellow')
plt.scatter(P_norm[0], P_norm[1], color='red', marker='x')
plt.xlabel('area_A')
plt.ylabel('kernel_width')


# In[61]:


# Mahalanobis Distance, normalized_dataset
closest10_x_std_ma_s = standardized_dataset.iloc[closest_10_ma_s.index]['area_A']
closest10_y_std_ma_s = standardized_dataset.iloc[closest_10_ma_s.index]['kernel_width']
plt.scatter(x_std, y_std)
plt.scatter(closest10_x_std_ma_s, closest10_y_std_ma_s, color='yellow')
plt.scatter(P_std[0], P_std[1], color='red', marker='x')
plt.xlabel('area_A')
plt.ylabel('kernel_width')


# In[62]:


# City Block Metric, raw_dataset
closest10_x_raw_cb_r = raw_dataset.iloc[closest_10_cb_r.index]['area_A']
closest10_y_raw_cb_r = raw_dataset.iloc[closest_10_cb_r.index]['kernel_width']
plt.scatter(x_raw, y_raw)
plt.scatter(closest10_x_raw_cb_r, closest10_y_raw_cb_r, color='yellow')
plt.scatter(P_raw[0], P_raw[1], color='red', marker='x')
plt.xlabel('area_A')
plt.ylabel('kernel_width')


# In[63]:


# City Block Metric, normalized_dataset
closest10_x_norm_cb_n = normalized_dataset.iloc[closest_10_cb_n.index]['area_A']
closest10_y_norm_cb_n = normalized_dataset.iloc[closest_10_cb_n.index]['kernel_width']
plt.scatter(x_norm, y_norm)
plt.scatter(closest10_x_norm_cb_n, closest10_y_norm_cb_n, color='yellow')
plt.scatter(P_norm[0], P_norm[1], color='red', marker='x')
plt.xlabel('area_A')
plt.ylabel('kernel_width')


# In[64]:


# City Block Metric, standardized_dataset
closest10_x_std_cb_s = standardized_dataset.iloc[closest_10_cb_s.index]['area_A']
closest10_y_std_cb_s = standardized_dataset.iloc[closest_10_cb_s.index]['kernel_width']
plt.scatter(x_std, y_std)
plt.scatter(closest10_x_std_cb_s, closest10_y_std_cb_s, color='yellow')
plt.scatter(P_std[0], P_std[1], color='red', marker='x')
plt.xlabel('area_A')
plt.ylabel('kernel_width')


# In[65]:


# Minkowski metric, raw_dataset
closest10_x_raw_mi_raw = raw_dataset.iloc[closest_indices_mi_raw]['area_A']
closest10_y_raw_mi_raw = raw_dataset.iloc[closest_indices_mi_raw]['kernel_width']
plt.scatter(x_raw, y_raw)
plt.scatter(closest10_x_raw_mi_raw, closest10_y_raw_mi_raw, color='yellow')
plt.scatter(P_raw[0], P_raw[1], color='red', marker='x')
plt.xlabel('area_A')
plt.ylabel('kernel_width')


# In[66]:


# Minkowski metric, normalized_dataset
closest10_x_norm_mi_norm = normalized_dataset.iloc[closest_indices_mi_norm]['area_A']
closest10_y_norm_mi_norm = normalized_dataset.iloc[closest_indices_mi_norm]['kernel_width']
plt.scatter(x_norm, y_norm)
plt.scatter(closest10_x_norm_mi_norm, closest10_y_norm_mi_norm, color='yellow')
plt.scatter(P_norm[0], P_norm[1], color='red', marker='x')
plt.xlabel('area_A')
plt.ylabel('kernel_width')


# In[67]:


# Minkowski metric, standardized_dataset
closest10_x_std_mi_std = standardized_dataset.iloc[closest_indices_mi_std]['area_A']
closest10_y_std_mi_std = standardized_dataset.iloc[closest_indices_mi_std]['kernel_width']
plt.scatter(x_std, y_std)
plt.scatter(closest10_x_std_mi_std, closest10_y_std_mi_std, color='yellow')
plt.scatter(P_std[0], P_std[1], color='red', marker='x')
plt.xlabel('area_A')
plt.ylabel('kernel_width')


# In[68]:


# Chebyshev distance, raw_dataset
closest10_x_raw_ch_r = raw_dataset.iloc[closest_10_ch_r.index]['area_A']
closest10_y_raw_ch_r = raw_dataset.iloc[closest_10_ch_r.index]['kernel_width']
plt.scatter(x_raw, y_raw)
plt.scatter(closest10_x_raw_ch_r, closest10_y_raw_ch_r, color='yellow')
plt.scatter(P_raw[0], P_raw[1], color='red', marker='x')
plt.xlabel('area_A')
plt.ylabel('kernel_width')


# In[69]:


# Chebyshev distance, normalized_dataset
closest10_x_norm_ch_n = normalized_dataset.iloc[closest_10_ch_n.index]['area_A']
closest10_y_norm_ch_n = normalized_dataset.iloc[closest_10_ch_n.index]['kernel_width']
plt.scatter(x_norm, y_norm)
plt.scatter(closest10_x_norm_ch_n, closest10_y_norm_ch_n, color='yellow')
plt.scatter(P_norm[0], P_norm[1], color='red', marker='x')
plt.xlabel('area_A')
plt.ylabel('kernel_width')


# In[70]:


# Chebyshev distance, standardized_dataset
closest10_x_std_ch_s = standardized_dataset.iloc[closest_10_ch_s.index]['area_A']
closest10_y_std_ch_s = standardized_dataset.iloc[closest_10_ch_s.index]['kernel_width']
plt.scatter(x_std, y_std)
plt.scatter(closest10_x_std_ch_s, closest10_y_std_ch_s, color='yellow')
plt.scatter(P_std[0], P_std[1], color='red', marker='x')
plt.xlabel('area_A')
plt.ylabel('kernel_width')


# In[71]:


# Cosine distance, raw_dataset
closest10_x_raw_cos_r = raw_dataset.iloc[closest_10_cos_r.index]['area_A']
closest10_y_raw_cos_r = raw_dataset.iloc[closest_10_cos_r.index]['kernel_width']
plt.scatter(x_raw, y_raw)
plt.scatter(closest10_x_raw_cos_r, closest10_y_raw_cos_r, color='yellow')
plt.scatter(P_raw[0], P_raw[1], color='red', marker='x')
plt.xlabel('area_A')
plt.ylabel('kernel_width')


# In[72]:


# Cosine distance, normalized_dataset
closest10_x_norm_cos_n = normalized_dataset.iloc[closest_10_cos_n.index]['area_A']
closest10_y_norm_cos_n = normalized_dataset.iloc[closest_10_cos_n.index]['kernel_width']
plt.scatter(x_norm, y_norm)
plt.scatter(closest10_x_norm_cos_n, closest10_y_norm_cos_n, color='yellow')
plt.scatter(P_norm[0], P_norm[1], color='red', marker='x')
plt.xlabel('area_A')
plt.ylabel('kernel_width')


# In[73]:


# Cosine distance, standardized_dataset
closest10_x_std_cos_s = standardized_dataset.iloc[closest_10_cos_s.index]['area_A']
closest10_y_std_cos_s = standardized_dataset.iloc[closest_10_cos_s.index]['kernel_width']
plt.scatter(x_std, y_std)
plt.scatter(closest10_x_std_cos_s, closest10_y_std_cos_s, color='yellow')
plt.scatter(P_std[0], P_std[1], color='red', marker='x')
plt.xlabel('area_A')
plt.ylabel('kernel_width')


# In[74]:


# Canberra distance, raw_dataset
closest10_x_raw_ca_r = raw_dataset.iloc[closest_10_ca_r.index]['area_A']
closest10_y_raw_ca_r = raw_dataset.iloc[closest_10_ca_r.index]['kernel_width']
plt.scatter(x_raw, y_raw)
plt.scatter(closest10_x_raw_ca_r, closest10_y_raw_ca_r, color='yellow')
plt.scatter(P_raw[0], P_raw[1], color='red', marker='x')
plt.xlabel('area_A')
plt.ylabel('kernel_width')


# In[75]:


# Canberra distance, normalized_dataset
closest10_x_norm_ca_n = normalized_dataset.iloc[closest_10_ca_n.index]['area_A']
closest10_y_norm_ca_n = normalized_dataset.iloc[closest_10_ca_n.index]['kernel_width']
plt.scatter(x_norm, y_norm)
plt.scatter(closest10_x_norm_ca_n, closest10_y_norm_ca_n, color='yellow')
plt.scatter(P_norm[0], P_norm[1], color='red', marker='x')
plt.xlabel('area_A')
plt.ylabel('kernel_width')


# In[76]:


# Canberra distance, standardized_dataset
closest10_x_std_ca_s = standardized_dataset.iloc[closest_10_ca_s.index]['area_A']
closest10_y_std_ca_s = standardized_dataset.iloc[closest_10_ca_s.index]['kernel_width']
plt.scatter(x_std, y_std)
plt.scatter(closest10_x_std_ca_s, closest10_y_std_ca_s, color='yellow')
plt.scatter(P_std[0], P_std[1], color='red', marker='x')
plt.xlabel('area_A')
plt.ylabel('kernel_width')

