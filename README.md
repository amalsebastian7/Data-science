# <img src="https://www.teahub.io/photos/full/88-885793_data-science.jpg" width="1000" height="500" />
Data science Projects And works.
Data science is the field of study that combines domain expertise, programming skills, and knowledge of mathematics and statistics to extract meaningful insights from data. Data science practitioners apply machine learning algorithms to numbers, text, images, video, audio, and more to produce artificial intelligence (AI) systems to perform tasks that ordinarily require human intelligence. In turn, these systems generate insights which analysts and business users can translate into tangible business value.
# Projects

- [Netflix](#netflix)


## Netflix
### Steps
- Load all required Libraries

```python
 import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
import matplotlib
```
- Load the netflix.csv file
     - https://github.com/amalsebastian7/Data-science/blob/main/ALL_RESOURCES/netflix_titles.csv

```python
netflix_titles_df = pd.read_csv('../input/netflix-shows/netflix_titles.csv')
netflix_titles_df.head()
```
<img src="https://github.com/amalsebastian7/Data-science/blob/main/ALL_RESOURCES/screenshots/netflix.jpg" width="900" height="400" />

-  To see the comparison between the total number of movies and shows in this dataset just to get an idea of which one is the majority.
```python
plt.figure(figsize=(3,6))
g = sns.countplot(netflix_titles_df.type, palette="pastel");
plt.title("Count of Movies and TV Shows")
plt.xlabel("Type (Movie/TV Show)")
plt.ylabel("Total Count")
plt.show()

```
<img src="https://github.com/amalsebastian7/Data-science/blob/main/ALL_RESOURCES/Screenshot2.jpg" width="900" height="400" />




```python

```








```python

```



```python

```








