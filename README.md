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
netflix_titles_df.info()
netflix_titles_df.nunique()
```
<img src="https://github.com/amalsebastian7/Data-science/blob/main/ALL_RESOURCES/screenshots/Screenshot%202021-09-03%20104840.jpg" width="400" height="400" />


```python
sns.heatmap(netflix_titles_df.isnull(), cbar=True)
plt.title('Null Values Heatmap')
plt.show()
```
<img src="https://github.com/amalsebastian7/Data-science/blob/main/ALL_RESOURCES/screenshots/heatmap.jpg" width="400" height="400" />



```python
netflix_titles_df['director'].fillna('No Director', inplace=True)
netflix_titles_df['cast'].fillna('No Cast', inplace=True)
netflix_titles_df['country'].fillna('Country Unavailable', inplace=True)
netflix_titles_df.dropna(subset=['date_added','rating'],inplace=True)
netflix_titles_df.isnull().any()
netflix_titles_df.isnull().any()
```
<img src="https://github.com/amalsebastian7/Data-science/blob/main/ALL_RESOURCES/screenshots/false.jpg" width="300" height="300" />

```python
netflix_movies_df = netflix_titles_df[netflix_titles_df['type']=='Movie'].copy()
netflix_movies_df.head()
netflix_shows_df = netflix_titles_df[netflix_titles_df['type']=='TV Show'].copy()
netflix_shows_df.head()
netflix_movies_df.duration = netflix_movies_df.duration.str.replace(' min','').astype(int)
netflix_shows_df.rename(columns={'duration':'seasons'}, inplace=True)
netflix_shows_df.replace({'seasons':{'1 Season':'1 Seasons'}}, inplace=True)
netflix_shows_df.seasons = netflix_shows_df.seasons.str.replace(' Seasons','').astype(int)
netflix_movies_df.head()
```
<img src="https://github.com/amalsebastian7/Data-science/blob/main/movies.jpg" width="800" height="400" />


```python
netflix_shows_df.head()
```
<img src="https://github.com/amalsebastian7/Data-science/blob/main/series.jpg" width="800" height="400" />



```python
plt.figure(figsize=(3,6))
g = sns.countplot(netflix_titles_df.type, palette="pastel");
plt.title("Count of Movies and TV Shows")
plt.xlabel("Type (Movie/TV Show)")
plt.ylabel("Total Count")
plt.show()
```
<img src="https://github.com/amalsebastian7/Data-science/blob/main/ALL_RESOURCES/screenshots/bar.jpg" width="300" height="400" />



```python
plt.figure(figsize=(12,6))
plt.title("% of Netflix Titles that are either Movies or TV Shows")
g = plt.pie(netflix_titles_df.type.value_counts(), explode=(0.025,0.025), labels=netflix_titles_df.type.value_counts().index, colors=['skyblue','navajowhite'],autopct='%1.1f%%', startangle=180);
plt.legend()
plt.show()
```

<img src="https://github.com/amalsebastian7/Data-science/blob/main/ALL_RESOURCES/screenshots/pie.jpg" width="500" height="500" />


```python
order =  ['G', 'TV-Y', 'TV-G', 'PG', 'TV-Y7', 'TV-Y7-FV', 'TV-PG', 'PG-13', 'TV-14', 'R', 'NC-17', 'TV-MA']
plt.figure(figsize=(15,7))
g = sns.countplot(netflix_titles_df.rating, hue=netflix_titles_df.type, order=order, palette="pastel");
plt.title("Ratings for Movies & TV Shows")
plt.xlabel("Rating")
plt.ylabel("Total Count")
plt.show()
```
<img src="https://github.com/amalsebastian7/Data-science/blob/main/ALL_RESOURCES/screenshots/rating.jpg" width="800" height="500" />



```python
fig, ax = plt.subplots(1,2, figsize=(19, 5))
g1 = sns.countplot(netflix_movies_df.rating, order=order,palette="Set2", ax=ax[0]);
g1.set_title("Ratings for Movies")
g1.set_xlabel("Rating")
g1.set_ylabel("Total Count")
g2 = sns.countplot(netflix_shows_df.rating, order=order,palette="Set2", ax=ax[1]);
g2.set(yticks=np.arange(0,1600,200))
g2.set_title("Ratings for TV Shows")
g2.set_xlabel("Rating")
g2.set_ylabel("Total Count")
fig.show()
```
<img src="https://github.com/amalsebastian7/Data-science/blob/main/ALL_RESOURCES/screenshots/ratingforboth.jpg"  />




```python
netflix_titles_df['year_added'] = pd.DatetimeIndex(netflix_titles_df['date_added']).year
netflix_movies_df['year_added'] = pd.DatetimeIndex(netflix_movies_df['date_added']).year
netflix_shows_df['year_added'] = pd.DatetimeIndex(netflix_shows_df['date_added']).year
netflix_titles_df['month_added'] = pd.DatetimeIndex(netflix_titles_df['date_added']).month
netflix_movies_df['month_added'] = pd.DatetimeIndex(netflix_movies_df['date_added']).month
netflix_shows_df['month_added'] = pd.DatetimeIndex(netflix_shows_df['date_added']).month

netflix_year = netflix_titles_df['year_added'].value_counts().to_frame().reset_index().rename(columns={'index': 'year','year_added':'count'})
netflix_year = netflix_year[netflix_year.year != 2022]
netflix_year
```
<img src="https://github.com/amalsebastian7/Data-science/blob/main/ALL_RESOURCES/screenshots/year.jpg"  />


```python
netflix_year2 = netflix_titles_df[['type','year_added']]
movie_year = netflix_year2[netflix_year2['type']=='Movie'].year_added.value_counts().to_frame().reset_index().rename(columns={'index': 'year','year_added':'count'})
movie_year = movie_year[movie_year.year != 2022]
show_year = netflix_year2[netflix_year2['type']=='TV Show'].year_added.value_counts().to_frame().reset_index().rename(columns={'index': 'year','year_added':'count'})
show_year = show_year[show_year.year != 2022]

fig, ax = plt.subplots(figsize=(10, 6))
sns.lineplot(data=netflix_year, x='year', y='count')
sns.lineplot(data=movie_year, x='year', y='count')
sns.lineplot(data=show_year, x='year', y='count')
ax.set_xticks(np.arange(2008, 2022, 1))
plt.title("Total content added each year (up to 2019)")
plt.legend(['Total','Movie','TV Show'])
plt.ylabel("Releases")
plt.xlabel("Year")
plt.show()

```
<img src="https://github.com/amalsebastian7/Data-science/blob/main/ALL_RESOURCES/screenshots/contentyearwise.jpg"  />


```python
month_year_df = netflix_titles_df.groupby('year_added')['month_added'].value_counts().unstack().fillna(0).T

plt.figure(figsize=(11,8))
sns.heatmap(month_year_df, linewidths=0.025, cmap="YlGnBu")
plt.title("Content Heatmap")
plt.ylabel("Month")
plt.xlabel("Year")
plt.show()
```
<img src="https://github.com/amalsebastian7/Data-science/blob/main/ALL_RESOURCES/screenshots/contentheatmap.jpg"  />


```python
fig, ax = plt.subplots(1,2, figsize=(19, 5))
g1 = sns.distplot(netflix_movies_df.duration, color='skyblue',ax=ax[0]);
g1.set_xticks(np.arange(0,360,30))
g1.set_title("Duration Distribution for Netflix Movies")
g1.set_ylabel("% of All Netflix Movies")
g1.set_xlabel("Duration (minutes)")
g2 = sns.countplot(netflix_shows_df.seasons, color='skyblue',ax=ax[1]);
g2.set_title("Netflix TV Shows Seasons")
g2.set_ylabel("Count")
g2.set_xlabel("Season(s)")
fig.show()
```

<img src="https://github.com/amalsebastian7/Data-science/blob/main/ALL_RESOURCES/screenshots/duration.jpg"  />

```python
filtered_countries = netflix_titles_df.set_index('title').country.str.split(', ', expand=True).stack().reset_index(level=1, drop=True);
filtered_countries = filtered_countries[filtered_countries != 'Country Unavailable']

plt.figure(figsize=(7,9))
g = sns.countplot(y = filtered_countries, order=filtered_countries.value_counts().index[:20])
plt.title('Top 20 Countries on Netflix')
plt.xlabel('Titles')
plt.ylabel('Country')
plt.show()
```
<img src="https://github.com/amalsebastian7/Data-science/blob/main/ALL_RESOURCES/screenshots/country.jpg"  />


```python
filtered_genres = netflix_titles_df.set_index('title').listed_in.str.split(', ', expand=True).stack().reset_index(level=1, drop=True);

plt.figure(figsize=(7,9))
g = sns.countplot(y = filtered_genres, order=filtered_genres.value_counts().index[:20])
plt.title('Top 20 Genres on Netflix')
plt.xlabel('Titles')
plt.ylabel('Genres')
plt.show()
```
<img src="https://github.com/amalsebastian7/Data-science/blob/main/ALL_RESOURCES/screenshots/genre.jpg"  />







