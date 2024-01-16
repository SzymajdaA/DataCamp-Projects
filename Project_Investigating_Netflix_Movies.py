# Importing pandas and matplotlib

import pandas as pd
import matplotlib.pyplot as plt

# Start coding!
netflix_df = pd.read_csv('netflix_data.csv')
print(netflix_df)

#subseting the movies

netflix_subset = netflix_df[netflix_df.type != 'TV Show']
print(netflix_subset)

#subseting the columns

netflix_movies = netflix_subset[['title', 'country', 'genre', 'release_year', 'duration']]
list(netflix_movies.columns)
print(netflix_movies)

#choosing the short movies

short_movies = netflix_movies[netflix_movies['duration']<60]
print(short_movies)

#adding colors to the movies

colors = []
for index, row in netflix_movies.iterrows():
    if row['genre'] == 'Children':
        colors.append('green')
    elif row['genre'] == 'Documentaries':
        colors.append('red')
    elif row['genre'] == 'Stand-Up':
        colors.append('black')
    else:
        colors.append('blue')
        
#visualisation

fig = plt.figure()
plt.xlabel("Release year")
plt.ylabel("Duration (min)")
plt.title("Movie Duration by Year of Release")
plt.scatter(netflix_movies['release_year'], netflix_movies['duration'], c=colors)
plt.show()

#answer to the question

answer = 'maybe'