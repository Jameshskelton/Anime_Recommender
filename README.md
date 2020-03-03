# Kintoun
Kintoun is an anime recommender compiled using the myanimelist dataset. 
It uses matrix factorization techniques to generate predicted ratings on unseen anime from previously given ratings on seen anime. 

I chose to conduct this analysis because I did not want to sit in front of my computer for hours at a time looking for something to watch. Furthermore, I had noticed there was a dearth of functionally useful anime recommender's available to the public

I opted to use Azathoth42's Kaggle myanimelist dataset to provide the reviews for this analysis (https://www.kaggle.com/azathoth42/myanimelist)

## EDA

The total amount of reviews in the the corpus was 31,284,030. The average rating by a user was 4.65, and the average anime rating was 6.14. The reviews date from 2006-09-27 to 2018-05-22. Here are some histograms for the dataset: 

[img]

[img]

[img]

As you can see in these histograms, there were a massive amount of reviews with a score of 0 compared to the others, comprising around a third of the original dataset. Most of the measures taken to determine which data was passed to the model were done to deal with this unusual data characteristic

Furthermore, while there exist 'power reviewers' (the most reviews left by any one user is 6, and the most reviewed anime had 81,331), the vast majority of anime received less than 100 reviews, and the vast majority of users left less than 50 reviews. 


## Cleaning & Analysis 

After performing EDA, a number of cleaning actions were made to reduce the high amount of reviews with a zero rating. 

The vast majority of these reviews with a zero rating were made by users who indicated they had an incomplete viewing relationship with the anime. The 'my_status' feature, which ranges from 1-6, was used to determine which users were to be removed. Only users which had a 1, meaning they were watching, or a 2, meaning they had completed the anime, were retained. After that, any review that had teh 'my_watched_episodes' feature at less than 3 were also removed. This was a personal decision made because i believe that these user's could not have generated a fully opinionated review with only that much of the show watched. 
Users with less than 6 reviews were dropped. Anime with less than 6 reviews were also dropped. This functionally removed several of the problems associated with the 'cold start', where a recommender cannot make predictions due to insufficient data. 

In the end, this reduced the zero count to 16.507% of the original count. The final data set consisted of 15,546,712 reviews.

## Modeling
____________________________________________________________________________________

PySpark's Alternating Least Squares matrix factorization model was applied for this analysis. A custom script using this model was made that would first generate the predictions it was able to using matrix factorization, then loop back through to generate average values for each anime using the given scores, and then a final loop to fill the missing values from loop one with the second loop values (see src/recommender_anime.py).

## Results

The model was able to achieve a peak Root Mean Squared Error (RMSE) of 1.614. 
Here is a histogram showing the distribution of the counts of the actual versus predicted anime ratings:

[img]

As you can see, the model was able to succesfully predict much of the more middle range of scores, but struggled with the ends of the scoring range. In particular, it was still unable to capture the high amount of present 0 ratings. It was also unable to model the 8-10 range of ratings as well as the rest of the scores. 

In the end, i believe the failures of this model are largely related to how missing values were filled. Using the average rating for each anime works well as a stopgap for a better method, but it is evident from this histogram that this method does not address the problem very well. 


## Next Steps and Future Work


The first major next step for this project would be to build a second recommender system using the same dataset. There 56 features in total that could have been used, so i would like to generate a content-based recommender using these features. This new model could then be hybridized with my current model to yield, in theory, a more predictive model than either on their own. 

After that work is completed, I have reached out to developers about creating a web application for the recommender. Please look here for updates on the project


Thank you for taking the time to read about this project! Please reach out to me at my linkedin if my other work is of interest to you, i would love to chat about it!


## Acknowledgements

Thanks to Dan Rupp, Dr. Joe Gartner, and Brent Goldberg for their guidance during this project. I would have missed so many steps without your assistance along the way. 

Thanks to Kaggle user Azerthoth42 for creating the awesome dataset. While i have experience building scrapers, this saved me a seriously notable amount of time and effort. 

Thanks to Kaggle user Martelloti for his work on building his own recommender system using the surprise library. It was very informative to read about how he approached a similar problem using different cleaning techniques and modeling technologies, and it gave me a baseline to compare my model against. https://www.kaggle.com/martelloti/myanimelist-recommender-system


