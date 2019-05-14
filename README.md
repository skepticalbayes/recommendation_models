# recommendation_models
Multiple recommendation models implemented in tensorflow/torch.


## Collaborative filtering for recommendation systems
The collaborative filtering technique is a powerful method for generating user recommendations. Collaborative filtering relies only on observed user behavior to make recommendations—no profile data or content access is necessary.

The technique is based on the following observations:

Users who interact with items in a similar manner (for example, buying the same products or viewing the same articles) share one or more hidden preferences.
Users with shared preferences are likely to respond in the same way to the same items.
Combining these basic observations allows a recommendation engine to function without needing to determine the precise nature of the shared user preferences. All that's required is that the preferences exist and are meaningful. The basic assumption is that similar user behavior reflects similar fundamental preferences, allowing a recommendation engine to make suggestions accordingly.

For example, suppose User 1 has viewed items A, B, C, D, E, and F. User 2 has viewed items A, B, D, E and F, but not C. Because both users viewed five of the same six items, it's likely that they share some basic preferences. User 1 liked item C, and it's probable that User 2 would also like item C if the user were aware of its existence. This is where the recommendation engine steps in: it informs User 2 about item C, piquing that user's interest.

## Matrix factorization for collaborative filtering
The collaborative filtering problem can be solved using matrix factorization. Suppose you have a matrix consisting of user IDs and their interactions with your products. Each row corresponds to a unique user, and each column corresponds to an item. The item could be an product in a catalog, an article, or a video. Each entry in the matrix captures a user's rating or preference for a single item. The rating could be explicit, directly generated by user feedback, or it could be implicit, based on user purchases or time spent interacting with an article or video.

If a user has never rated an item or shown any implied interest in it, the matrix entry is zero. Figure 1 shows a representation of a MovieLens rating matrix.
