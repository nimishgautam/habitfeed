# INTRO

We use some standard techniques to try and recommend an article to the user.

## Part 1: Reading in articles

Various feeds exist as Feed objects (db/models.py). When called, services/feed_services will read those feeds and populate Article objects with a bit of data from the feed, notably the Article's full URL (NOTE: the title is deliberately not populated here as a flag indicating "this article needs to be hydrated later")

After this, services/article_services then uses Goose3 to read in information about the article, most notably the title and a bare-text version of the article (the main feature of Goose3)

Finally, topics are assigned via bertopic, also done in services/article_services.

## Part 2: Embedding similarity

After the fulltext of the article is read in, services/article_services also calculates the embedding from a sentence-transformer for each article and stores it.

### User embeddings

There is a "positive" embedding and "negative" embedding for the user. The "positive" embedding is supposed to represent all the articles the user has interacted positively with. The "negative" embedding is supposed to represent all the articles the user has interacted negatively with.

We use the positive and negative embeddings, compare it with the article embeddings and generate a "vector_similarity_score" for the article. This is the first part of the article score that we ultimately use for recommendation.

#### Weights of events

We will weigh events in the following way for the positive vector:
click  = 0.5 · e-Δt/τ
upvote  = 2 · e-Δt/τ
save  = 3 · e-Δt/τ
(if dwell and action_duration_ms ≥ 30 000) = +1 bonus

And for the negative vector:
meh  = 1 · e-Δt/τ
skip  = 2 · e-Δt/τ
downvote  = 3 · e-Δt/τ

Δt = now − event_ts
τ (half-life) ≈ 14 days keeps the profile fresh.

When calculating the user vectors, we will only consider the articles where these interactions have happened in the last 90 days.

#### Scoring articles

embedding_services/calculate_article_embedding_weights() calculates the scores for all articles and stores them in the vector_similarity_score field

The score is:
score = positive_similarity - α * negative_similarity
with α = 0.7

Note: cosine_similarity() is a helper fn, as is calculate_article_similarity_score and update_single_article_score

### Usage notes
Process all articles
count = calculate_article_embedding_weights(db)

Process specific articles
count = calculate_article_embedding_weights(db, article_ids=[1, 2, 3])

Update single article (for real-time updates after user interactions)
success = update_single_article_score(db, article_id=123)

We store the numerator and denominator for the user vectors seperately so that we can do incremental updates if needed (rather than recalculating x times a day).

Those are done from the function supdate_profile_vectors_incremental
and update_profile_and_article_score_on_interaction

### Sanity checks
We have a convenience function calculate_scores_for_recent_uninteracted_articles() that only calculates scores on articles the user hasn't interacted with (so you don't get articles scored that you have already seen)

and mark_interacted_articles_low_score() to set a negative score (-2 when values are between +1 and -1 normally) so you don't get articles recommended that you have already seen

## Part 3: LightFM with Warp (DEFERRED FOR NOW)

The next stage would be to use LightFM with WARP to take the top articles based on vector similarity and re-rank them.

HOWEVER, LightFM with WARP only really shines when we have social signals (upvotes, other users liking, etc.)

At some point, this may make sense (especially with reddit, hackernews etc.)

Some signals for now:
topic, vector score, source

Addiotional proposed signals:
# Temporal patterns
- hour_of_day (when user typically engages positively)
- day_of_week  
- time_since_publication (recency preference)

# Feed characteristics  
- feed_popularity (articles_per_day)
- feed_domain (news vs blog vs academic)
- your_interaction_rate_with_feed (personalized feed preference)

# Content features
- article_length_bucket (short/medium/long)
- topic_diversity_score (how many topics vs focused)
- reading_time_estimate

# Interaction history features
- previous_topic_before_positive_action
- session_context (what you read before this)

# Embedding-derived features
- embedding_cluster_id (k-means on embeddings)
- similarity_to_recent_positive_articles
- topic_confidence_distribution (full BERTopic probabilities)

# Behavioral features  
- typical_dwell_time_for_similar_articles
- interaction_velocity (how quickly you act on articles)

## Part 4: LinUCB ('bandit') layer

This is to add some variance as well to the final recommendations