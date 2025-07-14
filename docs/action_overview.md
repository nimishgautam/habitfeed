# Actions

We have different actions with different meaning. 

## In the code
The master list of actions is in db/models.py ActionEnum.
The weights of actions for recommendations in services/embedding_services
Some of the business logic of when to record and not record actions (or convert them etc) is in the api/main.py


## What they mean

### Potential signal actions:

* click - when the user clicks on an article (ie opens it to read it from the title on the sidebar or clicks to read it in the general feed)
 * This is seen as a weak positive signal. When a clearly negative signal is sent though, then clicks are converted to "click_archive" so as not to be used for recommendation

* dwell - when the user spends some time hovering over the article but doesn't click it
 * This is also seen as a weak positive signal. Unlike clicks, this one is not archived if there's a negative signal because it captures some interest still (up for debate if this is a sensible interpretation or not)

 ### Clear positive actions:

 Articles with clear positive actions are never deleted.
 However, their recommendation scores are set arbitrarily low so they won't be recommended again (the user knows about them already)

 * upvote - just an indication of "I like this"

 * save - when the article is saved/bookmarked for easier viewing later. This is seen as a stronger positive signal than just upvoting

 ### Clear negative actions:

Articles with clear negative actions will likely be deleted after some time. Their recommendation scores are also set arbitrarily low so the user doesn't see them again

 * meh - a weak negative interaction. The user just feels 'meh' about the article. 

 * skip - a medium negative interaction. The user is saying 'not bad, but a little of a waste of time'

 * downvote - a strong negative interaction. "I did not like this"

 ### Misc/admin actions:

 * click_archive - we may want to know that the user did click, even if the interaction was negative (in fact, it's very likely they clicked to read through the article in order to make the negative judgement) but since 'click' is a positive signal, we want an event that won't be counted towards positive signals, so we have this