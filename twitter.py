#!/usr/bin/env python
# encoding: utf-8

import tweepy

#Twitter API
consumer_key = "####"
consumer_secret = "####"
access_key = "####"
access_secret = "####"


def get_all_tweets(screen_name):
	#Twitter only allows access to a users most recent 3240 tweets with this method
	
	auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
	auth.set_access_token(access_key, access_secret)
	api = tweepy.API(auth)
	
	alltweets = []	
	
	#zahtev za najnovije tvitove (200 je maksimalni dozvoljen broj)
	new_tweets = api.user_timeline(screen_name = screen_name,count=200)
	
	#cuva te najnovije tvitove u listu
	alltweets.extend(new_tweets)
	
	#id najstarijeg tvita - 1 
	oldest = alltweets[-1].id - 1
	
	#dokle god moze da preuzima tvitove, preuzima
	while len(new_tweets) > 0:
		print("getting tweets before %s" % (oldest))
		
		#all subsiquent requests use the max_id param to prevent duplicates
		new_tweets = api.user_timeline(screen_name = screen_name,count=200,max_id=oldest)
		
		alltweets.extend(new_tweets)
		
		#update za id najstarijeg tvita - 1
		oldest = alltweets[-1].id - 1
		
		print("...%s tweets downloaded so far" % (len(alltweets)))
	
	#outtweets = [[tweet.id_str, tweet.created_at, tweet.text.encode("utf-8")] for tweet in alltweets]
	outtweets = [tweet.text.encode("utf-8") for tweet in alltweets]
	
	with open('%s_tweets.txt' % screen_name, 'w+') as f:
		for item in outtweets:
			if b"@" not in item and b"RT" not in item and b"http" not in item:
				f.write("%s\n" % item)

if __name__ == '__main__':
	#username profila po izboru
	get_all_tweets("realDonaldTrump")
