import praw
reddit = praw.Reddit(client_id = '####',
                    client_secret = '####',
                    username = '####,
                    password = '####',
                    user_agent = '####')

appended_data = []

subreddit = reddit.subreddit('HFY')  

top_python = subreddit.top(limit=15)     
for submission in top_python:
    if not submission.stickied:
        appended_data.append(submission.selftext)

print('Fetching complete')
file = open('results.txt', 'w')
for item in appended_data:
    file.write("%s\n" % item)
print('Done')
