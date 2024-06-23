import praw
import json

# Define your Reddit API credentials
client_id = 'qtn3PRoAXu1o4tZ8BEsmTw'
client_secret = 'KjN6Tsk5_Oa-n3BCdH-eziyRDpB2Yw'
user_agent = 'my_reddit_app by /u/First_Acanthaceae_23'

# Initialize the Reddit instance
reddit = praw.Reddit(client_id=client_id,
                     client_secret=client_secret,
                     user_agent=user_agent)

# Define the subreddit
subreddit_name = 'Forex'

# Fetch comments from the subreddit
comments = []
subreddit = reddit.subreddit(subreddit_name)
for comment in subreddit.comments(limit=10000):  # Increase limit if needed
    if 'strategy' in comment.body.lower():  # Check if 'strategy' is in the comment body
        comments.append({
            'author': comment.author.name,
            'body': comment.body,
            'created_utc': comment.created_utc
        })

# Print out the comments that mention 'strategy'
for comment in comments:
    author = comment['author']
    body = comment['body']
    created_utc = comment['created_utc']
    print(f"Author: {author}\nComment: {body}\nCreated UTC: {created_utc}\n")

# Optionally, save the comments to a JSON file
with open("forex_strategy_comments.json", "w") as f:
    json.dump(comments, f, indent=4)
