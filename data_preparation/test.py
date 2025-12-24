import asyncio
from twscrape import API, gather
from twscrape.logger import set_log_level

async def main():
    api = API()
    
    # Add your accounts if you haven't already
    # await api.pool.add_account("username", "password", "email", "email_password")
    # await api.pool.login_all()

    target_user_id = "dogeai_gov"               # Replace with your commenter's ID
    
    # limit=50 fetches the last 50 interactions (tweets + replies)
    # This is usually enough to spot a pattern.
    q = api.user_tweets_and_replies(target_user_id, limit=50)
    
    comments = []
    async for tweet in q:
        # We filter for replies because you specifically want "comments"
        if tweet.inReplyToTweetId: 
            comments.append(tweet)
            print(f"[{tweet.date}] Replied to {tweet.inReplyToUser.username}: {tweet.rawContent}")

    print(f"Found {len(comments)} replies.")

if __name__ == "__main__":
    asyncio.run(main())