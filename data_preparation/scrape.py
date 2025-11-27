import asyncio
import json
import os
from twscrape import API, gather
from twscrape.logger import set_log_level
import time
from pathlib import Path

# List of influencers from your existing file
INFLUENCE_TWEET_USERNAME = ["AdamParkhomenko", "ArielleScarcell", "AstroTerry", "bariweiss", "blackintheempir", "bresreports", "brithume", "burgessev", "ByYourLogic", "CarlHigbie", "CensoredMen", "ChrisLoesch", "CollinRugg", "Daminous_Purity", "danpfeiffer", "danprimack", "davetroy", "dlacalle_IA", "DrLoupis", "ejmalrai", "eveforamerica", "feliciasonmez", "FiorellaIsabelM", "FrankDangelo23", "garethicke", "Gdad1", "GregRubini", "HilzFuld", "IamBrookJackson", "jacksonhinkle", "jayrosen_nyu", "JillFillipovic", "jimsciutto", "JoeConchaTV", "JonahDispatch", "JonathanTurley", "JoshDenny", "kacdnp91", "KatiePhang", "KeneAkers", "kristina_wong", "kyledcheney", "laurashin", "LEBassett", "LeftAtLondon", "Leslieoo7", "LibertyCappy", "MarchandSurgery", "MarkHertling", "MaryLTrump", "MattBruenig", "MelonieMac", "MikeASperrazza", "MikeSington", "MollyOxFFF", "MsAvaArmstrong", "NAChristakis", "nic_carter", "OMGN02Trump", "Prolotario1", "Rach_IC", "RightWingCope", "RogerJStoneJr", "saifedean", "SamParkerSenate", "SarahTheHaider", "secupp", "simon_schama", "StellaParton", "Tatarigami_UA", "thatdayin1992", "TheBigMigShow", "thecoastguy", "thejackhopkins", "TimlnHonolulu", "WBrettWilson", "yarahawari", "ylecun"]

# Create output directory
OUTPUT_DIR = Path("h:/Dev/University/DataScience/Project/data_preparation/influencer_data")
OUTPUT_DIR.mkdir(exist_ok=True)

# async def extract_tweet_data(tweet):
#     """Extract required fields from a tweet object"""
#     return {
#         "username": tweet.user.username,
#         "content": tweet.rawContent,
#         "likes": tweet.likeCount,
#         "retweet": tweet.rQetweetCount,
#         "share": tweet.quoteCount,
#         "views": tweet.viewCount,
#         "replies": tweet.replyCount
#     }
async def extract_tweet_data(tweet):
    """Extract required fields from a tweet object"""
    return {
        "username": tweet.user.username,
        "content": tweet.rawContent,
        "likes": tweet.likeCount,
        "retweet": tweet.retweetCount,
        "share": tweet.quoteCount,
        "views": tweet.viewCount,
        "replies": tweet.replyCount,
        # "date": tweet.date,
        "link": tweet.url
    }

async def process_influencer(api, username):
    try:
        print(f"Processing influencer: {username}")
        
        # Get user by login
        user = await api.user_by_login(username)
        
        # Get 100 most recent tweets
        influencer_data = []
        tweets = []
        
        try:
            tweets = await gather(api.user_tweets(user.id, limit=3))
            print(f"  - Found {len(tweets)} tweets for {username}")
        except Exception as e:
            print(f"  - Error getting tweets for {username}: {e}")
            return []
        
        # Process each tweet
        for i, tweet in enumerate(tweets):
            try:
                print(f"  - Processing tweet {i+1}/{len(tweets)} (id: {tweet.id})")
                tweet_data = await extract_tweet_data(tweet)
                tweet_data["replies_content"] = []
                
                # Get up to 200 earliest comments
                comments = []
                try:
                    comments = await gather(api.tweet_replies(tweet.id, limit=200))
                    print(f"    - Found {len(comments)} comments for tweet {tweet.id}")
                except Exception as e:
                    print(f"    - Error getting comments for tweet {tweet.id}: {e}")
                
                # Process each comment
                for j, comment in enumerate(comments):
                    try:
                        comment_data = await extract_tweet_data(comment)
                        comment_data["replies_content"] = []
                        
                        # Get up to 100 earliest replies to this comment
                        replies = []
                        try:
                            replies = await gather(api.tweet_replies(comment.id, limit=100))
                            print(f"      - Found {len(replies)} replies for comment {comment.id}")
                        except Exception as e:
                            print(f"      - Error getting replies for comment {comment.id}: {e}")
                        
                        # Process each reply
                        for reply in replies:
                            try:
                                reply_data = await extract_tweet_data(reply)
                                comment_data["replies_content"].append(reply_data)
                            except Exception as e:
                                print(f"      - Error processing reply {reply.id}: {e}")
                        
                        tweet_data["replies_content"].append(comment_data)
                    except Exception as e:
                        print(f"    - Error processing comment {comment.id}: {e}")
                
                influencer_data.append(tweet_data)
                
                # Add a small delay to avoid rate limits
                await asyncio.sleep(1)
                
            except Exception as e:
                print(f"  - Error processing tweet {tweet.id}: {e}")
        
        return influencer_data
    
    except Exception as e:
        print(f"Error processing influencer {username}: {e}")
        return []

async def main():
    # Set up API
    api = API()
    
    # Add your authentication here if needed
    # await api.pool.add_account("username", "password", "email", "email_password")
    
    # Set log level to see more details
    set_log_level("INFO")
    
    # Process each influencer
    for username in INFLUENCE_TWEET_USERNAME:
        try:
            # Check if we already processed this influencer
            output_file = OUTPUT_DIR / f"{username}.json"
            if output_file.exists():
                print(f"Skipping {username} - already processed")
                continue
                
            # Process influencer
            influencer_data = await process_influencer(api, username)
            
            # Save data to JSON file
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(influencer_data, f, ensure_ascii=False, indent=2)
                
            print(f"Completed processing for {username}")
            
            # Add delay between influencers to avoid rate limits
            await asyncio.sleep(5)
            
        except Exception as e:
            print(f"Error processing {username}: {e}")
    
    print("Data collection complete!")

if __name__ == "__main__":
    asyncio.run(main())