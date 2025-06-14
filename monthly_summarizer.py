import os
import pandas as pd
import openai
from collections import Counter
from datetime import datetime, timedelta
from pymongo import MongoClient
import time
import smtplib
from email.message import EmailMessage
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from dotenv import load_dotenv
load_dotenv()

# Configuration
openai.api_key = os.getenv("TOGETHER_API_KEY")
openai.api_base = "https://api.together.xyz/v1"
MODEL_NAME = "deepseek-ai/DeepSeek-R1-Distill-Llama-70B-Free"

MONGO_URI = os.getenv("MONGO_URI")
MONGO_DB = os.getenv("MONGO_DB", "brand_monitoring")
MONGO_COLLECTION = os.getenv("MONGO_COLLECTION", "processed_articles")
MONGO_SUMMARIES_COLLECTION = "brand_monitoring_summaries"

EMAIL_PORT = int(os.getenv("EMAIL_PORT", 587))
EMAIL_CONFIG = {
    "EMAIL_SENDER": os.getenv("EMAIL_SENDER"),
    "EMAIL_PASSWORD": os.getenv("EMAIL_PASSWORD"),
    "EMAIL_RECEIVER": os.getenv("EMAIL_RECEIVER"),
    "SMTP_SERVER": os.getenv("SMTP_SERVER", "smtp.gmail.com"),
    "SMTP_PORT": int(os.getenv("SMTP_PORT", 587)),
}

# MongoDB
def load_daily_articles():
    """Load articles scraped TODAY (execution day: April 6, May 6, etc.)"""
    client = MongoClient(MONGO_URI)
    collection = client[MONGO_DB][MONGO_COLLECTION]
    
    today = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
    query = {
        "scraped_date": {
            "$gte": today,
            "$lt": today + timedelta(days=1)
        }
    }
    
    data = list(collection.find(query))
    if not data:
        print(f"No articles for {today.strftime('%Y-%m-%d')}")
        return pd.DataFrame()

    df = pd.DataFrame(data)
    df['scraped_date'] = pd.to_datetime(df['scraped_date'], errors='coerce')
    return df.dropna(subset=['scraped_date'])

# Prompt Engineering
def build_prompt(df):
    sentiment_summary = Counter(
        s['sentiment']
        for row in df['sentiment_analysis'].dropna()
        for s in row
    )

    tags = [tag for row in df['tags'].dropna() for tag in row]
    top_keywords = Counter(tags).most_common(10)

    engagement_df = df.dropna(subset=['upvotes', 'comments'])
    top_engaged = engagement_df.sort_values(by=['upvotes', 'comments'], ascending=False).head(5)

    content_snippets = "\n\n---\n\n".join(df['content'].fillna('').str[:1000].tolist())

    prompt = f"""
You are an analytics and communications expert reviewing online conversations and articles related to the Condominium Authority of Ontario (CAO).

Based on the following scraped content from various sources (news, forums, Reddit, etc.), provide a structured, insightful summary about how CAO is being discussed in the public sphere.

Be concise, but thorough. Avoid speculation, relying on what's present in the content. Use a consistent tone suitable for a communications briefing or stakeholder update.

Summarize the following dimensions:

1. **Sentiment Trends**: General tone toward CAO. Are users supportive, critical, or neutral? Any noticeable shifts?
2. **Common Issues**: What recurring complaints or issues are associated with CAO? Legal disputes? Governance concerns?
3. **Top Keywords & Topics**: Which terms or subjects are most frequently mentioned in connection with CAO?
4. **Engagement Highlights**: Which posts have the highest numbers of comments and upvotes?
5. **Discussion Peaks**: Any timeframe where discussion volume surged? Why?
6. **Other Insights**: Subtle tone patterns, changing sentiment across platforms, or surprising mentions related to CAO.
7. **Summary Insights**: In 3-5 sentences, synthesize the main message of the posts and articles, list some key concerns, and the perceived sentiment towards the CAO. Highlight any reputational risks or public perception challenges.
8. **Top Reddit Posts**: List the top 3 Reddit posts with the most upvotes and comments, only if there are enough entries where the source is Reddit, if not, leave this out.

---

**Sentiment Breakdown**:
{dict(sentiment_summary)}

**Top Keywords**:
{top_keywords}

**Top Reddit Posts by Engagement**:
{top_engaged[['title', 'upvotes', 'comments']].to_string(index=False)}

---

**Content Sample (truncated)**:
{content_snippets}
"""
    return prompt

# Execution Logic
def generate_summary(prompt):
    try:
        response = openai.ChatCompletion.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=2048
        )
        return response["choices"][0]["message"]["content"].strip()
    except Exception as e:
        raise RuntimeError(f"LLM failed: {str(e)}")

def send_report(summary, article_count):
    date_str = datetime.utcnow().strftime("%Y-%m-%d")
    subject = f"CAO Summary - {date_str} ({article_count} articles)"
    
    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = EMAIL_CONFIG["EMAIL_SENDER"]
    msg["To"] = EMAIL_CONFIG["EMAIL_RECEIVER"]
    msg.set_content(
        f"Summary for {date_str}\n\n"
        f"Articles processed: {article_count}\n\n"
        f"Summary:\n{summary[:3000]}..."
    )

    try:
        with smtplib.SMTP(EMAIL_CONFIG["SMTP_SERVER"], EMAIL_CONFIG["SMTP_PORT"]) as server:
            server.starttls()
            server.login(EMAIL_CONFIG["EMAIL_SENDER"], EMAIL_CONFIG["EMAIL_PASSWORD"])
            server.send_message(msg)
        print("Summary email sent.")
    except Exception as e:
        print(f"Email failed: {str(e)}")

def run_summary():
    df = load_daily_articles()
    # Use datetime object for date fields
    date_dt = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)

    if df.empty:
        print(f"No data for {date_dt}")
        send_report("No articles found today.", 0)
        return

    try:
        prompt = build_prompt(df)
        summary = generate_summary(prompt)
        
        # Format the summary with metadata (markdown)
        formatted_summary = f"{summary}\n\n---\n\n#### ** Metadata**  \n- **Generated at**: {date_dt.isoformat()}Z  \n- **Total Articles Analyzed**: {len(df)}"
        
        # Save to MongoDB with consistent structure (date as datetime)
        client = MongoClient(MONGO_URI)
        client[MONGO_DB][MONGO_SUMMARIES_COLLECTION].update_one(
            {"date": date_dt},
            {"$set": {
                "date": date_dt,
                "summary": formatted_summary,
                "articles": len(df)
            }},
            upsert=True
        )
        
        send_report(formatted_summary, len(df))
        print(f"Processed {len(df)} articles for {date_dt}")

    except Exception as e:
        error_msg = f"Summary failed: {str(e)}"
        print(error_msg)
        send_report(error_msg, 0)

# Main function
if __name__ == "__main__":
    run_summary()
