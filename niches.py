# niches.py
# 50 niches + drop-in QUERY_PACKS (5 queries each) optimized for global Shorts discovery.
#
# Usage:
#   from niches import NICHES, QUERY_PACKS
#
# Notes:
# - Queries are designed to bias toward Shorts via "#shorts", "shorts", or "short".
# - Keep queries broad/global to avoid overfitting to one region or language.
# - You can tune/extend per niche over time, but this file is complete for MVP.

from __future__ import annotations

from typing import Dict, List

NICHES: List[str] = [
    "AI Tools & Prompts",
    "Motivation & Self-Improvement",
    "Side Hustles & Money Tips",
    "Personal Finance & Budgeting",
    "Investing & Trading",
    "Crypto & Web3",
    "Business & Entrepreneurship",
    "Career Advice & Interviews",
    "Productivity & Study",
    "Coding & Software Dev",
    "Cybersecurity",
    "Data Science & ML",
    "Tech Reviews & Gadgets",
    "Gaming Highlights",
    "Minecraft",
    "Roblox",
    "Fortnite",
    "Mobile Games",
    "Sports Clips (General)",
    "Football/Soccer",
    "Basketball",
    "Combat Sports (Boxing/MMA)",
    "Fitness Workouts",
    "Weight Loss & Nutrition",
    "Recipes & Cooking",
    "Food Hacks",
    "Travel & Hidden Gems",
    "Cars & Motorcycles",
    "Luxury Lifestyle",
    "Fashion & Style",
    "Beauty & Skincare",
    "Relationships & Dating",
    "Psychology & Human Behavior",
    "Life Hacks",
    "Home Organization & Cleaning",
    "DIY & Crafts",
    "Woodworking & Tools",
    "Gardening & Plants",
    "Pets (Cats/Dogs)",
    "Wildlife & Nature",
    "Health Facts & Wellness",
    "History Facts",
    "Science Facts",
    "Space & Astronomy",
    "Geography & Maps",
    "Language Learning",
    "Music & Singing",
    "Dance",
    "Movies & TV (Edits/Clips/Trivia)",
    "Memes & Comedy",
]

QUERY_PACKS: Dict[str, List[str]] = {

    "AI Tools & Prompts": [
        "chatgpt prompts #shorts",
        "ai tools #shorts",
        "best ai apps #shorts",
        "ai productivity #shorts",
        "midjourney tips #shorts",
    ],

    "Motivation & Self-Improvement": [
        "motivation #shorts",
        "discipline mindset #shorts",
        "self improvement #shorts",
        "success habits #shorts",
        "mental toughness #shorts",
    ],

    "Side Hustles & Money Tips": [
        "side hustle ideas #shorts",
        "make money online #shorts",
        "online income tips #shorts",
        "passive income #shorts",
        "extra income ideas #shorts",
    ],

    "Personal Finance & Budgeting": [
        "personal finance tips #shorts",
        "budgeting tips #shorts",
        "money management #shorts",
        "saving money hacks #shorts",
        "financial literacy #shorts",
    ],

    "Investing & Trading": [
        "investing tips #shorts",
        "stock market basics #shorts",
        "trading strategies #shorts",
        "long term investing #shorts",
        "day trading #shorts",
    ],

    "Crypto & Web3": [
        "crypto explained #shorts",
        "bitcoin facts #shorts",
        "altcoin news #shorts",
        "blockchain explained #shorts",
        "web3 projects #shorts",
    ],

    "Business & Entrepreneurship": [
        "entrepreneur mindset #shorts",
        "startup advice #shorts",
        "business tips #shorts",
        "small business ideas #shorts",
        "scaling a business #shorts",
    ],

    "Career Advice & Interviews": [
        "career advice #shorts",
        "job interview tips #shorts",
        "resume tips #shorts",
        "career growth #shorts",
        "workplace success #shorts",
    ],

    "Productivity & Study": [
        "productivity hacks #shorts",
        "study tips #shorts",
        "focus better #shorts",
        "time management #shorts",
        "deep work #shorts",
    ],

    "Coding & Software Dev": [
        "coding tips #shorts",
        "learn programming #shorts",
        "python coding #shorts",
        "software developer life #shorts",
        "coding mistakes #shorts",
    ],

    "Cybersecurity": [
        "cybersecurity tips #shorts",
        "hacking explained #shorts",
        "online safety #shorts",
        "password security #shorts",
        "ethical hacking #shorts",
    ],

    "Data Science & ML": [
        "machine learning explained #shorts",
        "data science tips #shorts",
        "ai models explained #shorts",
        "deep learning basics #shorts",
        "ml projects #shorts",
    ],

    "Tech Reviews & Gadgets": [
        "tech gadgets #shorts",
        "smartphone review #shorts",
        "cool tech #shorts",
        "tech unboxing #shorts",
        "latest gadgets #shorts",
    ],

    "Gaming Highlights": [
        "gaming highlights #shorts",
        "best gaming moments #shorts",
        "pro gamer clips #shorts",
        "funny gaming clips #shorts",
        "gaming fails #shorts",
    ],

    "Minecraft": [
        "minecraft shorts",
        "minecraft builds #shorts",
        "minecraft hacks #shorts",
        "minecraft survival #shorts",
        "minecraft funny #shorts",
    ],

    "Roblox": [
        "roblox shorts",
        "roblox funny moments #shorts",
        "roblox gameplay #shorts",
        "roblox tips #shorts",
        "roblox hacks #shorts",
    ],

    "Fortnite": [
        "fortnite shorts",
        "fortnite highlights #shorts",
        "fortnite tips #shorts",
        "fortnite wins #shorts",
        "fortnite funny #shorts",
    ],

    "Mobile Games": [
        "mobile gaming #shorts",
        "android games #shorts",
        "ios games #shorts",
        "mobile game tips #shorts",
        "best mobile games #shorts",
    ],

    "Sports Clips (General)": [
        "sports highlights #shorts",
        "crazy sports moments #shorts",
        "sports skills #shorts",
        "sports fails #shorts",
        "sports motivation #shorts",
    ],

    "Football/Soccer": [
        "football highlights #shorts",
        "soccer skills #shorts",
        "goal compilation #shorts",
        "football facts #shorts",
        "best goals #shorts",
    ],

    "Basketball": [
        "basketball highlights #shorts",
        "nba moments #shorts",
        "basketball skills #shorts",
        "nba facts #shorts",
        "basketball training #shorts",
    ],

    "Combat Sports (Boxing/MMA)": [
        "boxing knockouts #shorts",
        "mma highlights #shorts",
        "ufc moments #shorts",
        "fight analysis #shorts",
        "combat sports facts #shorts",
    ],

    "Fitness Workouts": [
        "workout routine #shorts",
        "home workout #shorts",
        "gym workout #shorts",
        "fitness motivation #shorts",
        "hiit workout #shorts",
    ],

    "Weight Loss & Nutrition": [
        "weight loss tips #shorts",
        "fat loss hacks #shorts",
        "healthy eating #shorts",
        "nutrition facts #shorts",
        "diet mistakes #shorts",
    ],

    "Recipes & Cooking": [
        "easy recipes #shorts",
        "cooking tips #shorts",
        "quick meals #shorts",
        "home cooking #shorts",
        "recipe ideas #shorts",
    ],

    "Food Hacks": [
        "food hacks #shorts",
        "kitchen hacks #shorts",
        "cooking shortcuts #shorts",
        "food tricks #shorts",
        "meal prep hacks #shorts",
    ],

    "Travel & Hidden Gems": [
        "travel shorts",
        "hidden places #shorts",
        "travel tips #shorts",
        "budget travel #shorts",
        "beautiful places #shorts",
    ],

    "Cars & Motorcycles": [
        "car shorts",
        "supercars #shorts",
        "car modifications #shorts",
        "motorcycle clips #shorts",
        "car facts #shorts",
    ],

    "Luxury Lifestyle": [
        "luxury lifestyle #shorts",
        "millionaire lifestyle #shorts",
        "luxury homes #shorts",
        "luxury cars #shorts",
        "rich life #shorts",
    ],

    "Fashion & Style": [
        "fashion tips #shorts",
        "outfit ideas #shorts",
        "streetwear #shorts",
        "mens fashion #shorts",
        "womens fashion #shorts",
    ],

    "Beauty & Skincare": [
        "skincare routine #shorts",
        "beauty tips #shorts",
        "makeup hacks #shorts",
        "skin care facts #shorts",
        "beauty transformation #shorts",
    ],

    "Relationships & Dating": [
        "dating advice #shorts",
        "relationship tips #shorts",
        "dating psychology #shorts",
        "love advice #shorts",
        "toxic relationships #shorts",
    ],

    "Psychology & Human Behavior": [
        "psychology facts #shorts",
        "human behavior #shorts",
        "body language #shorts",
        "mind tricks #shorts",
        "cognitive bias #shorts",
    ],

    "Life Hacks": [
        "life hacks #shorts",
        "daily hacks #shorts",
        "smart tricks #shorts",
        "useful hacks #shorts",
        "everyday hacks #shorts",
    ],

    "Home Organization & Cleaning": [
        "cleaning hacks #shorts",
        "home organization #shorts",
        "clean with me #shorts",
        "declutter tips #shorts",
        "satisfying cleaning #shorts",
    ],

    "DIY & Crafts": [
        "diy projects #shorts",
        "craft ideas #shorts",
        "easy diy #shorts",
        "home diy #shorts",
        "creative crafts #shorts",
    ],

    "Woodworking & Tools": [
        "woodworking projects #shorts",
        "carpentry tips #shorts",
        "power tools #shorts",
        "woodworking hacks #shorts",
        "satisfying woodworking #shorts",
    ],

    "Gardening & Plants": [
        "gardening tips #shorts",
        "plant care #shorts",
        "urban gardening #shorts",
        "houseplants #shorts",
        "grow vegetables #shorts",
    ],

    "Pets (Cats/Dogs)": [
        "funny cats #shorts",
        "funny dogs #shorts",
        "pet care tips #shorts",
        "cute animals #shorts",
        "pet training #shorts",
    ],

    "Wildlife & Nature": [
        "wildlife shorts",
        "nature facts #shorts",
        "animals in the wild #shorts",
        "amazing nature #shorts",
        "animal behavior #shorts",
    ],

    "Health Facts & Wellness": [
        "health facts #shorts",
        "wellness tips #shorts",
        "healthy habits #shorts",
        "medical facts #shorts",
        "mental health tips #shorts",
    ],

    "History Facts": [
        "history facts #shorts",
        "did you know history #shorts",
        "ancient history #shorts",
        "war history #shorts",
        "historical events #shorts",
    ],

    "Science Facts": [
        "science facts #shorts",
        "did you know science #shorts",
        "physics facts #shorts",
        "biology facts #shorts",
        "chemistry facts #shorts",
    ],

    "Space & Astronomy": [
        "space facts #shorts",
        "astronomy facts #shorts",
        "universe explained #shorts",
        "nasa facts #shorts",
        "black holes #shorts",
    ],

    "Geography & Maps": [
        "geography facts #shorts",
        "map facts #shorts",
        "country facts #shorts",
        "world geography #shorts",
        "borders explained #shorts",
    ],

    "Language Learning": [
        "learn english #shorts",
        "language tips #shorts",
        "english vocabulary #shorts",
        "learn spanish #shorts",
        "language hacks #shorts",
    ],

    "Music & Singing": [
        "singing shorts",
        "vocal tips #shorts",
        "music facts #shorts",
        "singing challenge #shorts",
        "cover song #shorts",
    ],

    "Dance": [
        "dance shorts",
        "dance challenge #shorts",
        "hip hop dance #shorts",
        "dance tutorial #shorts",
        "freestyle dance #shorts",
    ],

    "Movies & TV (Edits/Clips/Trivia)": [
        "movie scenes #shorts",
        "movie facts #shorts",
        "tv show clips #shorts",
        "film trivia #shorts",
        "cinema edits #shorts",
    ],

    "Memes & Comedy": [
        "funny shorts",
        "comedy clips #shorts",
        "relatable memes #shorts",
        "viral memes #shorts",
        "sketch comedy #shorts",
    ],
}

# Basic integrity check (optional): ensures all niches have query packs.
# You can comment out if you prefer a pure data module.
_missing = [n for n in NICHES if n not in QUERY_PACKS]
if _missing:
    raise ValueError(f"Missing QUERY_PACKS entries for: {_missing}")
