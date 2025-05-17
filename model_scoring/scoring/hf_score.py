from huggingface_hub import model_info
from datetime import datetime, timezone
import math


def get_model_downloads(model_name):
    """Get the last 30 days downloads for a model"""
    model = model_info(model_name)
    return model.downloads


def get_model_likes(model_name):
    """Get the number of likes for a model"""
    model = model_info(model_name)
    return model.likes


def get_model_age(model_name):
    """Get the age of a model in weeks and months"""
    model = model_info(model_name)
    created_at = model.created_at
    now = datetime.now(timezone.utc)
    age_delta = now - created_at
    age_weeks = age_delta.days // 7
    age_months = age_delta.days // 30
    return age_weeks, age_months


def compute_hf_score(model_info):
    """Improved scoring formula with better discrimination for less popular models"""
    downloads = model_info["downloads in last 30 days"]
    likes = model_info["total likes"]
    age_weeks = model_info["age in weeks"]
    
    # Enhanced download scaling (0-4 points)
    # Better differentiation across orders of magnitude
    if downloads <= 0:
        download_score = 0
    else:
        log_downloads = math.log10(downloads)
        # Base points for different tiers
        if log_downloads < 1:  # <10 downloads
            download_score = 0
        elif log_downloads < 3:  # 10-999 downloads
            download_score = log_downloads / 3  # 0-1 points
        elif log_downloads < 5:  # 1K-99K downloads
            download_score = 1 + (log_downloads - 3) / 1  # 1-3 points
        else:  # 100K+ downloads
            download_score = 3 + min(1, (log_downloads - 5) / 2)  # 3-4 points

    download_score = min(4, download_score)
    
    # Stricter likes scaling (0-4 points)
    # Requires >1,000 likes to get full points
    likes_score = min(4, 0 if likes <= 0 else max(0, (math.log10(max(1, likes)) - 0.5) / 0.75))
    
    # More restrictive age scoring (0-2 points)
    # Rewards models between 3-12 months old
    age_score = min(2, max(0, (age_weeks - 4) / 12) if age_weeks < 16 else 
                   (2.0 if age_weeks < 52 else max(0.5, 2.0 - (age_weeks - 52) / 104)))
    
    return round(download_score + likes_score + age_score, 1)


def extract_model_info(model_name):
    """Extract all information for a model and return as a dictionary"""
    downloads = get_model_downloads(model_name)
    likes = get_model_likes(model_name)
    age_weeks, age_months = get_model_age(model_name)
    
    info = {
        "model_name": model_name,
        "downloads in last 30 days": downloads,
        "total likes": likes,
        "age in weeks": age_weeks,
        "age in months": age_months
    }
    
    # Add community score
    info["community_score"] = compute_hf_score(info)
    
    return info


if __name__ == "__main__":
    # Example usage
    model_name = "deepseek-ai/DeepSeek-V3-0324" # Change this to the model you want to score
    
    # Get all metrics at once
    info = extract_model_info(model_name)
    
    # Print score prominently
    print(f"\nHF COMMUNITY SCORE: {info['community_score']}/10")
    
    print("\nDetailed metrics:")
    for key, value in info.items():
        if key != "community_score":  # Skip community_score as we already displayed it
            print(f"{key}: {value}")



