from ai_modules.email_generator import generate as generate_email
from ai_modules.landing_page_generator import generate as generate_landing
from ai_modules.whatsapp_generator import generate as generate_whatsapp

campaign_data = {
    "business": {
        "name": "Aqwaya",
        "industry": "Skincare",
        "website": "https://aqwaya.com",
        "brand_tone": "friendly, professional",
        "brand_keywords": ["natural", "glow", "africa"]
    },
    "prompt": "Promote our new vitamin C serum for glowing skin",
    "audience": "women 18-30 in Lagos",
    "desired_cta": "Shop Now",
    "language": "en-NG"
}

email_result = generate_email(campaign_data)
landing_result = generate_landing(campaign_data)
whatsapp_result = generate_whatsapp(campaign_data)