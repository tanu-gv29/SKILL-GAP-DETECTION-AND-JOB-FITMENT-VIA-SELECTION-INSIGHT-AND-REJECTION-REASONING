def format_roles(raw_text):
    roles = [line.strip() for line in raw_text.splitlines() if line.strip()]

    print("\nroles = [")
    for role in roles:
        print(f'    "{role}",')
    print("]")

# 🔽 Paste your raw roles here
raw_input_text = """
Blockchain Solutions Architect
AI-Powered Healthcare Consultant
Digital Product Marketing Specialist
AI-Enhanced Customer Experience Manager
Environmental Data Analyst
AI-Driven Talent Acquisition Specialist
Robotics Software Developer
AI-Powered Insurance Underwriter
Smart Energy Systems Engineer
AI-Powered Compliance Officer
Digital Forensics Analyst
AI-Enhanced Language Translator
AI-Powered Content Strategist
Autonomous Vehicle Safety Engineer
AI-Powered Retail Operations Manager
Smart Transportation Analyst
AI-Driven Cloud Solutions Engineer
Digital Media Specialist
Smart Farming Consultant
AI-Enhanced Supply Chain Analyst
Cybersecurity Solutions Architect
AI-Powered UX Researcher
Autonomous Manufacturing Engineer
Smart Waste Management Engineer
AI-Driven Crisis Response Coordinator
Virtual Reality Training Specialist
AI-Powered Marketing Automation Specialist
AI-Powered Energy Efficiency Analyst
Smart Construction Project Manager
AI-Driven Financial Crime Investigator
Smart Waste Collection Manager
AI-Enhanced Language Processing Engineer
Digital Experience Optimization Manager
Smart Water Management Engineer
AI-Powered Risk Management Consultant
Digital Supply Chain Manager
AI-Powered Investment Analyst
Smart Manufacturing Automation Engineer
AI-Driven Data Governance Specialist
Smart Grid Integration Engineer
AI-Powered Financial Planning Analyst
Smart City Mobility Solutions Engineer
AI-Enhanced Talent Management Consultant
Digital Innovation Consultant
AI-Powered Fraud Risk Analyst
Smart Wearable Technology Developer
AI-Driven Product Recommendation Engineer
Smart Logistics Network Engineer
AI-Powered Healthcare Data Engineer
Smart Traffic Management Engineer


"""

format_roles(raw_input_text)