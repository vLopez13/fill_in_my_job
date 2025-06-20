
import time
import os

# this willl be the profile for filling out job fields
profile = {
    "firstName": "Mary",
    "lastName": "Doe",
    "email_address": "mary_doe@gmail.com",
    "phone_number": "123-456-7890",
    "address": "415 Mission Street",
    "city": "San Francisco",
    "state": "California",
    "zip_code": "94105",
    "country": "United States",
    "years_of_experience": "2",
    "skills": "Python, Java, C++, Apex, SQL, Excel, LWC, Visualforce, Javascript, HTML, CSS",
    "linkedin_profile": "https://www.linkedin.com/in/marydoe",
    "resume_file": "path/to/resume.pdf",  # Update this with an actual file
    "date_of_birth": "2000-01-02",
    "salary_expectation_per_year": "84,000.00",
    "salary_expectation_per_hour": "41.00",
    "salary_range_per_hour": "40.00-55.00",
    "salary_range_per_year": "80,000.00-100,000.00",
    "willing_to_relocate": "No",
    "willing_to_travel": "No",
    "available_start_date": "2025-07-10",
    "preferred_working_hours": "7 AM - 5 PM",
    "preferred_job_location": "Remote",
    "need_accomodation": "No",
    "veteran_status": "I am not a protected veteran",
    "disability_status": "No, I do not have a disability",
    "require_sponsorship": "No",
    "us_citizen": "Yes",
    "hear_about_us":"Linkedin",
    "education_level": "Bachelor's Degree of Computer Science",
    "certifications": "Salesforce Certified Administrator, Salesforce Certified Platform Developer",
    "languages": "English, Russian, Spanish",
    "additional_information": "Looking for a challenging role in a fast-paced environment.",
    "references": "Available upon request",
    "cover_letter_file": "path/to/cover_letter.pdf",  # Update this with an actual file
    "agree_terms": True,
    "agree_privacy_policy": True,
    "subscribe_newsletter": False,
    

}
service = Service(executable_path="C:\Users\Vanessa\Desktop\chromedriver-win64")  # <-- UPDATE
driver = webdriver.Chrome(service=service)

driver.get("https://example.com/job-application") # <--REPLACE
#Helper
    def fill_if_exists(by, identifier, value):
    try:
        field = driver.find_element(by, identifier)
        field.clear()
        field.send_keys(value)
        print(f"Filled: {identifier}")
    except Exception as e:
        print(f"Skipped: {identifier} ({e})")