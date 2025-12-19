import os
import traceback
import datetime
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from flask_sqlalchemy import SQLAlchemy
from werkzeug.utils import secure_filename
import PyPDF2
import json

import google.generativeai as genai
from pgvector.sqlalchemy import Vector
from sqlalchemy import func, text
from dotenv import load_dotenv

load_dotenv()
app = Flask(__name__)
CORS(app)

# --- Configuration ---
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

NEON_DB_URL = os.getenv("POSTGRES_API_KEY")
app.config['SQLALCHEMY_DATABASE_URI'] = NEON_DB_URL
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
# Ensure this folder exists and place your PDFs here
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'resumes')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

db = SQLAlchemy(app)

# --- Helper Functions ---
def get_embedding(text):
    try:
        text = text.replace("\n", " ").strip()
        if not text:
            return None
        # Using a standard embedding model
        result = genai.embed_content(
            model="models/text-embedding-004",
            content=text,
            task_type="retrieval_document"
        )
        return result['embedding']
    except Exception as e:
        print(f"Google API Error: {e}")
        return None

def extract_text_from_pdf(filepath):
    text = ""
    try:
        with open(filepath, 'rb') as f:
            reader = PyPDF2.PdfReader(f)
            for page in reader.pages:
                extracted = page.extract_text()
                if extracted:
                    text += extracted
    except Exception as e:
        print(f"Error reading file {filepath}: {e}")
    return text

# --- Models ---
class Resume(db.Model):
    __tablename__ = 'resumes'
    id = db.Column(db.Integer, primary_key=True)
    filename = db.Column(db.String(255), nullable=False)
    filepath = db.Column(db.String(255), nullable=False)
    full_text = db.Column(db.Text, nullable=False)
    tag = db.Column(db.String(50))
    embedding = db.Column(Vector(768)) 
    date_uploaded = db.Column(db.DateTime, default=datetime.datetime.utcnow)

class Job(db.Model):
    __tablename__ = 'jobs'
    id = db.Column(db.Integer, primary_key=True)
    title = db.Column(db.String(100), nullable=False)
    company = db.Column(db.String(100), nullable=True)
    url = db.Column(db.String(500), nullable=False)
    status = db.Column(db.String(20), default='Applied')
    date_applied = db.Column(db.DateTime, default=datetime.datetime.now)
    notes = db.Column(db.Text, nullable=True)
    resume_id = db.Column(db.Integer, db.ForeignKey('resumes.id'), nullable=True)
    
    # Relationship to access resume filename easily
    resume = db.relationship('Resume', backref='jobs')

    def to_dict(self):
        return {
            "id": self.id,
            "title": self.title,
            "company": self.company,
            "url": self.url,
            "status": self.status,
            "date": self.date_applied.strftime("%Y-%m-%d"),
            "resume_used": self.resume.filename if self.resume else "None"
        }

with app.app_context():
    db.session.execute(func.text("CREATE EXTENSION IF NOT EXISTS vector"))
    db.create_all()

# --- Logic to Scan Folder ---
def ingest_folder_resume():
    processed_count = 0
    errors = []

    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        return 0, ['Resume folder not found']
    
    # Get all PDF files
    files = [f for f in os.listdir(app.config['UPLOAD_FOLDER']) if f.lower().endswith('.pdf')]
    print(f"--- Scanning Folder: Found {len(files)} PDFs ---")

    for filename in files:
        file_exist = Resume.query.filter_by(filename=filename).first()
        if file_exist:
            print(f"Skipping {filename} (Already in DB)")
            continue

        print(f"Processing new file: {filename}...")
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)

        extracted_text = extract_text_from_pdf(filepath)
        if not extracted_text:
            errors.append(f"Could not extract text from {filename}")
            continue

        
        vector = get_embedding(extracted_text[:8000])
        if not vector:
            errors.append(f"Failed to generate embedding for {filename}")
            continue
        
        # 4. Create and Save Resume Object (Fixed this part)
        new_resume = Resume(
            filename=filename,
            filepath=filepath,
            full_text=extracted_text,
            tag='Local Folder',  # Tag indicating it came from local scan
            embedding=vector
        )
        db.session.add(new_resume)
        processed_count += 1
    
    db.session.commit()
    return processed_count, errors

@app.route('/')
def dashboard():
    """View all tracked jobs"""
    jobs = Job.query.order_by(Job.date_applied.desc()).all()
    return render_template('dashboard.html', jobs=[j.to_dict() for j in jobs])

@app.route('/scan-resumes', methods=['POST', 'GET'])
def scan_resumes_route():
    """Trigger the folder scan manually via API or Button"""
    count, errors = ingest_folder_resume()
    return jsonify({
        "message": f"Scan complete. Processed {count} new resumes.",
        "errors": errors
    })

@app.route('/get-best-resume', methods=['POST'])
def get_best_resume():
    print("\nXXX --- STARTING REQUEST: /get-best-resume --- XXX")
    
    try:
        # --- STEP 1: Receive Data ---
        data = request.json
        job_description = data.get('job_description', '')
        print(f"1. Data received. Job Description Length: {len(job_description)} chars")
        
        if not job_description or len(job_description) < 10:
            print("ERROR: Job description is too short.")
            return jsonify({"error": "Job description too short"}), 400
        
        # --- STEP 2: Generate Embedding ---
        print("2. Generating Google AI embedding...")
        query_embedding = get_embedding(job_description)
        
        if not query_embedding:
            print("ERROR: Embedding generation returned None.")
            return jsonify({"error": "Failed to generate AI embedding"}), 500
        print("   -> Embedding generated successfully.")

        sql = text("""
            SELECT id, full_text, filename, 
                   1 - (embedding <=> CAST(:query_embedding AS vector)) as match_score
            FROM resumes
            ORDER BY match_score DESC
            LIMIT 1;
        """)
        
        embedding_str = str(query_embedding)
        print("3. Executing SQL query...")
        
        result = db.session.execute(sql, {'query_embedding': embedding_str}).fetchone()

        if not result:
            print("   -> No resumes found in the database.")
            return jsonify({"message": "No resumes found. Please scan folder."}), 404
        
        best_filename = result[2]
        match_score = result[3]
        best_resume_text = result[1]
        print(f"   -> Found Match: {best_filename} (Score: {match_score})")

        print("4. Calling Gemini 1.5 Flash for extraction...")
        model = genai.GenerativeModel('gemini-pro') 
        
        prompt = f"""
        You are an assistant and a job tracker for end user named: Vanessa Lopez.
        Extract the following fields from the best Vanessa Lopez's resume text below into a valid JSON Object.
        Keys: first_name, last_name, email, phone, linkedin, github, portfolio_url, city_address, zipcode, education, work_experience_titles, work_experience_summary, summary,bio
        
        Rules:
        1. If information is missing, write: "missing" or "na"
        2. Return ONLY the raw JSON string. Do not use Markdown formatting (no ```json).
        
        RESUME TEXT:
        {best_resume_text[:10000]}
        """
        
        try:
            ai_response = model.generate_content(prompt)
            print("   -> Gemini responded.")

            clean_json = ai_response.text.strip()
            if clean_json.startswith("```"):
                clean_json = clean_json.replace("```json", "").replace("```", "")
            
            print(f"   -> Raw JSON from AI: {clean_json[:100]}...") 
            
            profile_data = json.loads(clean_json)
            print("   -> JSON parsed successfully.")
            
        except Exception as ai_error:
            print(f"WARNING: AI Extraction failed: {ai_error}")
            # AI
            profile_data = {}

        profile_data['matched_filename'] = best_filename
        profile_data['matched_id'] = result[0]
        profile_data['match_score'] = round(float(match_score), 2)

        print("5. Sending successful response to client.")
        return jsonify(profile_data)

    except Exception as e:
        print("\n!!!!!!!! CRITICAL ERROR !!!!!!!!")
        print(f"Error Message: {str(e)}")
        print("Full Traceback:")
        traceback.print_exc()  # This prints the specific line number where it failed
        return jsonify({"error": str(e)}), 500
@app.route('/track', methods=['POST'])
def track_job():
    data = request.json
    existing_job = Job.query.filter_by(url=data.get('url')).first()
    if existing_job:
        return jsonify({"message": "Already tracked", "id": existing_job.id}), 200

    new_job = Job(
        title=data.get('title', 'Unknown Role'),
        company=data.get('company', 'Unknown Company'),
        url=data.get('url'),
        resume_id=data.get('resume_id'),
        status='Applied'
    )
    db.session.add(new_job)
    db.session.commit()
    return jsonify({"message": "Application saved!", "id": new_job.id})
@app.route('/tailor-resume', methods=['POST'])
def tailorResume():
    """ Receives job description from chrome extension then finds best matching resume in db. 
        Then uses gemini to help rewrite resume to match ATS scanner
    """
    try:
        data = request.json
        job_description = data.get('job_description', '')
        if len(job_description) < 50:
            return jsonify({"error": "job is not there, please highlight the ENTIRE job description"}), 400
        query_embedding = get_embedding(job_description)
        embedding_str = str(query_embedding)
        sql = text("""
            SELECT id, full_text, filename,
            1 - (embedding <=> CAST(:query_embedding AS vector)) as match_score
            FROM resumes
            ORDER BY match_score DESC
            LIMIT 1;
        """)
        result = db.session.execute(sql, {'query_embedding': embedding_str}).fetchone()
        if not result:
            return jsonify({"error": "No resumes found in DB"}), 404
        base_resume = result[1]

        model = genai.GenerativeModel('gemini-pro')
        # Prompt designed to pass ATS
        prompt = f"""
        You are an professional Resume Writer and ATS Optimizer
        Target Role: Based on the Job Description below.

        JOB DESCRIPTION:
        {job_description[:5000]}

        CURRENT RESUME:
        {base_resume[:5000]}

        TASK:
        1. Identify the top 5 hard skills required by the Job Description.
        2. Rewrite the "Professional Summary" to specifically highlight these skills.
        3. Rewrite 3 key bullet points from the experience to use keywords from the Job Description (Action Verbs + Result).
        4. Generate a short "Cover Letter" specific to this role.

        OUTPUT FORMAT (JSON ONLY):
        {{
            "matched_skills": ["Skill 1", "Skill 2"...],
            "tailored_summary": "...",
            "tailored_bullets": ["...", "...", "..."],
            "cover_letter_body": "..."
        }}
        """

        response = model.generate_content(prompt)
        clean_json = response.text.replace("```json", "").replace("```", "").strip()
        tailored_data = json.loads(clean_json)

        return jsonify(tailored_data)

    except Exception as e:
        print(f"Tailoring Error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/update-status/<int:job_id>', methods=['PUT'])
def update_status(job_id):
    data = request.json
    job = Job.query.get(job_id)
    if not job:
        return jsonify({"error": "Job not found"}), 404
        
    job.status = data.get('status', job.status)
    db.session.commit()
    return jsonify({"message": "Status updated", "new_status": job.status})

if __name__ == '__main__':
    with app.app_context():
        ingest_folder_resume()
    app.run(port=8080, debug=True)