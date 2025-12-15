from app import app, db
from sqlalchemy import text

with app.app_context():
    print("🗑️  Deleting old database table...")
    # Force delete the table
    try:
        db.session.execute(text("DROP TABLE IF EXISTS resumes CASCADE;"))
        db.session.commit()
        print("✅ Old table deleted.")
    except Exception as e:
        print(f"⚠️  Note: {e}")

    print("🔨 Building new fresh table...")
    # Create the new table based on your updated app.py code
    db.create_all()
    print("✨ Database is ready for Google Gemini (768) and Long Text!")