from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()

class Postings(db.Model):
    __tablename__ = 'postings' 
    term_id = db.Column(db.Integer, primary_key=True) 
    postings = db.Column(db.LargeBinary, nullable=False)

class Articles(db.Model):
    __tablename__ = 'articles'
    id = db.Column(db.Integer, primary_key=True)
    article_id = db.Column(db.Integer)
    article_name = db.Column(db.String)
    article_text = db.Column(db.Text)
