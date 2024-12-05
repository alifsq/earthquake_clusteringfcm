from flask_sqlalchemy import SQLAlchemy
from flask_login import UserMixin

db = SQLAlchemy()

class User(db.Model, UserMixin):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(150), unique=True, nullable=False)
    password = db.Column(db.String(150), nullable=False)

class Earthquake(db.Model):
    __tablename__ = 'earthquake'
    id = db.Column(db.Integer, primary_key=True)
    tanggal = db.Column(db.Date, nullable=False)
    waktu = db.Column(db.String(5), nullable=False)  # MM.SS
    latitude = db.Column(db.Float, nullable=False)
    longitude = db.Column(db.Float, nullable=False)
    magnitude = db.Column(db.Float, nullable=False)
    depth = db.Column(db.Float, nullable=False)
    # Tambahkan kolom cluster_label
    # cluster_label = db.Column(db.Integer, nullable=False)  # Nilai default None jika belum di-cluster

    def __init__(self, tanggal, waktu, latitude, longitude, magnitude, depth,):
        self.tanggal = tanggal
        self.waktu = waktu
        self.latitude = latitude
        self.longitude = longitude
        self.magnitude = magnitude
        self.depth = depth

class ClusteredEarthquake(db.Model):
    __tablename__ = 'clustered_earthquake'
    
    id = db.Column(db.Integer, primary_key=True)
    earthquake_id = db.Column(db.Integer, db.ForeignKey('earthquake.id'), nullable=False)
    cluster_label = db.Column(db.Integer, nullable=True)

    def __init__(self, earthquake_id, cluster_label):
        self.earthquake_id = earthquake_id
        self.cluster_label = cluster_label
