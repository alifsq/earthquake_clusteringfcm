
from flask import Flask, request, redirect, session, url_for, flash, render_template, send_file
from flask_login import LoginManager, login_user, login_required, logout_user, current_user 
from math import ceil
import pandas as pd
import os
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from fcmeans import FCM
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import io
import base64

from io import BytesIO

import folium
from folium.plugins import HeatMap

from datetime import datetime
from models import db, Earthquake,ClusteredEarthquake, User
from sqlalchemy import func
#EarthquakeCluster


app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI']='mysql+mysqlconnector://root:@localhost/clusterdb'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['SECRET_KEY'] = 'admin'

# Register the max and min filters
app.jinja_env.globals['max'] = max
app.jinja_env.globals['min'] = min

db.init_app(app)

# Konfigurasi Flask-Login
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))


@app.route('/', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        
        user = User.query.filter_by(username=username, password=password).first()
        if user:
            login_user(user)
            return redirect(url_for('inputdata'))
        # render_template("dashboard.html") # Mengarahkan ke dashboard
        else:
            flash('Invalid username or password', 'error')
            return render_template('login.html')
    return render_template('login.html')

# Tambahkan pengaturan untuk menangani pesan ketika user belum login
@login_manager.unauthorized_handler
def unauthorized():
    flash('Tolong login terlebih dahulu', 'info')  # Pesan flash khusus
    return redirect(url_for('login'))  # Redirect ke halaman login jika tidak login

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('login'))

@app.route("/inputdata")
@login_required
def inputdata():
   # Ambil halaman dari query parameter atau default ke halaman 1
    page = request.args.get('page', 1, type=int)
    per_page = 100  # Jumlah data per halaman

    # Hitung total data
    total_data = Earthquake.query.count()

    # Hitung total halaman
    total_pages = ceil(total_data / per_page)

    # Ambil data sesuai halaman dan batasi 100 data per halaman
    earthquakes = Earthquake.query.offset((page - 1) * per_page).limit(per_page).all()

    # Render template dan kirim data untuk paginasi
    return render_template('inputdata.html', earthquakes=earthquakes, page=page, total_pages=total_pages)

@app.route("/submit", methods=["POST"])
@login_required
def submit():
    tanggal = request.form['tanggal']
    waktu = request.form['waktu']
    latitude = float(request.form['latitude'])
    longitude = float(request.form['longitude'])
    magnitude = float(request.form['magnitude'])
    depth = float(request.form['depth'])

    # Membuat objek Earthquake baru
    new_earthquake = Earthquake(tanggal=tanggal, waktu=waktu, latitude=latitude,
                                 longitude=longitude, magnitude=magnitude, depth=depth)

    # Menyimpan ke database
    db.session.add(new_earthquake)
    db.session.commit()

    flash('Data gempa bumi berhasil disimpan!', 'success')
    return redirect(url_for('inputdata'))

#hapus data
@app.route("/delete/<int:id>", methods=["POST"])
@login_required
def delete(id):
    # Ambil data gempa yang ingin dihapus berdasarkan id_gempa
    earthquake_to_delete = Earthquake.query.get_or_404(id)

    try:
        # Hapus data dari database
        db.session.delete(earthquake_to_delete)
        db.session.commit()
        flash('Data gempa berhasil dihapus', 'success')
    except:
        db.session.rollback()
        flash('Terjadi kesalahan saat menghapus data', 'error')

    return redirect("/inputdata")

#edit data gempa
@app.route("/update", methods=["POST"])
@login_required
def update():
    # Tangkap data dari form edit
    id_gempa = request.form.get('id_gempa')
    earthquake = Earthquake.query.get(id_gempa)

    # Update field berdasarkan input dari form
    earthquake.tanggal = request.form['tanggal']
    earthquake.waktu = request.form['waktu']
    earthquake.latitude = request.form['latitude']
    earthquake.longitude = request.form['longitude']
    earthquake.magnitude = request.form['magnitude']
    earthquake.depth = request.form['depth']

    db.session.commit()
    flash('Data berhasil diperbarui', 'success')

    return redirect("/inputdata")

# Add data csv
# Rute untuk menangani upload file CSV
@app.route('/upload', methods=['POST'])
@login_required
def upload():
    if 'file' not in request.files:
        return "No file part"
    
    file = request.files['file']
    
    if file.filename == '':
        return "No selected file"

    # Baca CSV dengan pandas
    try:
        data = pd.read_csv(file)

        # Loop dan masukkan ke database
        for index, row in data.iterrows():

           # Pastikan bahwa nilai 'tanggal' bukan NaN
            if pd.notnull(row['tanggal']) and isinstance(row['tanggal'], str):
                try:
                    # Ubah format parsing tanggal ke 'm/d/Y'
                    tanggal = datetime.strptime(row['tanggal'], '%m/%d/%Y')
                except ValueError:
                    return f"Invalid date format in row {index + 1}"
            else:
                return f"Missing or invalid date in row {index + 1}"
        
            new_item = Earthquake(
                tanggal=tanggal,
                waktu=row['waktu'], 
                latitude=row['latitude'], 
                longitude=row['longitude'], 
                depth=row['depth'], 
                magnitude=row['magnitude'])
            db.session.add(new_item)
        db.session.commit()
        return redirect("/inputdata")
    
    except Exception as e:
        return f"Error: {e}"


@app.route('/prosesdata', methods=['GET'])
@login_required
def prosesdata():
    # Get the current page number from the request (default to page 1)
    page = request.args.get('page', 1, type=int)
    
    # Set the number of records per page (in this case, 100)
    records_per_page = 100

    # Query the data and paginate
    earthquakes = Earthquake.query.with_entities(
        Earthquake.latitude, 
        Earthquake.longitude, 
        Earthquake.magnitude, 
        Earthquake.depth
    ).paginate(page=page, per_page=records_per_page, error_out=False)

    # Extract data from the paginated result
    data_array = np.array([(eq.latitude, eq.longitude, eq.magnitude, eq.depth) for eq in earthquakes.items])

    # Normalize the data
    scaler = MinMaxScaler()
    normalized_data = scaler.fit_transform(data_array)

    # Convert the normalized data to a list
    normalized_data_list = normalized_data.tolist()

    # Calculate the total number of pages
    total_pages = earthquakes.pages

    return render_template(
        'prosesdata.html',
        data=normalized_data_list,
        page=page,
        total_pages=total_pages
    )


optimal_clusters= None
standardized_data=None
@app.route('/proseslanjut', methods=['GET', 'POST'])
@login_required
def proseslanjut():
    global optimal_clusters
    plot_url = None
    optimal_clusters = None
    fcm_centers = None
    silhouette_data = []  # Menyimpan data jumlah kluster dan skor silhouette
    
    # Ambil data dari kolom latitude, longitude, magnitude, dan depth
    earthquakes = Earthquake.query.with_entities(
            Earthquake.latitude, 
            Earthquake.longitude, 
            Earthquake.magnitude, 
            Earthquake.depth
    ).all()

    # Ubah data ke dalam format numpy array
    data_array = np.array(earthquakes)

    # Inisialisasi MinMaxScaler
    scaler = MinMaxScaler()

    # Normalisasi data
    standardized_data = scaler.fit_transform(data_array)

    # Pencarian jumlah kluster terbaik menggunakan FCM dan Silhouette Score
    cluster_range = range(2, 10)
    silhouette_scores = []

    for n_clusters in cluster_range:
        # Fuzzy C-Means clustering
        fcm = FCM(n_clusters=n_clusters, max_iter=100, m=2, error=0.01)
        fcm.fit(standardized_data)

        # Menentukan keanggotaan terbesar untuk setiap titik data
        cluster_labels = np.argmax(fcm.u, axis=1)

        # Menghitung skor silhouette
        silhouette_avg = silhouette_score(standardized_data, cluster_labels)
        silhouette_scores.append(silhouette_avg)

        # Simpan jumlah kluster dan skor silhouette ke dalam list
        silhouette_data.append((n_clusters, silhouette_avg))

    # Menentukan jumlah kluster terbaik berdasarkan skor silhouette tertinggi
    optimal_clusters = cluster_range[np.argmax(silhouette_scores)]

    # Sekarang gunakan jumlah kluster terbaik untuk FCM
    fcm = FCM(n_clusters=optimal_clusters, max_iter=100, m=2, error=0.01)
    fcm.fit(standardized_data)

    # Ambil pusat kluster (centroid)
    fcm_centers = fcm.centers.tolist()

    # Plot Skor Silhouette
    plt.figure()
    plt.plot(cluster_range, silhouette_scores, marker='o')
    plt.xlabel('Jumlah Kluster')
    plt.ylabel('Skor Silhouette')
    plt.title('Skor Silhouette untuk berbagai jumlah kluster')

    # Simpan plot ke dalam buffer
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()

    # Kirim plot, centroid kluster, dan data silhouette ke template
    return render_template('proseslanjut.html', plot_url=plot_url, optimal_clusters=optimal_clusters, fcm_centers=fcm_centers, silhouette_data=silhouette_data)
closest_events = []
@app.route('/hasil')
@login_required
def hasil():
    global closest_events
    # Setel seed untuk hasil yang konsisten
    np.random.seed(42)

    # Ambil parameter halaman dari URL, default ke halaman 1
    page = request.args.get('page', 1, type=int)
    
    # Batas per halaman
    per_page = 100

    # Ambil data untuk tampilan dengan paginasi
    dataview = Earthquake.query.with_entities(
        Earthquake.id,
        Earthquake.tanggal,
        Earthquake.waktu,
        Earthquake.latitude, 
        Earthquake.longitude, 
        Earthquake.magnitude, 
        Earthquake.depth
    ).paginate(page=page, per_page=per_page, error_out=False)

    # Ambil semua data untuk proses clustering dan tampilan centroid
    earthquakes = Earthquake.query.with_entities(
        Earthquake.id,
        Earthquake.tanggal,
        Earthquake.waktu,
        Earthquake.latitude,
        Earthquake.longitude,
        Earthquake.magnitude,
        Earthquake.depth
    ).all()

    # Konversi data menjadi format array numpy
    data_array = np.array([(e.latitude, e.longitude, e.magnitude, e.depth) for e in earthquakes])

    # Inisialisasi MinMaxScaler
    scaler = MinMaxScaler()

    # Normalisasi data
    normalized_data = scaler.fit_transform(data_array)

    # Inisialisasi model Fuzzy C-Means dengan seed yang sama
    fcm = FCM(n_clusters=optimal_clusters, max_iter=100, m=2, error=0.01, random_state=42)
    
    # Fit data ke model
    fcm.fit(normalized_data)

    # Dapatkan keanggotaan untuk setiap titik data
    membership_matrix = fcm.u

    # Tetapkan setiap titik data ke cluster dengan keanggotaan tertinggi
    labels = np.argmax(membership_matrix, axis=1) + 1

    # Gabungkan data dengan label cluster untuk tampilan
    labeled_data = list(zip(dataview.items, labels))

    # Simpan label ke tabel ClusteredEarthquake jika belum disimpan
    for i, label in enumerate(labels):
        earthquake_id = earthquakes[i].id
        existing_cluster = ClusteredEarthquake.query.filter_by(earthquake_id=earthquake_id).first()
        if existing_cluster is None:
            new_cluster = ClusteredEarthquake(
                earthquake_id=earthquake_id, 
                cluster_label=int(label)
            )
            db.session.add(new_cluster)

    # Commit perubahan ke database
    db.session.commit()

    # Hitung jumlah untuk setiap label cluster
    label_counts = db.session.query(
        ClusteredEarthquake.cluster_label, 
        db.func.count(ClusteredEarthquake.id).label('jumlah')
    ).group_by(ClusteredEarthquake.cluster_label).all()

    # Hitung centroid
    centroids = fcm.centers

    # Hitung jarak dari setiap titik data ke setiap centroid
    distances = np.linalg.norm(normalized_data[:, np.newaxis] - centroids, axis=2)

    # Identifikasi kejadian terdekat dengan setiap centroid dan posisi mereka
    closest_events = []
    for i in range(centroids.shape[0]):
        closest_index = np.argmin(distances[:, i])
        closest_event = earthquakes[closest_index]
        table_position = (page - 1) * per_page + closest_index + 1 

        closest_events.append({
            'Centroid': f'Cluster {i + 1}',
            'No': table_position,
            'Date': closest_event.tanggal,
            'Time': closest_event.waktu,
            'Latitude': closest_event.latitude,
            'Longitude': closest_event.longitude,
            'Depth': closest_event.depth,
            'Mag': closest_event.magnitude
        })

    # Hitung min, max, dan rata-rata untuk setiap cluster
    cluster_stats = []
    for cluster_label in range(1, optimal_clusters + 1):
        stats = db.session.query(
            db.func.min(Earthquake.latitude).label('min_latitude'),
            db.func.max(Earthquake.latitude).label('max_latitude'),
            db.func.avg(Earthquake.latitude).label('mean_latitude'),
            db.func.min(Earthquake.longitude).label('min_longitude'),
            db.func.max(Earthquake.longitude).label('max_longitude'),
            db.func.avg(Earthquake.longitude).label('mean_longitude'),
            db.func.min(Earthquake.magnitude).label('min_magnitude'),
            db.func.max(Earthquake.magnitude).label('max_magnitude'),
            db.func.avg(Earthquake.magnitude).label('mean_magnitude'),
            db.func.min(Earthquake.depth).label('min_depth'),
            db.func.max(Earthquake.depth).label('max_depth'),
            db.func.avg(Earthquake.depth).label('mean_depth')
        ).join(ClusteredEarthquake, Earthquake.id == ClusteredEarthquake.earthquake_id)\
         .filter(ClusteredEarthquake.cluster_label == cluster_label).first()

        cluster_stats.append({
            'Cluster': f'Cluster {cluster_label}',
            'Min Latitude': stats.min_latitude,
            'Max Latitude': stats.max_latitude,
            'Mean Latitude': stats.mean_latitude,
            'Min Longitude': stats.min_longitude,
            'Max Longitude': stats.max_longitude,
            'Mean Longitude': stats.mean_longitude,
            'Min Magnitude': stats.min_magnitude,
            'Max Magnitude': stats.max_magnitude,
            'Mean Magnitude': stats.mean_magnitude,
            'Min Depth': stats.min_depth,
            'Max Depth': stats.max_depth,
            'Mean Depth': stats.mean_depth
        })

    # Hitung total halaman
    total_data = Earthquake.query.count()
    total_pages = ceil(total_data / per_page)

    # Render halaman hasil menampilkan cluster optimal, data yang dilabeli, dan paginasi
    return render_template(
        'hasil.html', 
        optimal=optimal_clusters, 
        labeled_data=labeled_data, 
        total_pages=total_pages, 
        current_page=page,
        label_counts=label_counts,
        closest_events=closest_events,
        cluster_stats=cluster_stats  # Kirimkan statistik cluster untuk tampilan
    )



@app.route('/hapusdatainput', methods=['POST'])
@login_required
def hapus_data():
    db.session.query(ClusteredEarthquake).delete()
    db.session.query(Earthquake).delete()
    db.session.commit()
    flash('Data berhasil dihapus!', 'error')
    return redirect(url_for('inputdata'))

@app.route('/export_excel/<int:label_id>')
@login_required
def export_excel(label_id):
    # Ambil data dari database untuk label yang diberikan dengan join ke tabel Earthquake
    data = db.session.query(
        Earthquake.id, 
        Earthquake.tanggal,
        Earthquake.waktu,
        Earthquake.latitude,
        Earthquake.longitude,
        Earthquake.magnitude,
        Earthquake.depth,
        ClusteredEarthquake.cluster_label
    ).join(ClusteredEarthquake, Earthquake.id == ClusteredEarthquake.earthquake_id) \
     .filter(ClusteredEarthquake.cluster_label == label_id).all()

    # Ubah data menjadi DataFrame
    data_list = [{
        "ID": d.id,
        "Tanggal": d.tanggal,
        "Waktu": d.waktu,
        "Latitude": d.latitude,
        "Longitude": d.longitude,
        "Magnitude": d.magnitude,
        "Depth": d.depth,
        "Cluster Label": d.cluster_label
    } for d in data]
    
    df = pd.DataFrame(data_list)

    # Simpan DataFrame ke dalam buffer Excel
    output = BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=False, sheet_name=f'Cluster_{label_id}')
    output.seek(0)

    # Kirim file Excel sebagai respons untuk diunduh
    return send_file(output, as_attachment=True, download_name=f'Cluster_{label_id}.xlsx', mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')

@app.route('/visualisasi')
@login_required
def visualisasi():
    # Membuat peta dengan Folium
    m = folium.Map(location=[closest_events[0]['Latitude'], closest_events[0]['Longitude']], zoom_start=5)

    # Tentukan warna untuk setiap cluster
    cluster_colors = {
        1: 'red',
        2: 'blue',
        3: 'green',
        4: 'purple',
        5: 'orange',
        # Tambahkan lebih banyak warna jika diperlukan
    }

    for event in closest_events:
        cluster_id = int(event['Centroid'].split()[-1])  # Mengambil nomor cluster dari 'Cluster X'
        color = cluster_colors.get(cluster_id, 'gray')  # Dapatkan warna berdasarkan nomor cluster
        folium.Marker(
            location=[event['Latitude'], event['Longitude']],
            popup=folium.Popup(f"Cluster: {event['Centroid']}<br>No: {event['No']}<br>Date: {event['Date']}<br>Time: {event['Time']}<br>Magnitude: {event['Mag']}<br>Depth: {event['Depth']}", max_width=300),
            icon=folium.Icon(color=color)
        ).add_to(m)


    # Mengembalikan peta ke template sebagai HTML
    return render_template('visualisasi.html', map=m._repr_html_())


if __name__ == "__main__":
    with app.app_context():
        db.create_all()  # Membuat tabel jika belum ada
    app.run(debug=True)
