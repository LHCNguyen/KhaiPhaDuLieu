import tkinter as tk
from tkinter import filedialog, messagebox
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, MeanShift, DBSCAN
from sklearn.preprocessing import StandardScaler
import numpy as np

# Đọc dữ liệu từ file
def read_data(filename):
    data = []
    try:
        with open(filename, 'r') as f:
            lines = f.readlines()
            for line in lines:
                try:
                    x, y = map(float, line.strip().split(','))
                    data.append([x, y])
                except ValueError:
                    continue
    except FileNotFoundError:
        messagebox.showerror("Error", "File not found!")
    return np.array(data)

# Hiển thị dữ liệu trên mặt phẳng
def plot_data(data):
    plt.figure(figsize=(8, 6))
    plt.scatter(data[:, 0], data[:, 1], color='blue')
    plt.title('Biểu đồ dữ liệu các điểm')
    plt.xlabel('Tọa độ x')
    plt.ylabel('Tọa độ y')
    plt.grid(True)
    plt.show()

# Thực hiện KMeans clustering
def perform_kmeans(data, num_clusters):
    kmeans = KMeans(n_clusters=num_clusters, random_state=0)
    kmeans.fit(data)
    centroids = kmeans.cluster_centers_
    labels = kmeans.labels_
    return centroids, labels

# Thực hiện Mean-Shift clustering
def perform_meanshift(data):
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data)
    meanshift = MeanShift()
    meanshift.fit(scaled_data)
    centroids = scaler.inverse_transform(meanshift.cluster_centers_)
    labels = meanshift.labels_
    return centroids, labels

# Thực hiện DBScan clustering
def perform_dbscan(data, eps, min_samples):
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    dbscan.fit(data)
    labels = dbscan.labels_
    return labels

# Hiển thị kết quả clustering
def plot_clusters(data, centroids, labels):
    plt.figure(figsize=(8, 6))
    colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k']
    for i in range(len(centroids)):
        cluster_points = np.array([data[j] for j in range(len(data)) if labels[j] == i])
        plt.scatter(cluster_points[:, 0], cluster_points[:, 1], color=colors[i % len(colors)], label=f'Cluster {i+1}')
    plt.scatter(centroids[:, 0], centroids[:, 1], color='black', marker='x', label='Centroids')
    plt.title('Biểu đồ các cụm sau khi áp dụng thuật toán')
    plt.xlabel('Tọa độ x')
    plt.ylabel('Tọa độ y')
    plt.legend()
    plt.grid(True)
    plt.show()

# Xử lý sự kiện khi nhấn nút Clustering
def on_clustering():
    algorithm = var_algorithm.get()
    if algorithm == "KMeans":
        num_clusters = int(entry_num_clusters.get())
        centroids, labels = perform_kmeans(data, num_clusters)
    elif algorithm == "MeanShift":
        centroids, labels = perform_meanshift(data)
    elif algorithm == "DBScan":
        eps = float(entry_eps.get())
        min_samples = int(entry_min_samples.get())
        labels = perform_dbscan(data, eps, min_samples)
    else:
        messagebox.showerror("Error", "Unknown algorithm!")
        return

    plot_clusters(data, centroids, labels)

# Xử lý sự kiện khi nhấn nút Open File
def on_open_file():
    global data
    filename = filedialog.askopenfilename(filetypes=[("Text files", "*.txt")])
    if filename:
        data = read_data(filename)
        if len(data) > 0:
            plot_data(data)

# Tạo giao diện đồ họa
root = tk.Tk()
root.title("Clustering App")

# Khởi tạo biến global
data = np.array([])

# Frame chứa các phần điều khiển
frame_controls = tk.Frame(root, padx=10, pady=10)
frame_controls.pack()

# Nút Open File
btn_open_file = tk.Button(frame_controls, text="Open File", command=on_open_file)
btn_open_file.grid(row=0, column=0, padx=5, pady=5)

# Label và Entry cho số cụm (KMeans)
label_num_clusters = tk.Label(frame_controls, text="Number of clusters (KMeans):")
label_num_clusters.grid(row=1, column=0, padx=5, pady=5)
entry_num_clusters = tk.Entry(frame_controls)
entry_num_clusters.grid(row=1, column=1, padx=5, pady=5)

# Label và Entry cho eps (DBScan)
label_eps = tk.Label(frame_controls, text="Eps (DBScan):")
label_eps.grid(row=2, column=0, padx=5, pady=5)
entry_eps = tk.Entry(frame_controls)
entry_eps.grid(row=2, column=1, padx=5, pady=5)

# Label và Entry cho min_samples (DBScan)
label_min_samples = tk.Label(frame_controls, text="Min samples (DBScan):")
label_min_samples.grid(row=3, column=0, padx=5, pady=5)
entry_min_samples = tk.Entry(frame_controls)
entry_min_samples.grid(row=3, column=1, padx=5, pady=5)

# Dropdown cho thuật toán Clustering
label_algorithm = tk.Label(frame_controls, text="Choose algorithm:")
label_algorithm.grid(row=4, column=0, padx=5, pady=5)
var_algorithm = tk.StringVar(root)
var_algorithm.set("KMeans")  # Mặc định là KMeans
dropdown_algorithm = tk.OptionMenu(frame_controls, var_algorithm, "KMeans", "MeanShift", "DBScan")
dropdown_algorithm.grid(row=4, column=1, padx=5, pady=5)

# Nút Clustering
btn_clustering = tk.Button(frame_controls, text="Run Clustering", command=on_clustering)
btn_clustering.grid(row=5, column=0, columnspan=2, padx=5, pady=10)

# Hiển thị giao diện
root.mainloop()
