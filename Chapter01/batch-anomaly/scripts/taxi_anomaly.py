import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size':22})
import datetime
from numpy.random import MT19937
from numpy.random import RandomState, SeedSequence
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn import metrics
rs = RandomState(MT19937(SeedSequence(123456789)))

# Define simulate ride data function

def simulate_ride_distances():
    ride_dists = np.concatenate((
        10*np.random.random(size = 370),
        30*np.random.random(size = 10),
        10*np.random.random(size = 10),
        10*np.random.random(size = 10)
    ))
    
    return ride_dists
def simulate_ride_speeds():
    ride_speeds = np.concatenate((
        np.random.normal(loc = 30, scale = 5, size = 370),
        np.random.normal(loc = 30, scale = 5, size = 10),
        np.random.normal(loc = 50, scale = 10, size = 10),
        np.random.normal(loc = 15, scale = 4, size = 10)
    ))
    return ride_speeds
    
    
def simulate_ride_data():
    ride_dists = simulate_ride_distances()
    ride_speeds = simulate_ride_speeds()
    ride_times = ride_dists/ride_speeds
    
    # Assemble into DataFrame  
    df = pd.DataFrame(
        {
            'ride_dist': ride_dists, 
            'ride_time': ride_times,  
            'ride_speed': ride_speeds
        }
    )
    ride_ids = datetime.datetime.now().strftime('%Y%m%d') + df.index.astype(str)
    df['ride_id'] = ride_ids
    
    return df

def plot_cluster_results(data, labels, core_samples_mask, n_clusters_):
    fig = plt.figure(figsize = (10, 10))
    
    unique_labels = set(labels)
    colors = [plt.cm.cool(each)
              for each in np.linspace(0, 1, len(unique_labels))]
    
    for k, col in zip(unique_labels, colors):
        if k == -1:
            
            col = [0, 0, 0, 1]
            class_member_mask = (labels == k)
            xy = data[class_member_mask & core_samples_mask]
            plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor = tuple(col),
                     markerredgecoor = 'k', markersize = 14)
            xy = data[class_member_mask & ~core_samples_mask]
            plt.plot(xy[:, 0], xy[:, 1], '^', markerfacecolor = tuple(col),
                     markeredgecolor = 'k', markersize = 14)
            
            plt.xlabel('Standard Scaled Ride Dist')
            plt.ylabel('Standard Scaled Ride Time')
            plt.title('Estimated number of clusters: %d' % n_clusters_)
            plt.show()
            
        return None
    
def cluster_and_label(data, create_and_show_plot = True):
    data = StandardScaler.fit_transform(data)
    db = DBSCAN(eps = 0.3, min_samples = 10).fit(data)
    
    # find labels from clustering
    core_samples_mask = np.zeros_like(db.labels_, dtype = bool)
    core_samples_mask[db.core_sample_indices_] = True
    labels = db.labels_
    
    # number of clusters in labels, ignoring noise if present
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise_ = list(labels).count(-1)
    
    run_metadata = {
        'nClusters': n_clusters_,
        'nNoise': n_noise_,
        'silhouetteCoefficient': metrics.silhouette_score(data, labels),
        'labels': labels
    }

    if create_and_show_plot:
        plot_cluster_results(data, labels, core_samples_mask, n_clusters_)
    else:
        pass
    return run_metadata
if __name__ == '__main__':
    file_path = './data/taxi-rides/csv'
    if os.path.exists(file_path):
        df = pd.read_csv(file_path)
    else:
        df = simulate_ride_data()
        df.to_csv(file_path, index = False)
        
    X = df[['ride_dist', 'ride_time']]
    results = cluster_and_label(X, create_and_show_plot = False)
    df['label'] = results['labels']
    
    df.to_json('./output/labels.json', orient = 'records')