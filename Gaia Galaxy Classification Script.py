# -*- coding: utf-8 -*-
"""
Created on Wed May 20 18:15:51 2026

@author: ed
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture



#Load the dataset
file_path = "m45-clean.csv"  # Replace with the corresponding File Path

#Reading dataset and printing dataset shape
data = pd.read_csv(file_path)
print("Initial dataset shape:", data.shape)

#Scatter plot of proper motion characteristics
plt.figure(figsize=(8, 6))
sns.scatterplot(x=data["pmra"], y=data["pmdec"], alpha=0.5)
plt.xlabel("Proper Motion in RA (pmra)")
plt.ylabel("Proper Motion in Dec (pmdec)")
plt.title("Proper Motion of Stars")
plt.show()

###################### Part 1 ##################################

#Adding filtering criteria
data_filtered = data[
    (data["parallax"] > 0) &  #Positive parallaxes only
    ((data["parallax_error"] / data["parallax"]) <= 0.2) &  #Relative error 20%
    (data["pmra"].between(-50, 50)) &  #Removing extreme proper motion in RA
    (data["pmdec"].between(-80, 50))]   #Removing extreme proper motion in Dec 

#Scatter plot with colour by parallax
plt.figure(figsize=(8, 6))

#Create the scatter plot using parallax to determine the color of each point 
scatter = sns.scatterplot(x=data_filtered["pmra"], y=data_filtered["pmdec"], 
                          hue=data_filtered["parallax"], palette="viridis", alpha=0.7)

#Adding labels and title
plt.xlabel("Proper Motion in RA (pmra)")
plt.ylabel("Proper Motion in Dec (pmdec)")
plt.title("Proper Motion of Stars")

#Get the current axes to adjust colour bar
ax = plt.gca()

#Manually set the colour bar based on parallax data range
norm = plt.Normalize(vmin=data_filtered["parallax"].min(), vmax=data_filtered["parallax"].max())
sm = plt.cm.ScalarMappable(cmap="viridis", norm=norm)
sm.set_array([])  #Empty array because it doesn't need data for the colour map


#Add the colour bar and label it
cbar = plt.colorbar(sm, ax=ax)
cbar.set_label("Parallax (as)")

#Show the plot
plt.show()


#Filter the dataset for parallax between 6 and 14
data_parallax_filtered = data_filtered[(data_filtered["parallax"] >= 6) & (data_filtered["parallax"] <= 14)]

#Scatter plot of proper motion with filtered data parralax between 6 and 14
plt.figure(figsize=(8, 6))
sns.scatterplot(x=data_parallax_filtered["pmra"], y=data_parallax_filtered["pmdec"], alpha=0.7)
plt.xlabel("Proper Motion in RA (pmra)")
plt.ylabel("Proper Motion in Dec (pmdec)")
plt.title("Proper Motion of Stars (Parallax between 6 and 14)")
plt.show()

#Remove NaN values from the original filtered data
data_filtered = data_filtered.dropna()

############## Part 2 ###############################


#Scaling all features
features_to_scale = data_filtered.select_dtypes(include=["number"]).columns
scaler = StandardScaler()
scaled_data = scaler.fit_transform(data_filtered[features_to_scale])

#Creating a new DataFrame with the same column names and scaled values
data_scaled = pd.DataFrame(scaled_data, columns=features_to_scale)


############ Part 3a (K-means)###############################

#Select only the motion columns for clustering
data_for_clustering = data_scaled[["pmra", "pmdec"]]  

#KMeans Clustering, variable n_clusters
kmeans = KMeans(n_clusters=10, random_state=42)  
kmeans_labels = kmeans.fit_predict(data_for_clustering)

#Scatter plot with KMeans labels
plt.figure(figsize=(8, 6))
sns.scatterplot(x=data_for_clustering["pmra"], y=data_for_clustering["pmdec"], hue=kmeans_labels, palette="viridis", alpha=0.7)
plt.xlabel("Proper Motion in RA (pmra)")
plt.ylabel("Proper Motion in Dec (pmdec)")
plt.title("KMeans Clustering of Stars")
plt.legend(title="Cluster", loc='upper left')
plt.show()

############ Part 3b (Gaussian)###############################

#Gaussian Mixture Model Clustering with variable n_component
gmm = GaussianMixture(n_components=4, random_state=42)  
gmm_labels = gmm.fit_predict(data_for_clustering)

#Scatter plot with GMM labels
plt.figure(figsize=(8, 6))
sns.scatterplot(x=data_for_clustering["pmra"], y=data_for_clustering["pmdec"], hue=gmm_labels, palette="viridis", alpha=0.7)
plt.xlabel("Proper Motion in RA (pmra)")
plt.ylabel("Proper Motion in Dec (pmdec)")
plt.title("Gaussian Mixture Model Clustering of Stars")
plt.legend(title="Cluster", loc='upper left')
plt.show()

################ Part 3c (Interesting cluster) ##########################



#Filtering the interesting cluster, variable interesting_cluster
interesting_cluster = 1
interesting_stars = data_filtered[gmm_labels == interesting_cluster].copy()




#Checking the shape of the interesting cluster
print("Number of stars in the interesting cluster:", interesting_stars.shape[0])

#eScatter plot of the interesting stars only
plt.figure(figsize=(8, 6))
sns.scatterplot(x=interesting_stars["pmra"], y=interesting_stars["pmdec"], alpha=0.7)
plt.xlabel("Proper Motion in RA (pmra)")
plt.ylabel("Proper Motion in Dec (pmdec)")
plt.title("Interesting Cluster of Stars (GMM)")
plt.show()

print(f"Number of stars in the interesting cluster: {interesting_stars.shape[0]}")

#################### Part 4 ###############################


#Select the features for PCA
features_to_use = data_filtered.select_dtypes(include=["number"]).columns
#Standardize the data 
scaler = StandardScaler()
scaled_interesting_stars = scaler.fit_transform(interesting_stars[features_to_use])

#Compute the covariance matrix
cov_matrix = np.cov(scaled_interesting_stars, rowvar=False)

#Perform eigenvalue decomposition to get eigenvalues and eigenvectors
eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

#Sort the eigenvalues and eigenvectors in descending order
sorted_indices = np.argsort(eigenvalues)[::-1]
eigenvalues_sorted = eigenvalues[sorted_indices]
eigenvectors_sorted = eigenvectors[:, sorted_indices]

#The principal components are the eigenvectors corresponding to the largest eigenvalues
PC1 = eigenvectors_sorted[:, 0]  #First principal component
PC2 = eigenvectors_sorted[:, 1]  #Second principal component

#The amount of variance explained by each component
variance_explained = eigenvalues_sorted / np.sum(eigenvalues_sorted)

#Print the results
print("Eigenvalues:", eigenvalues_sorted)
print("Eigenvectors (PC1, PC2):")
print("PC1:", PC1)
print("PC2:", PC2)
print(f"Variance explained by PC1: {variance_explained[0]:.4f}")
print(f"Variance explained by PC2: {variance_explained[1]:.4f}")
print(f"Total variance explained by PC1 and PC2: {np.sum(variance_explained[:2]):.4f}")

#Plotting the eigenvalues (PC's) and their contribution to variance
sns.lineplot(x=range(1, len(eigenvalues_sorted) + 1), y=eigenvalues_sorted, markers='o')
plt.title('Principal Components and their Contribution to Variance')
plt.xlabel('Principal Component')
plt.ylabel('Eigenvalue')
plt.grid(True)
plt.show()

############################ Part 5 #################################



#Projecting the standardized interesting stars data onto the first two principal components
pc1_values = np.dot(scaled_interesting_stars, PC1)
pc2_values = np.dot(scaled_interesting_stars, PC2)

#Create a DataFrame for the projected data
pca_df = pd.DataFrame({'PC1': pc1_values, 'PC2': pc2_values})

#Plot the 2D scatter plot of PC1 and PC2
plt.figure(figsize=(8, 6))
sns.scatterplot(x=pca_df["PC1"], y=pca_df["PC2"], alpha=0.7)
plt.xlabel("Principal Component 1 (PC1)")
plt.ylabel("Principal Component 2 (PC2)")
plt.title("Projection of Data onto PC1 and PC2")
plt.show()

#Compute the loadings for each feature for both PC1 and PC2
loadings_pc1 = eigenvectors_sorted[:, 0] * np.sqrt(eigenvalues_sorted[0])
loadings_pc2 = eigenvectors_sorted[:, 1] * np.sqrt(eigenvalues_sorted[1])

#Create a DataFrame for the loadings
loadings_df = pd.DataFrame({'Feature': features_to_use, 'PC1 Loadings': loadings_pc1, 'PC2 Loadings': loadings_pc2})

#Get top 3 features for PC1
top_pc1 = loadings_df.loc[:, ['Feature', 'PC1 Loadings']]
top_pc1 = top_pc1.iloc[top_pc1['PC1 Loadings'].abs().argsort()[::-1][:3]].reset_index(drop=True)

#Get top 3 features for PC2
top_pc2 = loadings_df.loc[:, ['Feature', 'PC2 Loadings']]
top_pc2 = top_pc2.iloc[top_pc2['PC2 Loadings'].abs().argsort()[::-1][:3]].reset_index(drop=True)

#Print clean results
print("\nTop 3 features dominating PC1:")
print(top_pc1)

print("\nTop 3 features dominating PC2:")
print(top_pc2)


#Calculate the colour index bp - rp for the x-axis
interesting_stars.loc[:, "colour_index"] = interesting_stars["bp"] - interesting_stars["rp"]

#Choose the y-axis feature based on PCA
y_axis_feature = "g" 

#Scatter plot using colour index bp - rp for x and g 
plt.figure(figsize=(8, 6))
sns.scatterplot(x=interesting_stars["colour_index"], y=interesting_stars[y_axis_feature], alpha=0.7)
plt.xlabel("Colour Index (bp - rp)")
plt.ylabel(f"{"G-Band Magnitude"}")
plt.title(f"{"G-Band Magnitude"} vs. Colour Index")

#Invert the y-axis to simulate the Hertzsprung-Russell diagram 
plt.gca().invert_yaxis()

plt.show()


#Print the loadings for all features
print("\nFull Loadings Table for PC1 and PC2:")
print(loadings_df.to_string(index=False))




