import streamlit as st
import numpy as np
import pandas as pd
from skimage import io, measure
from scipy.spatial import cKDTree
import tempfile
import os
import matplotlib.pyplot as plt

# Function for morphological feature extraction (2D)
def morphological_features_2d(seg_path, track_path):
    labels_4d = io.imread(seg_path)
    features = []

    for t in range(labels_4d.shape[0]):
        labels_3d = labels_4d[t].astype(np.int32)
        regions = measure.regionprops(labels_3d)

        for region in regions:            
            features.append({
                'Volume': region.area,
                'Centroid_Y': region.centroid[0],
                'Centroid_X': region.centroid[1],
                'Major_Axis_Length': region.axis_major_length,
                'Minor_Axis_Length': region.axis_minor_length,
                'Elongation': region.axis_major_length / region.axis_minor_length if region.axis_minor_length != 0 else None
            })

    features_df = pd.DataFrame(features).drop_duplicates()
    
    # Load TrackMate data
    trackmate_df = pd.read_csv(track_path, skiprows=lambda x: x in [1,2,3])
    trackmate_coords = trackmate_df[['POSITION_X', 'POSITION_Y']].values
    tree = cKDTree(trackmate_coords)
    
    morph_coords = features_df[['Centroid_X', 'Centroid_Y']].values
    distances, indices = tree.query(morph_coords)

    features_df['Distance_to_Trackmate'] = distances
    matched_trackmate_data = trackmate_df.iloc[indices].reset_index(drop=True)
    merged_df = pd.concat([features_df.reset_index(drop=True), matched_trackmate_data], axis=1)
    merged_df = merged_df[merged_df['TRACK_ID'].notna()].drop_duplicates()

    return merged_df

# Function for morphological feature extraction (3D)
def morphological_features_3d(seg_path, track_path):
    labels_4d = io.imread(seg_path)
    features = []

    for t in range(labels_4d.shape[0]):
        labels_3d = labels_4d[t].astype(np.int32)
        regions = measure.regionprops(labels_3d)

        for region in regions:
            if region.area < 2:
                continue
            
            try:
                minor_axis_length = region.axis_minor_length
            except ValueError:
                minor_axis_length = np.nan
            
            features.append({
                'Volume': region.area,
                'Centroid_Z': region.centroid[0],
                'Centroid_Y': region.centroid[1],
                'Centroid_X': region.centroid[2],
                'Major_Axis_Length': region.axis_major_length,
                'Minor_Axis_Length': minor_axis_length,
                'Elongation': region.axis_major_length / minor_axis_length if minor_axis_length > 0 else np.nan
            })

    features_df = pd.DataFrame(features).drop_duplicates()

    # Load TrackMate data
    trackmate_df = pd.read_csv(track_path, skiprows=lambda x: x in [1,2,3])
    trackmate_coords = trackmate_df[['POSITION_X', 'POSITION_Y', 'POSITION_Z']].values
    tree = cKDTree(trackmate_coords)

    morph_coords = features_df[['Centroid_X', 'Centroid_Y', 'Centroid_Z']].values
    distances, indices = tree.query(morph_coords)

    features_df['Distance_to_Trackmate'] = distances
    matched_trackmate_data = trackmate_df.iloc[indices].reset_index(drop=True)
    merged_df = pd.concat([features_df.reset_index(drop=True), matched_trackmate_data], axis=1)
    merged_df = merged_df[merged_df['TRACK_ID'].notna()].drop_duplicates()

    return merged_df

# Function to plot the tracks
def plot_tracks(df, mode):
    fig, ax = plt.subplots(figsize=(8, 6)) if mode == "2D" else plt.figure(figsize=(10, 8)).add_subplot(projection='3d')

    for track_id, track in df.groupby("TRACK_ID"):
        if mode == "2D":
            ax.plot(track["Centroid_X"], track["Centroid_Y"], color="black", alpha=0.5)
        else:
            ax.plot(track["Centroid_X"], track["Centroid_Y"], track["Centroid_Z"], color="black", alpha=0.5)

    if mode == "2D":
        ax.set_xlabel("Centroid X")
        ax.set_ylabel("Centroid Y")
        ax.set_title("2D Tracks")
    else:
        ax.set_xlabel("Centroid X")
        ax.set_ylabel("Centroid Y")
        ax.set_zlabel("Centroid Z")
        ax.set_title("3D Tracks")

    st.pyplot(fig)

# Streamlit UI
st.title("Morphological Feature Extraction & Track Visualization")

# Toggle for 2D/3D mode
mode = st.segmented_control("Choose Processing Mode:", ["2D", "3D"])

# File upload
seg_file = st.file_uploader("Upload Segmentation TIFF", type=['tif'], key="seg_upload")
track_file = st.file_uploader("Upload TrackMate CSV", type=['csv'], key="track_upload")

# User-defined output filename
output_filename = st.text_input("Enter output filename (without extension)", "processed_data")

# Processing Button
process_button = st.button("Process Data")

if process_button and seg_file and track_file:
    with st.spinner("Processing..."):
        # Save uploaded files to temporary locations
        with tempfile.NamedTemporaryFile(delete=False, suffix=".tif") as temp_seg:
            temp_seg.write(seg_file.read())
            seg_path = temp_seg.name
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as temp_track:
            temp_track.write(track_file.read())
            track_path = temp_track.name

        # Process files based on selected mode
        if mode == "2D":
            merged_df = morphological_features_2d(seg_path, track_path)
        else:
            merged_df = morphological_features_3d(seg_path, track_path)

        # Delete temporary files
        os.unlink(seg_path)
        os.unlink(track_path)

        st.success("Processing completed!")

        # Show results
        st.write(merged_df.head())

        # Generate output CSV
        csv_output = merged_df.to_csv(index=False).encode()
        st.download_button("Download Processed Data", csv_output, f"{output_filename}.csv", "text/csv")

        # Visualization
        st.subheader("Tracks Visualization")
        plot_tracks(merged_df, mode)
        
