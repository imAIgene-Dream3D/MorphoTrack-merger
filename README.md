# MorphoTrack-merger

MorphoTrack-Merger is a Python-based tool designed to extract morphological features from segmented images and merge them with tracking data from TrackMate. This tool is particularly useful for researchers working in bioimaging and cell tracking, as it allows for seamless integration of morphological data with trajectory information.

## Streamlit online App

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://app-starter-kit.streamlit.app)

**Key Features:**
- 2D and 3D Support: Extract morphological features from both 2D and 3D segmented images.

- TrackMate Integration: Merge extracted features with TrackMate tracking data for comprehensive analysis.

- Visualization: Visualize tracks in 2D or 3D space directly within the application.

- User-Friendly Interface: Built with Streamlit, providing an intuitive web-based interface for easy data processing and visualization.

**How It Works:**
1. *Upload Files*: Users upload a segmentation TIFF file and a TrackMate CSV file.

2. *Select Mode*: Choose between 2D or 3D processing based on the data.

3. *Process Data*: The tool extracts morphological features (e.g., volume, centroid, major/minor axis length) and merges them with the corresponding TrackMate data.

4. *Visualize and Download*: Visualize the merged tracks and download the processed data as a CSV file.

## Dependencies:
- streamlit
- numpy
- pandas
- scikit-image
- scipy
- matplotlib

## Usage:
Install the required dependencies see [requirements.txt](https://github.com/imAIgene-Dream3D/MorphoTrack-merger/blob/master/requirements.txt)


## Example Workflow:
- Input: Segmentation TIFF and TrackMate CSV.

- Processing: Extract features and merge with TrackMate data.

- Output: CSV file with merged data and visual plots of tracks.

## Contributing:
Contributions are welcome! Please fork the repository and submit a pull request with your improvements.

## License:
This project is licensed under the MIT License. See the LICENSE file for details.




