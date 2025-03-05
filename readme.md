# Earth-like Exoplanet Identification Using Deep Learning

## Project Overview

This project implements a deep learning approach to identify exoplanets that most closely resemble Earth, using our home planet's characteristics as the benchmark. The model analyzes multiple planetary and stellar parameters to determine which exoplanets are most "Earth-like" and therefore potentially habitable.

## Key Features

- **Earth as Reference**: Uses Earth's key characteristics as the standard for comparison
- **Similarity Scoring**: Calculates a continuous measure of "Earth-likeness" based on multiple parameters
- **Feature Engineering**: Derives planetary density from mass and radius measurements
- **Deep Neural Network**: Implements a multi-layer neural network with regularization techniques
- **High Accuracy**: Achieves 97% accuracy in identifying Earth-like exoplanets

## Data and Methodology

### Earth Reference Parameters

The model uses the following Earth characteristics as reference points:
- Planetary radius (1 Earth radius)
- Planetary mass (1 Earth mass)
- Orbital period (365.25 days)
- Equilibrium temperature (255K)
- Orbital eccentricity (0.0167)
- Insolation flux (1 Earth flux)
- Semimajor axis (1 AU)
- Planetary density (5.51 g/cm³)

### Host Star Parameters
- Stellar temperature (5778K - Sun's temperature)
- Stellar radius (1 solar radius)
- Stellar mass (1 solar mass)
- Stellar metallicity (0 - solar value)

### Methodology

1. **Data Preparation**: Loads and cleans exoplanet data from NASA's exoplanet archive
2. **Feature Engineering**: Calculates planetary density and similarity scores
3. **Classification**: Labels planets as Earth-like based on similarity threshold (95th percentile)
4. **Model Training**: Implements a neural network with batch normalization and dropout
5. **Evaluation**: Assesses model performance using accuracy, precision, and recall metrics

## Neural Network Architecture

```
Model: "sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
dense (Dense)                (None, 32)                416       
_________________________________________________________________
batch_normalization (BatchNo (None, 32)                128       
_________________________________________________________________
dense_1 (Dense)              (None, 64)                2112      
_________________________________________________________________
dropout (Dropout)            (None, 64)                0         
_________________________________________________________________
dense_2 (Dense)              (None, 64)                4160      
_________________________________________________________________
dense_3 (Dense)              (None, 1)                 65        
=================================================================
Total params: 6,881
Trainable params: 6,817
Non-trainable params: 64
```

## Results

- **Accuracy**: 97% on test data
- **Precision**: 100% for non-Earth-like planets, 97% for Earth-like planets
- **Recall**: 50% for non-Earth-like planets, 100% for Earth-like planets

## Interpretation of Results

The model shows excellent performance in identifying Earth-like planets (100% recall), meaning it doesn't miss potentially habitable candidates. The lower recall (50%) for non-Earth-like planets indicates some false positives, which is preferable to missing potentially habitable worlds. This bias toward identifying more candidates rather than missing important ones aligns with the exploratory goals of exoplanet research.

## Applications

This model can be used to:
- Screen large exoplanet databases for Earth-like candidates
- Prioritize targets for follow-up observations with limited telescope time
- Identify potentially habitable worlds for more detailed study
- Support mission planning for future space telescopes focused on exoplanet characterization

## Data Source

The model was trained on data from the NASA Exoplanet Archive, specifically the Planetary Systems Composite Data table which combines information from multiple detection methods and sources.

## Requirements

- Python 3.x
- TensorFlow 2.x
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn

## Usage

1. Clone the repository
2. Install required packages: `pip install -r requirements.txt`
3. Run the Jupyter notebook: `jupyter notebook earth-like-exoplanets-ml.ipynb`

## Future Work

- Incorporate additional exoplanet parameters as they become available
- Explore more complex similarity metrics
- Implement ensemble methods to improve classification performance
- Develop a web interface for real-time exoplanet classification
- Extend the model to classify exoplanets into multiple habitability categories

## Author

Tyler Malka

## License

This project is proprietary work by Tyler Malka. All rights reserved.

## Acknowledgments

- NASA Exoplanet Archive for providing the dataset
- The exoplanet research community for ongoing discoveries
