# Model Card
Adam Lott created the model.
## Model Details
The model uses a random tree from Scikit-Learn version 1.5.1 with the default hyperparameters. 
## Intended Use
The intended use of this model is to predict if a person's income is above or below $50,000 based on several life factor attributes. The expected user of this model is non-profit institutions to determine donor viability.
## Training Data
The data used in this model creation is sourced from UC Irvine Machine Learning Repository and titled "Census Income". https://archive.ics.uci.edu/dataset/20/census-income The entire dataset is used for this model and includes 48,842 instances; 80% of this data is utilized for model training. The attribute "salary" is the target, and the remaining features are used as target features for predictions. Scikit-Learn's OneHotEncoder is used to encode categorical values, and LabelBinarizer is used to convert categorical labels to binary for model training. 
## Evaluation Data
The evaluation data is a result of the census data split and consists of 20% of the data. 
## Metrics
The metrics used on this model are precision, recall, and F1 scores. This model achieved a precision score of approximately 0.74, a recall score of 0.64, and an F1 score of 0.68. 
## Ethical Considerations
The data contained within is a subset extracted without great detail about how the sample was chosen from the larger census database. The individuals represented in this set are anonymized; however, the fields provided can still be biased in the model's production. 
## Caveats and Recommendations
The data used for this model is quite old, being a subset of the 1994 census. This should be taken into account when using this model; many of the economic indicators and amounts are likely to have changed. 
