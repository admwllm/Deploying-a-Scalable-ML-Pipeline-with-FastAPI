# Model Card
Adam Lott created the model.
## Model Details
The model uses random tree from Scikit Learn version 1.5.1 with the default hyperparameters. 
## Intended Use
The intended use of this model is to predict if a person's income is above or below $50,000 based on several life factor attributes. The expected user of this model is non profit institutions to determine donor viability.
## Training Data
The data used in this model creation is sourced from UC Irvine Machine Learning Repository and titled "Census Income". https://archive.ics.uci.edu/dataset/20/census+income The entire data set is used for this model and includes 48,842 instances. The attribute "salary" is the target, and the remaining features are used as target features for predictions. Scikit Learn's OneHotEncoder is used to encode categorical values, and LabelBinarizer is used to convert categorical labes to bianry for model training. The training dataset is a result of a splitting the data with 80% of the data utilized for model training.
## Evaluation Data
The evaluation data is a result of the census data split and conists of 20% of the data. 
## Metrics
The metrics used on this model are precision, recall, and F1 scores. This model scored about a 0.74 on precision, 0.64 for recall, and 0.68 for the F1 score. 
## Ethical Considerations
The data contained within is subset extracted without great detail about how the sample was chosen from the larger census data base. The set does not contain personal information, however the fields given can still be biased in weighing within the model produciton. 
## Caveats and Recommendations
The data used for this model is quite old being a subset of the 1994 census. This should be taken into account when utilizing this model. 
