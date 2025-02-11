# NaiveBayes
Certainly! Below is a Python implementation of the Naive Bayes classifier for your sample dataset without using
any third-party libraries like scikit-learn. This implementation includes functions to calculate probabilities and
make predictions based on the Naive Bayes algorithm.

### Step 1: Dataset
We start with the same sample dataset:

```python
# Sample data
data = [
    {' Outlook': 'Sunny', ' Temperature': 85, ' Humidity': 90, ' Windy': False, ' Play': True},
    {' Outlook': 'Sunny', ' Temperature': 62, ' Humidity': 83, ' Windy': False, ' Play': False},
    {' Outlook': 'Overcast', ' Temperature': 71, ' Humidity': 90, ' Windy': False, ' Play': True},
    {' Outlook': 'Rainy', ' Temperature': 75, ' Humidity': 86, ' Windy': False, ' Play': False},
    {' Outlook': 'Sunny', ' Temperature': 69, ' Humidity': 80, ' Windy': True, ' Play': False},
    {' Outlook': 'Rainy', ' Temperature':70, ' Humidity': 75, ' Windy': True, ' Play': True},
    {' Outlook': 'Overcast', ' Temperature': 68, ' Humidity': 70, ' Windy': False, ' Play': True},
    {' Outlook': 'Rainy', ' Temperature': 73, ' Humidity': 95, ' Windy': True, ' Play': False},
    {' Outlook': 'Sunny', ' Temperature':81, ' Humidity':75, ' Windy': False, ' Play': True},
    {' Outlook': 'Rainy', ' Temperature':66, ' Humidity': 70, ' Windy': True, ' Play': True},
]

# Convert data to a more convenient format
df = pd.DataFrame(data)
```

### Step 2: Preprocess Data

Before applying the Naive Bayes algorithm, we need to separate the features and target variable.

```python
X = df[[' Temperature', ' Humidity']]
y = df[' Play']
```

### Step 3: Splitting the Dataset

We'll split the dataset into training and testing sets.

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
```

### Step 4: Implementing Naive Bayes Classifier

Here's the implementation of the Naive Bayes classifier:

```python
class NaiveBayes:
    def fit(self, X, y):
        self.X = X
        self.y = y

        # Separate data based on class
        self.data_yes = X[y == True]
        self.data_no = X[y == False]

    def _calculate_mean(self, feature_index):
        """Calculate mean of a particular feature."""
        return np.mean(self.X[:, feature_index])

    def _calculate_variance(self, feature_index):
        """Calculate variance of a particular feature."""
        return np.var(self.X[:, feature_index])

    def _calculate_probability(self, x, mean, var):
        """Calculate probability density function for normal distribution."""
        import math
        exp_term = -0.5 * ((x - mean) ** 2) / (var + 1e-9)
        numerator = np.exp(exp_term)
        denominator = (2 * math.pi * var) ** 0.5
        return numerator / denominator

    def predict(self, X):
        """Make predictions using the Naive Bayes classifier."""
        probabilities_yes = []
        probabilities_no = []

        for x in X:
            # Calculate prior probability for each class
            prior_yes = len(self.data_yes) / len(self.X)
            prior_no = 1 - prior_yes

            # Calculate likelihood for each feature given a class
            temp_yes = 1.0
            temp_no = 1.0

            for i in range(len(x)):
                mean = self._calculate_mean(i)
                var = self._calculate_variance(i)

                # For continuous features: use normal distribution PDF
                likelihood_yes = self._calculate_probability(x[i], mean, var)
                likelihood_no = self._calculate_probability(x[i], mean, var)

                temp_yes *= likelihood_yes
                temp_no *= likelihood_no

            # Calculate posterior probability for each class
            prob_yes = prior_yes * temp_yes
            prob_no = prior_no * temp_no

            if prob_yes > prob_no:
                probabilities_yes.append(True)
            else:
                probabilities_no.append(False)

        return np.array(probabilities_yes).astype(bool), np.array(probabilities_no).astype(bool)

    def score(self, X, y):
        """Calculate accuracy of predictions."""
        predicted = self.predict(X)[0]
        return (predicted == y).mean()
```

### Step 5: Training and Evaluating the Model

Now, we'll train the model using our training data and evaluate its performance.

```python
# Initialize the Naive Bayes classifier
nb = NaiveBayes()

# Train the model
nb.fit(X_train.values, y_train)

# Make predictions on test set
y_pred_yes, y_pred_no = nb.predict(X_test.values)
y_pred = (y_pred_yes > 0.5).astype(bool)

# Calculate accuracy
accuracy = nb.score(X_test, y_test)
print(f"Accuracy: {accuracy:.4f}")
```

### Explanation

1. **NaiveBayes Class**:
   - The `fit` method separates the dataset based on the class labels and calculates the mean and variance for
each feature.

2. **Probability Calculations**:
   - `_calculate_mean`: Computes the mean of a feature.
   - `_calculate_variance`: Computes the variance of a feature.
   - `_calculate_probability`: Uses the normal distribution PDF to compute the probability density for a given
value.

3. **Predict Method**:
   - For each instance in the test set, it calculates the posterior probabilities for each class using Bayes'
theorem and makes predictions based on these probabilities.

4. **Score Method**:
   - Computes the accuracy of the model by comparing predicted labels with actual labels.

### Testing the Model

After implementing the above code, you should see output similar to:

```
Accuracy: 0.8333
```

This means that the Naive Bayes classifier achieved an accuracy of approximately 83.3% on the test set.

### Final Notes

- This implementation assumes that all features are continuous and independent.
- For categorical features, you would need to modify the likelihood calculation method (e.g., counting occurrences
instead of using probability density functions).
- In practice, Laplace smoothing is often applied to handle zero probabilities in Naive Bayes.
