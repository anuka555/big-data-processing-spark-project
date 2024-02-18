from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.feature import StandardScaler
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

def main():
    # Initialize Spark session
    spark = SparkSession.builder \
        .appName("BigDataProcessing") \
        .getOrCreate()

    try:
        # Load dataset (replace 'data.csv' with your dataset path)
        data = spark.read.csv('data.csv', header=True, inferSchema=True)

        # Data cleaning and preprocessing
        data = data.dropna()

        # Feature engineering
        feature_cols = data.columns[:-1]  # Assuming last column is the target
        assembler = VectorAssembler(inputCols=feature_cols, outputCol='features')
        data = assembler.transform(data)

        # Scale features
        scaler = StandardScaler(inputCol='features', outputCol='scaled_features')
        scaler_model = scaler.fit(data)
        data = scaler_model.transform(data)

        # Split dataset into training and testing sets
        train_data, test_data = data.randomSplit([0.8, 0.2], seed=42)

        # Train Random Forest classifier
        rf_classifier = RandomForestClassifier(labelCol='target', featuresCol='scaled_features')
        model = rf_classifier.fit(train_data)

        # Make predictions
        predictions = model.transform(test_data)

        # Evaluate model performance
        evaluator = MulticlassClassificationEvaluator(labelCol='target', predictionCol='prediction', metricName='accuracy')
        accuracy = evaluator.evaluate(predictions)
        print("Accuracy:", accuracy)

    except Exception as e:
        print("An error occurred:", str(e))

    finally:
        # Stop Spark session
        spark.stop()

if __name__ == "__main__":
    main()
