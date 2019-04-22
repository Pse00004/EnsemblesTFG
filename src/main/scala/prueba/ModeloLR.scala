package prueba

import org.apache.spark.rdd.RDD
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.classification.{LogisticRegressionModel, LogisticRegressionWithLBFGS}
import org.apache.spark.mllib.evaluation.MulticlassMetrics

object ModeloLR {

    def Modelo(datos: RDD[LabeledPoint], numClasses: Int): LogisticRegressionModel = {

        // Run training algorithm to build the model
        val model = new LogisticRegressionWithLBFGS().setNumClasses(numClasses).run(datos)

        return model
    }

    def precisionModelo(model: LogisticRegressionModel, test: RDD[LabeledPoint]): Double = {

        // Compute raw scores on the test set.
        val predictionAndLabels = test.map {
            case LabeledPoint(label, features) =>
                val prediction = model.predict(features)
                (prediction, label)
        }

        // Get evaluation metrics.
        val metrics = new MulticlassMetrics(predictionAndLabels)
        val accuracy = metrics.accuracy
        //println("Precisión del modelo LR: " + accuracy)
        return accuracy
    }
}