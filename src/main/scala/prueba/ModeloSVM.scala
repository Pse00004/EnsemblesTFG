package prueba

import org.apache.spark.rdd.RDD
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.classification.{ SVMModel, SVMWithSGD }
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
import org.apache.spark.mllib.util.MLUtils

object ModeloSVM {

     def Modelo(datos: RDD[LabeledPoint], numIterations: Int): SVMModel = {

          // Run training algorithm to build the model
          //val numIterations = 100
          val model = SVMWithSGD.train(datos, numIterations)

          return model
     }

     def precisionModelo(model: SVMModel, test: RDD[LabeledPoint]) {

          // Compute raw scores on the test set.
          val scoreAndLabels = test.map { point =>
               val score = model.predict(point.features)
               (score, point.label)
          }

          // Get evaluation metrics.
          val metrics = new BinaryClassificationMetrics(scoreAndLabels)
          val auROC = metrics.areaUnderROC()
          println("Modelo SVM: area under ROC = $auROC")
     }
}