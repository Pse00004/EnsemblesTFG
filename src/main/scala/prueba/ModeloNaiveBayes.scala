package prueba

import org.apache.spark.rdd.RDD
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.classification.{NaiveBayes, NaiveBayesModel}

object ModeloNaiveBayes {

    def Modelo(datos: RDD[LabeledPoint], lambda: Float): NaiveBayesModel = {

        val lambda = 1.0
        val model = NaiveBayes.train(datos, lambda, modelType = "multinomial")

        return model
    }

    def precisionModelo(model: NaiveBayesModel, test: RDD[LabeledPoint]): Double = {

        val predictionAndLabel = test.map(p => (model.predict(p.features), p.label))
        val accuracy = 1.0 * predictionAndLabel.filter(x => x._1 == x._2).count() / test.count()

        return accuracy
    }

}