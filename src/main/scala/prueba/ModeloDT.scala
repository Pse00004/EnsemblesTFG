package prueba

import org.apache.spark.rdd.RDD
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.tree.DecisionTree
import org.apache.spark.mllib.tree.model.DecisionTreeModel

object ModeloDT {

    def Modelo(datos: RDD[LabeledPoint], numClasses: Int, maxDepth: Int, maxBins: Int): DecisionTreeModel = {

        // Train a DecisionTree model.
        //  Empty categoricalFeaturesInfo indicates all features are continuous.
        //val numClasses = 3
        val categoricalFeaturesInfo = Map[Int, Int]()
        val impurity = "gini"
        //val maxDepth = 5
        //val maxBins = 32

        val model = DecisionTree.trainClassifier(datos, numClasses, categoricalFeaturesInfo,
            impurity, maxDepth, maxBins)

        return model
    }

    def precisionModelo(model: DecisionTreeModel, test: RDD[LabeledPoint]) {

        // Evaluate model on test instances and compute test error
        val labelAndPreds = test.map { point =>
            val prediction = model.predict(point.features)
            (point.label, prediction)
        }
        val testErr = labelAndPreds.filter(r => r._1 != r._2).count().toDouble / test.count()
        println(s"Test Error = $testErr")
        println(s"Learned classification tree model:\n ${model.toDebugString}")
    }

}