package prueba

import org.apache.spark.SparkConf
import org.apache.spark.SparkContext
import org.apache.spark.rdd.RDD
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint

object MainBagging {

    def main(args: Array[String]) {

        val t1 = System.nanoTime

        val conf = new SparkConf().setAppName("ProyectoTFG").setMaster("local")
        val sc = new SparkContext(conf)
        sc.setLogLevel("ERROR")

        var argumentosCorrectos = false
        var ficheroEntrada = ""
        var ficheroSalida = ""
        var modelosLvl0 = Array[Array[String]]()

        if (args.length == 0) {
            val usage =
                """Uso: ficheroEntrada carpetaSalida
    -l0 [modelo] [argumentos_modelo]
        modelos a utilizar en el nivel 0, se debe repetir para introducir varios modelos

-------------------------------------------------------------------------------------------

    Modelos disponibles y sus parámetros:
    LR: logistic regression
        numClasses
    NB: naive bayes
        lambda

-------------------------------------------------------------------------------------------

    La carpeta de salida debe estar vacía
    Los modelos se deben introducir mediante sus iniciales

    Ejemplo de uso:
        C:/iris.csv C:/resultados -l0 NB 1.0 -l0 SVM 100"""

            println(usage)

        }
        else {

            ficheroEntrada = args.apply(0)
            ficheroSalida = args.apply(1)

            var nivel = -1
            var contador = 0
            var indiceModelosLvl0 = -1

            for (i <- 2 to args.length - 1) {

                contador += 1

                if (args.apply(i) == "-l0") {
                    nivel = 0
                    contador = 0
                    indiceModelosLvl0 += 1
                }

                if ((nivel == 0) && (args.apply(i) != "-l0")) {
                    if (contador == 1) {
                        modelosLvl0 :+= Array(args.apply(i))
                    } else {
                        modelosLvl0.apply(indiceModelosLvl0) :+= args.apply(i)
                    }

                }
            }

            if (modelosLvl0.length < 2) {
                println("Es necesario introducir 2 o más modelos de nivel 0")
            } else {
                println("Fichero de entrada especificado: " + ficheroEntrada)
                println("Carpeta de salida especificada: " + ficheroSalida)

                println("Modelos de nivel 0 especificados: ")
                for (i <- 0 to modelosLvl0.length - 1) {
                    for (j <- 0 to modelosLvl0.apply(i).length - 1) {
                        print(modelosLvl0.apply(i).apply(j) + " ")
                    }
                    println("")
                }
                argumentosCorrectos = true
            }
        }

        if (argumentosCorrectos == true) {

            //val lines = sc.textFile(ficheroEntrada)
            //val lines = sc.textFile("C:/Users/Pls/Desktop/iris.dat")

            val DS = new DataSet()
            DS.loadDataSet(ficheroEntrada, sc)

            DS.printAttributes()
            DS.printInstances()

            val instancias = DS.getInstances

            val RDDdeLabeledPoint = instancias.map { x => LabeledPoint(DS.vectorToDouble(x._2), Vectors.dense(x._1.toArray)) }

            RDDdeLabeledPoint.collect().foreach(println)

            // Dividir dataset en training y test
            val splits = RDDdeLabeledPoint.randomSplit(Array(0.2, 0.2, 0.2, 0.2, 0.2))
            var training: RDD[LabeledPoint] = null
            var test: RDD[LabeledPoint] = null

            println(instancias.count())
            println(RDDdeLabeledPoint.count())

            for (indiceCV <- 0 to 4) {

                var trainingInstanciado = false

                for (j <- 0 to 4) {

                    if (indiceCV == j) {
                        test = splits(j)
                        println("Instancias Test: " + test.count())
                    }
                    else if (trainingInstanciado == false) {
                        training = splits(j)
                        println("Instancias Training: " + training.count())
                        trainingInstanciado = true
                    }
                    else {
                        training = training.union(splits(j))
                        println("Instancias Training: " + training.count())
                    }
                    println("CV: " + j)
                }

                training.cache()

                val valoresTest = test.map({ case LabeledPoint(v1, v2) => v2 })
                var arrayCombinacionPredicciones = Array[RDD[Double]]()

                for (k <- 0 to 4) {

                    val subsetTraining = training.sample(true, 0.2d)

                    var predicciones: RDD[Double] = null

                    modelosLvl0.apply(k).apply(0) match {
                        case "NB" => {
                            val modelo = ModeloNaiveBayes.Modelo(subsetTraining, modelosLvl0.apply(k).apply(1).toFloat)
                            predicciones = modelo.predict(valoresTest)
                        }
                        case "LR" => {
                            val modelo = ModeloLR.Modelo(subsetTraining, modelosLvl0.apply(k).apply(1).toInt)
                            predicciones = modelo.predict(valoresTest)
                        }
                        case "DT" => {
                            val modelo = ModeloDT.Modelo(subsetTraining)
                            predicciones = modelo.predict(valoresTest)
                        }
                    }
                    arrayCombinacionPredicciones :+= predicciones
                }

                println("Numero instancias test: " + test.count())

                for (l <- 0 to 4) {
                    arrayCombinacionPredicciones.apply(l).collect().foreach((e: Double) => print(e + " "))
                    println()
                }

                val duration = (System.nanoTime - t1) / 1e9d
                println("Tiempo desde el comienzo de ejecución: " + duration)
            }
            //resultados.saveAsTextFile(ficheroSalida)
            println("Finalizado correctamente")
        }
    }
}