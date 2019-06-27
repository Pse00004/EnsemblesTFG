package prueba

import java.io.{File, PrintWriter}
import org.apache.spark.SparkConf
import org.apache.spark.SparkContext
import org.apache.spark.rdd.RDD
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint

object MainBagging {

    def main(args: Array[String]) {

        val tiempoInicioPrograma = System.nanoTime

        val conf = new SparkConf().setAppName("ProyectoTFG")
        val sc = new SparkContext(conf)
        sc.setLogLevel("ERROR")

        var argumentosCorrectos = false
        var ficheroEntrada = ""
        var ficheroSalida = ""
        var numParticiones = -1
        var modelosLvl0 = Array[Array[String]]()

        if (args.length == 0) {
            val usage =
                """Uso: ficheroEntrada ficheroSalida numParticiones
    -l0 [modelo] [argumentos_modelo]
        modelos a utilizar en el nivel 0, se debe repetir para introducir varios modelos

-------------------------------------------------------------------------------------------

    Modelos disponibles y sus parámetros:
    LR: logistic regression
        numClasses
    NB: naive bayes
        lambda
    DT: decision tree
        numClasses, maxDepth, maxBins

-------------------------------------------------------------------------------------------

     Los modelos se deben introducir mediante sus iniciales

     Ejemplo de uso:
          C:/iris.dat C:/resultados.txt 4 -l0 NB 1.0 -l0 NB 1.0 -l0 LR 10 -l0 LR 10 -l0 DT 3 5 32"""

            println(usage)

        }
        else {

            ficheroEntrada = args.apply(0)
            ficheroSalida = args.apply(1)
            numParticiones = args.apply(2).toInt

            var nivel = -1
            var contador = 0
            var indiceModelosLvl0 = -1

            for (i <- 3 to args.length - 1) {

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
                println("Fichero de salida especificado: " + ficheroSalida)
                println("Número de particiones especificado " + numParticiones)

                println("Modelos de nivel 0 especificados: ")
                for (i <- 0 to modelosLvl0.length - 1) {
                    for (j <- 0 to modelosLvl0.apply(i).length - 1) {
                        print(modelosLvl0.apply(i).apply(j) + " ")
                    }
                    println("")
                }
                println("-----")
                argumentosCorrectos = true
            }
        }

        if (argumentosCorrectos == true) {

            val DS = new DataSet()
            DS.loadDataSet(ficheroEntrada, sc, numParticiones)

            val instancias = DS.getInstances
            val RDDdeLabeledPoint = instancias.map { x => LabeledPoint(DS.vectorToDouble(x._2), Vectors.dense(x._1.toArray)) }

            // Dividir dataset en training y test
            val splits = RDDdeLabeledPoint.randomSplit(Array(0.6, 0.4))
            val training: RDD[LabeledPoint] = splits(0)
            training.cache()
            val test: RDD[LabeledPoint] = splits(1)

            val tiempoInicioEjecucion = System.nanoTime


            val valoresTest = test.map({ case LabeledPoint(v1, v2) => v2 })
            var arrayCombinacionPredicciones = Array[Array[Double]]()

            for (k <- 0 to modelosLvl0.length - 1) {

                println("Creando modelo " + k)

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
                        val modelo = ModeloDT.Modelo(subsetTraining, modelosLvl0.apply(k).apply(1).toInt, modelosLvl0.apply(k).apply(2).toInt, modelosLvl0.apply(k).apply(3).toInt)
                        predicciones = modelo.predict(valoresTest)
                    }
                }
                arrayCombinacionPredicciones :+= predicciones.take(test.count().toInt)
            }

            val numAtributos = DS.getnOutput

            var arrayAtributos = Array[Double]()
            var arrayVeces = Array[Int]()

            var arrayPrediccionesFinal = Array[Double]()

            for (i <- 0 to numAtributos - 1) {
                arrayAtributos :+= i.toDouble
                arrayVeces :+= 0
            }

            for (instancia <- 0 to test.count().toInt - 1) {

                for (indiceBootstrap <- 0 to modelosLvl0.length - 1) {

                    for (indiceAtributo <- 0 to numAtributos - 1) {
                        if (arrayCombinacionPredicciones.apply(indiceBootstrap).apply(instancia) == arrayAtributos.apply(indiceAtributo)) {
                            arrayVeces(indiceAtributo) = arrayVeces(indiceAtributo) + 1
                        }
                    }
                }

                var claseMasComun = 0d
                var vecesMasComun = 0

                for (i <- 0 to numAtributos - 1) {

                    if (arrayVeces(i) > vecesMasComun) {
                        claseMasComun = arrayAtributos(i)
                        vecesMasComun = arrayVeces(i)
                    }
                    arrayVeces(i) = 0
                }

                arrayPrediccionesFinal :+= claseMasComun
            }

            val clasesTest = test.map({ case LabeledPoint(v1, v2) => v1 }).take(test.count().toInt)
            var prediccionesCorrectas = 0

            for (i <- 0 to test.count().toInt - 1) {
                if (arrayPrediccionesFinal(i) == clasesTest(i)) {
                    prediccionesCorrectas = prediccionesCorrectas + 1
                }
            }

            val precision = prediccionesCorrectas.toDouble / test.count()

            println("Precisión final: " + (math rint precision * 100) / 100)

            val duration = (System.nanoTime - tiempoInicioPrograma) / 1e9d
            println("Tiempo desde el comienzo del programa: " + (math rint duration * 100) / 100 + "s")
            val duration2 = (System.nanoTime - tiempoInicioEjecucion) / 1e9d
            println("Tiempo de ejecución: " + (math rint duration2 * 100) / 100 + "s")

            val pw = new PrintWriter(new File(ficheroSalida))

            pw.write("Fichero de entrada especificado: " + ficheroEntrada + "\n")
            pw.write("Número de particiones especificado " + numParticiones + "\n")

            pw.write("Modelos de nivel 0 especificados: " + "\n")
            for (i <- 0 to modelosLvl0.length - 1) {
                for (j <- 0 to modelosLvl0.apply(i).length - 1) {
                    pw.write(modelosLvl0.apply(i).apply(j) + " ")
                }
                pw.write("\n")
            }

            pw.write("Precisión final: " + (math rint precision * 100) / 100 + "\n")
            pw.write("Tiempo desde el comienzo del programa: " + (math rint duration * 100) / 100 + "s" + "\n")
            pw.write("Tiempo de ejecución: " + (math rint duration2 * 100) / 100 + "s" + "\n")
            pw.close

            //resultados.saveAsTextFile(ficheroSalida)
            println("Fin de ejecución")
        }
    }
}