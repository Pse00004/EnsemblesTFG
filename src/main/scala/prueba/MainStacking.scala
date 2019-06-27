package prueba

import java.io.{File, PrintWriter}
import org.apache.spark.SparkConf
import org.apache.spark.SparkContext
import org.apache.spark.rdd.RDD
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint

object MainStacking {

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
        var modeloLvl1 = Array[String]()

        if (args.length == 0) {
            val usage =
                """Uso: ficheroEntrada ficheroSalida numParticiones
     -l0 [modelo] [argumentos_modelo]
          modelos a utilizar en el nivel 0, se debe repetir para introducir varios modelos
     -l1 [modelo]
          modelo a utilizar en el nivel 1
                            
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
          C:/iris.csv C:/resultados.txt 4 -l0 NB 1.0 -l0 DT 3 5 32 -l1 LR 100"""

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
                if (args.apply(i) == "-l1") {
                    nivel = 1
                    contador = 0
                }

                if ((nivel == 0) && (args.apply(i) != "-l0")) {
                    if (contador == 1) {
                        modelosLvl0 :+= Array(args.apply(i))
                    } else {
                        modelosLvl0.apply(indiceModelosLvl0) :+= args.apply(i)
                    }

                } else if ((nivel == 1) && (args.apply(i) != "-l1")) {
                    if (contador == 1) {
                        modeloLvl1 = Array(args.apply(i))
                    } else {
                        modeloLvl1 :+= args.apply(i)
                    }

                }
            }

            if (modelosLvl0.length < 2) {
                println("Es necesario introducir 2 o más modelos de nivel 0")
            } else if (modeloLvl1.length == 0) {
                println("Es necesario introducir un modelo de nivel 1")
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

                println("Modelo de nivel 1 especificado: ")
                for (i <- 0 to modeloLvl1.length - 1) {
                    print(modeloLvl1.apply(i) + " ")
                }
                println()
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

            val numParticionesStacking = 10


            //Paso 1
            //Dividir el dataset en particiones
            println("Realizando Paso 1")
            val pesoParticiones = Array.fill(numParticionesStacking)(1.0)
            val arrayParticiones = training.randomSplit(pesoParticiones)

            //Paso 5
            //Repetir los pasos 2 a 4 mediante la función Stacking con varios modelos
            var arrayResultadosNivel0 = Array[Array[RDD[LabeledPoint]]]()
            for (i <- 0 to modelosLvl0.length - 1) {
                arrayResultadosNivel0 :+= StackingModelos.Stacking(arrayParticiones, test, numParticionesStacking, modelosLvl0.apply(i))
            }

            //Paso 6
            //Realizar un nuevo modelo con los resultados del training dataset en el paso anterior
            println("Realizando Paso 6")
            var combinacionTrainDatasets = arrayResultadosNivel0.apply(0).apply(0)
            for (i <- 1 to modelosLvl0.length - 1) {
                combinacionTrainDatasets = combinacionTrainDatasets.union(arrayResultadosNivel0.apply(i).apply(0))
            }

            var combinacionTestDatasets = arrayResultadosNivel0.apply(0).apply(1)
            for (i <- 1 to modelosLvl0.length - 1) {
                combinacionTestDatasets = combinacionTestDatasets.union(arrayResultadosNivel0.apply(i).apply(1))
            }

            //Paso 7
            //Realizar predicciones con los resultados del test dataset en el paso 5
            println("Realizando Paso 7")
            val modeloLR = modeloLvl1.apply(0) match {
                //case "NB" => ModeloNaiveBayes.Modelo(combinacionTrainDatasets, modeloLvl1.apply(1).toFloat)
                //case "SVM" => ModeloSVM.Modelo(combinacionTrainDatasets, modeloLvl1.apply(1).toInt)
                case "LR" => ModeloLR.Modelo(combinacionTrainDatasets, modeloLvl1.apply(1).toInt)
                //case "DT"  => ModeloDT.Modelo(combinacionTrainDatasets, modeloLvl1.apply(1).toInt, modeloLvl1.apply(2).toInt, modeloLvl1.apply(3).toInt)
            }

            //val valoresCombinacionTestDatasets = combinacionTestDatasets.map({ case LabeledPoint(v1, v2) => v2 })
            //val predicciones = modeloLR.predict(valoresCombinacionTestDatasets)
            //val resultados = valoresCombinacionTestDatasets.zip(predicciones)

            println()
            println("Precisión final: " + (math rint ModeloLR.precisionModelo(modeloLR, test) * 100) / 100)

            val duration = (System.nanoTime - tiempoInicioPrograma) / 1e9d
            println("Tiempo desde el comienzo del programa: " + (math rint duration * 100) / 100 + "s")
            val duration2 = (System.nanoTime - tiempoInicioEjecucion) / 1e9d
            println("Tiempo de ejecución: " + (math rint duration2 * 100) / 100 + "s")

            val pw = new PrintWriter(new File(ficheroSalida))

            pw.write("Fichero de entrada especificado: " + ficheroEntrada + "\n")
            pw.write("Número de particiones especificado " + numParticiones + "\n")

            pw.write("Modelos de nivel 0 especificados: "  + "\n")
            for (i <- 0 to modelosLvl0.length - 1) {
                for (j <- 0 to modelosLvl0.apply(i).length - 1) {
                    pw.write(modelosLvl0.apply(i).apply(j) + " ")
                }
                pw.write("\n")
            }

            pw.write("Modelo de nivel 1 especificado: " + "\n")
            for (i <- 0 to modeloLvl1.length - 1) {
                pw.write(modeloLvl1.apply(i) + "\n")
            }

            pw.write("Precisión final: " + (math rint ModeloLR.precisionModelo(modeloLR, test) * 100) / 100 + "\n")
            pw.write("Tiempo desde el comienzo del programa: " + (math rint duration * 100) / 100 + "s" + "\n")
            pw.write("Tiempo de ejecución: " + (math rint duration2 * 100) / 100 + "s" + "\n")
            pw.close

            //resultados.saveAsTextFile(ficheroSalida)
            println("Fin de ejecución")
        }
    }
}