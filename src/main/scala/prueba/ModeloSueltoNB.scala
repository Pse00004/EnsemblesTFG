package prueba

import java.io.{File, PrintWriter}

import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.rdd.RDD
import org.apache.spark.{SparkConf, SparkContext}

object ModeloSueltoNB {

    def main(args: Array[String]) {

        val tiempoInicioPrograma = System.nanoTime

        val conf = new SparkConf().setAppName("ProyectoTFG")
        val sc = new SparkContext(conf)
        sc.setLogLevel("ERROR")

        var argumentosCorrectos = false
        var ficheroEntrada = ""
        var ficheroSalida = ""
        var numParticiones = -1

        if (args.length == 0) {
            val usage =
                """Uso: ficheroEntrada ficheroSalida numParticiones
     Ejemplo de uso:
          C:/iris.dat C:/resultados.txt 4 """

            println(usage)

        }
        else {

            ficheroEntrada = args.apply(0)
            ficheroSalida = args.apply(1)
            numParticiones = args.apply(2).toInt

            println("Fichero de entrada especificado: " + ficheroEntrada)
            println("Fichero de salida especificado: " + ficheroSalida)
            println("Número de particiones especificado " + numParticiones)
            println("-----")
            argumentosCorrectos = true
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

            println("Creando modelo Naive Bayes")

            val modelo = ModeloNaiveBayes.Modelo(training, 1.0f   )


            println("Precisión: " + (math rint ModeloNaiveBayes.precisionModelo(modelo, test) * 100) / 100)
            val duration = (System.nanoTime - tiempoInicioPrograma) / 1e9d
            println("Tiempo desde el comienzo del programa: " + (math rint duration * 100) / 100 + "s")
            val duration2 = (System.nanoTime - tiempoInicioEjecucion) / 1e9d
            println("Tiempo de ejecución: " + (math rint duration2 * 100) / 100 + "s")

            val pw = new PrintWriter(new File(ficheroSalida))

            pw.write("Fichero de entrada especificado: " + ficheroEntrada + "\n")
            pw.write("Número de particiones especificado " + numParticiones + "\n")
            pw.write("Tiempo desde el comienzo del programa: " + (math rint duration * 100) / 100 + "s" + "\n")
            pw.write("Tiempo de ejecución: " + (math rint duration2 * 100) / 100 + "s" + "\n")
            pw.close

            println("Fin de ejecución")
        }
    }
}