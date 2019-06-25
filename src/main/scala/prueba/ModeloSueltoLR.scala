package prueba

import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.rdd.RDD

object ModeloSueltoLR {

    def main(args: Array[String]) {

        val tiempoInicioPrograma = System.nanoTime

        val conf = new SparkConf().setAppName("ProyectoTFG").setMaster("local")
        val sc = new SparkContext(conf)
        sc.setLogLevel("ERROR")

        var argumentosCorrectos = false
        var ficheroEntrada = ""
        var ficheroSalida = ""
        var numParticiones = 1

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

            println("Creando modelo")

            val subsetTraining = training.sample(true, 0.2d)

            val modelo = ModeloLR.Modelo(subsetTraining, 10)


            println("Precisión final: " + (math rint ModeloLR.precisionModelo(modelo, test) * 100) / 100)
            val duration = (System.nanoTime - tiempoInicioPrograma) / 1e9d
            println("Tiempo desde el comienzo del programa: " + (math rint duration * 100) / 100 + "s")
            val duration2 = (System.nanoTime - tiempoInicioEjecucion) / 1e9d
            println("Tiempo de ejecución: " + (math rint duration2 * 100) / 100 + "s")

            println("Fin de ejecución")
        }
    }
}