package prueba

import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.rdd.RDD
import org.apache.spark.mllib.regression.LabeledPoint


object StackingModelos {

    def Stacking(particiones: Array[RDD[LabeledPoint]], test: RDD[LabeledPoint], numParticiones: Int, args: Array[String], numModelo: Int, numAtributosDataset: Int): Array[RDD[LabeledPoint]] = {

        var RDDcombinado = Array[RDD[LabeledPoint]]()

        //Paso 2
        //Para cada partición, realizar un modelo con las demás y realizar una predicción para ella
        println("Realizando Paso 2 para modelo " + numModelo)
        for (indiceParticion <- 0 to numParticiones - 1) {

            var arrayGruposParticiones = Array[RDD[LabeledPoint]]()

            for (j <- 0 to numParticiones - 1) {

                if (indiceParticion != j) {
                    arrayGruposParticiones :+= particiones.apply(j)
                }
            }

            //Combinar resultados de los grupos de particiones
            var combinacionGruposParticiones = arrayGruposParticiones.apply(0)
            for (k <- 1 to numParticiones - 2) {
                combinacionGruposParticiones = combinacionGruposParticiones.union(arrayGruposParticiones.apply(k))
            }

            //Realizar muestreo
            val probabilidadPasarAtributo = 0.75
            val r = scala.util.Random
            var arrayAtributosPasar = Array[Boolean]()

            for (indiceAtributo <- 0 to numAtributosDataset - 1) {

                val numeroAleatorio = r.nextFloat()

                if (probabilidadPasarAtributo > numeroAleatorio) {
                    arrayAtributosPasar :+= true
                } else {
                    arrayAtributosPasar :+= false
                }
            }

            combinacionGruposParticiones = combinacionGruposParticiones.map { x =>

                var arrayValores = Array[Double]()

                for (indiceAtributo <- 0 to x.features.size - 1) {

                    if (arrayAtributosPasar.apply(indiceAtributo) == true) {
                        arrayValores :+= x.features.apply(indiceAtributo)
                    } else {
                        arrayValores :+= 0d
                    }
                }
                LabeledPoint(x.label, Vectors.dense(arrayValores))
            }

            //println("Numero instancias: " + combinacionParticiones.count())

            args.apply(0) match {
                case "NB" => {
                    val modeloGrupoParticiones = ModeloNaiveBayes.Modelo(combinacionGruposParticiones, args.apply(1).toFloat)

                    val particionTratada = particiones.apply(indiceParticion).map({ case LabeledPoint(v1, v2) => v2 })
                    val prediccionParticion = modeloGrupoParticiones.predict(particionTratada)
                    val particionDevolver = prediccionParticion.zip(particionTratada).map({ case (v1, v2) => LabeledPoint(v1, v2) })
                    RDDcombinado :+= particionDevolver
                }
                case "LR" => {
                    val modeloGrupoParticiones = ModeloLR.Modelo(combinacionGruposParticiones, args.apply(1).toInt)

                    val particionTratada = particiones.apply(indiceParticion).map({ case LabeledPoint(v1, v2) => v2 })
                    val prediccionParticion = modeloGrupoParticiones.predict(particionTratada)
                    val particionDevolver = prediccionParticion.zip(particionTratada).map({ case (v1, v2) => LabeledPoint(v1, v2) })
                    RDDcombinado :+= particionDevolver
                }
                case "DT" => {
                    val modeloGrupoParticiones = ModeloDT.Modelo(combinacionGruposParticiones, args.apply(1).toInt, args.apply(2).toInt, args.apply(3).toInt)

                    val particionTratada = particiones.apply(indiceParticion).map({ case LabeledPoint(v1, v2) => v2 })
                    val prediccionParticion = modeloGrupoParticiones.predict(particionTratada)
                    val particionDevolver = prediccionParticion.zip(particionTratada).map({ case (v1, v2) => LabeledPoint(v1, v2) })
                    RDDcombinado :+= particionDevolver
                }
            }

        }

        //Paso 3
        //Combinar los resultados de todas las particiones
        println("Realizando Paso 3 para " + numModelo)
        var RDDprediccionesTraining = RDDcombinado.apply(0)

        for (i <- 1 to numParticiones - 2) {
            RDDprediccionesTraining = RDDprediccionesTraining.union(RDDcombinado.apply(i))
        }

        //Realizar modelo
        //Paso 4
        //Realizar predicciones para el test dataset
        args.apply(0) match {
            case "NB" => {
                val modelo = ModeloNaiveBayes.Modelo(RDDprediccionesTraining, args.apply(1).toFloat)

                println("Realizando Paso 4 para " + numModelo)
                val testLimpio = test.map({ case LabeledPoint(v1, v2) => v2 })
                val prediccion = modelo.predict(testLimpio)
                val RDDprediccionesTest = prediccion.zip(testLimpio).map({ case (v1, v2) => LabeledPoint(v1, v2) })
                Array(RDDprediccionesTraining, RDDprediccionesTest)
            }
            case "LR" => {
                val modelo = ModeloLR.Modelo(RDDprediccionesTraining, args.apply(1).toInt)

                println("Realizando Paso 4 para " + numModelo)
                val testLimpio = test.map({ case LabeledPoint(v1, v2) => v2 })
                val prediccion = modelo.predict(testLimpio)
                val RDDprediccionesTest = prediccion.zip(testLimpio).map({ case (v1, v2) => LabeledPoint(v1, v2) })
                Array(RDDprediccionesTraining, RDDprediccionesTest)
            }
            case "DT" => {
                val modelo = ModeloDT.Modelo(RDDprediccionesTraining, args.apply(1).toInt, args.apply(2).toInt, args.apply(3).toInt)

                println("Realizando Paso 4 para " + numModelo)
                val testLimpio = test.map({ case LabeledPoint(v1, v2) => v2 })
                val prediccion = modelo.predict(testLimpio)
                val RDDprediccionesTest = prediccion.zip(testLimpio).map({ case (v1, v2) => LabeledPoint(v1, v2) })
                Array(RDDprediccionesTraining, RDDprediccionesTest)
            }
        }
    }
}