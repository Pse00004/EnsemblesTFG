package prueba

import org.apache.spark.rdd.RDD
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.util.MLUtils

object StackingModelos {

     def Stacking(particiones: Array[RDD[LabeledPoint]], test: RDD[LabeledPoint], numParticiones: Int, args: Array[String]): Array[RDD[LabeledPoint]] = {

          var RDDcombinado = Array[RDD[LabeledPoint]]()

          //Paso 2
          //Para cada partición, realizar un modelo con las demás y realizar una predicción para ella
          println("Realizando Paso 2 para: " + args.apply(0))
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

               //println("Numero instancias: " + combinacionParticiones.count())

               args.apply(0) match {
                    case "NB" => {
                         val modeloGrupoParticiones = ModeloNaiveBayes.Modelo(combinacionGruposParticiones, args.apply(1).toFloat)

                         val particionTratada = particiones.apply(indiceParticion).map({ case LabeledPoint(v1, v2) => v2 })
                         val prediccionParticion = modeloGrupoParticiones.predict(particionTratada)
                         val particionDevolver = prediccionParticion.zip(particionTratada).map({ case (v1, v2) => LabeledPoint(v1, v2) })
                         RDDcombinado :+= particionDevolver
                    }
                    //case "SVM" => ModeloSVM.Modelo(combinacionGruposParticiones, args.apply(1).toInt)
                    case "LR" => {
                         val modeloGrupoParticiones = ModeloLR.Modelo(combinacionGruposParticiones, args.apply(1).toInt)

                         val particionTratada = particiones.apply(indiceParticion).map({ case LabeledPoint(v1, v2) => v2 })
                         val prediccionParticion = modeloGrupoParticiones.predict(particionTratada)
                         val particionDevolver = prediccionParticion.zip(particionTratada).map({ case (v1, v2) => LabeledPoint(v1, v2) })
                         RDDcombinado :+= particionDevolver
                    }
                    case "DT" => {
                         val modeloGrupoParticiones = ModeloDT.Modelo(combinacionGruposParticiones)

                         val particionTratada = particiones.apply(indiceParticion).map({ case LabeledPoint(v1, v2) => v2 })
                         val prediccionParticion = modeloGrupoParticiones.predict(particionTratada)
                         val particionDevolver = prediccionParticion.zip(particionTratada).map({ case (v1, v2) => LabeledPoint(v1, v2) })
                         RDDcombinado :+= particionDevolver
                    }
               }

          }

          //Paso 3
          //Combinar los resultados de todas las particiones
          println("Realizando Paso 3 para: " + args.apply(0))
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

                    println("Realizando Paso 4 para: " + args.apply(0))
                    val testLimpio = test.map({ case LabeledPoint(v1, v2) => v2 })
                    val prediccion = modelo.predict(testLimpio)
                    val RDDprediccionesTest = prediccion.zip(testLimpio).map({ case (v1, v2) => LabeledPoint(v1, v2) })
                    return Array(RDDprediccionesTraining, RDDprediccionesTest)
               }
               //case "SVM" => ModeloSVM.Modelo(RDDprediccionesTraining, args.apply(1).toInt)
               case "LR" => {
                    val modelo = ModeloLR.Modelo(RDDprediccionesTraining, args.apply(1).toInt)

                    println("Realizando Paso 4 para: " + args.apply(0))
                    val testLimpio = test.map({ case LabeledPoint(v1, v2) => v2 })
                    val prediccion = modelo.predict(testLimpio)
                    val RDDprediccionesTest = prediccion.zip(testLimpio).map({ case (v1, v2) => LabeledPoint(v1, v2) })
                    return Array(RDDprediccionesTraining, RDDprediccionesTest)
               }
               case "DT" => {
                    val modelo = ModeloDT.Modelo(RDDprediccionesTraining)

                    println("Realizando Paso 4 para: " + args.apply(0))
                    val testLimpio = test.map({ case LabeledPoint(v1, v2) => v2 })
                    val prediccion = modelo.predict(testLimpio)
                    val RDDprediccionesTest = prediccion.zip(testLimpio).map({ case (v1, v2) => LabeledPoint(v1, v2) })
                    return Array(RDDprediccionesTraining, RDDprediccionesTest)
               }
          }

     }
}