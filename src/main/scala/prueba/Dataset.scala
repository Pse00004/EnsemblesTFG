package prueba

import org.apache.spark.SparkContext
import org.apache.spark.rdd.RDD

class DataSet extends Serializable {

     //type Instance = (Vector[Double], Vector[Double])
     var task: Char = 'c'
     var nInput: Int = 0
     var nOutput: Int = 0
     var nAttributes: Int = 0
     var nInstances: Int = 0
     var outputValues: Array[String] = null
     var attributes: Array[Attribute] = null

     var instances: RDD[(Vector[Double], Vector[Double])] = null
     //val sc = new SparkContext((new SparkConf).setAppName("Attribute").setMaster("local[8]"))

     def getTask = this.task
     def getnInput = this.nInput
     def getnOutput = this.nOutput
     def getnAttributes = this.nAttributes
     def getOutputValues = this.outputValues
     def getAttributes = this.attributes
     def getInstances = this.instances
     def getnInstances = this.nInstances

     def loadDataSet(file: String, sc: SparkContext) = {

          val lines = sc.textFile(file).cache()

          val head = lines.filter { line => line.startsWith("@") }
          val linesHead = head.filter { line => line.startsWith("@attribute") }
          nInput = head.filter(line => line.startsWith("@inputs")).first().split("( *),( *)|( +)").length - 1
          outputValues = head.filter(line => line.startsWith("@outputs")).first().split("( *),( *)|( +)")

          nOutput = outputValues.length - 1
          nAttributes = nInput + nOutput

          val linesInstances = lines.filter(line => !line.contains("@") && !line.trim().isEmpty)

          loadAttributes(linesHead)
          loadInstances(linesInstances)

          //summaryNominalAttributes()
          println("Loading, inputs: " + nInput + " outputs: " + nOutput + " total attributes " + nAttributes)
          //printInstances()
     }

     def loadAttributes(lines: RDD[String]) = {

          var io = ' '
          val wordsLines = lines.map(line => line.split("( *)(\\{)( *)|( *)(\\})( *)|( *)(\\[)( *)|( *)(\\])( *)|( *)(,)( *)| "))
          println("Atributos")
          attributes = wordsLines.collect().map {
               words =>
                    words.foreach(println)
                    if (outputValues.indexOf(words(1)) != -1) io = 'o' else io = 'i'
                    if (words(2) == "real" || words(2) == "integer") Attribute(words(1), 'r', io, words(3).toDouble, words(4).toDouble, Array("Void")) //call the constructor real attribute
                    else Attribute(words(1), 'n', io, 0, words.length - 3, words.slice(2, words.length)) //call the constructor nominal attribute
          }

          // if classification we goint to consider the number of output attributes equal to number of different values of this attribute
          if (task == 'c') {
               outputValues = attributes.last.valNominal
               attributes.last.name = outputValues(0)
               attributes.last.rangeMax = 1
               attributes.last.rangeMin = 0
               attributes.last.valNominal = Array("0", "1")
               for (i <- 1 to outputValues.length - 1) {
                    attributes = attributes :+ Attribute(outputValues(i), 'n', 'o', 0.0, 1.0, Array("0", "1"))
               }

               // update the number of outputs
               nOutput = outputValues.length
               nAttributes = nInput + nOutput
          }
          attributes.foreach(atb => if (atb.kind == 'n') atb.numInstValue = Array.fill(atb.valNominal.length)(0))
     }

     def loadInstances(lines: RDD[String]) = {
          // No possible to use a class variable inside a map
          // No possible to reference a RDD inside a map of other RDD
          // unless used extend serializable

          println("Numero de salidas: " + nOutput)
          val nInp = nInput;
          val tsk = task
          val outV = outputValues

          val auxInstances = lines.map { line => line.split("( *)(,)( *)") }
          val auxRDD = auxInstances.map { line =>
               var i = -1;
               val auxVectorInput = new Array[Double](nInp);
               val auxVectorOutput = new Array[Double](nOutput);
               line.foreach { x =>
                    i = i + 1;
                    if (attributes(i).IO == 'i') {
                         if (attributes(i).kind == 'r') auxVectorInput(i) = x.toDouble
                         else {
                              auxVectorInput(i) = attributes(i).valNominal.indexOf(x)
                              if (auxVectorInput(i) == -1) println("*****ERROR " + x)
                         }
                    } else {
                         if (tsk == 'm')
                              auxVectorOutput(i - nInp) = x.toDouble
                         else {
                              if (attributes(i).kind == 'r') auxVectorOutput(0) = x.toDouble
                              else auxVectorOutput(outV.indexOf(x)) = 1 //put 1 into the active class
                         }
                    }
               }
               (auxVectorInput, auxVectorOutput)
          }
          this.nInstances = auxRDD.count().toInt
          instances = auxRDD.map(x => (x._1.toVector, (x._2).toVector))
     }

     def printAttributes(): Unit = {
          attributes.foreach { x =>
               println("Nombre: " + x.name + " Tipo: " + x.kind + " IO: " + x.IO + " Rango minimo: " + x.rangeMin + " Rango maximo: " + x.rangeMax)
               x.valNominal.foreach(println)
               if (x.kind == 'n' && x.IO == 'i') {
                    println("Summary")
                    println("InsValue")
                    x.numInstValue.foreach(println)
                    println("InsValueClass")
                    x.numInsValueClass.foreach { y => y.foreach { z => print(z); print(',') }; println() }
               }
          }
     }

     def printInstances() = {
          println("Numero instancias: " + instances.count())
          instances.collect().foreach(println)
     }
     
     def vectorToDouble(vec: Vector[Double]): Double ={
          
          var doubleValue: Double = -1
          var array = vec.toArray
          
          for (i <- 0 to array.length - 1){
               
               if(array.apply(i)==1){
                    doubleValue = i
               }
          }
          
          return doubleValue
     }
}


