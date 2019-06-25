package prueba

import org.apache.spark.SparkContext
import org.apache.spark.rdd.RDD

class DataSet extends Serializable {

    var nInput: Int = 0
    var nOutput: Int = 0
    var nAttributes: Int = 0
    var nInstances: Int = 0
    var outputValues: Array[String] = null
    var attributes: Array[Attribute] = null

    var instances: RDD[(Vector[Double], Vector[Double])] = null
    //val sc = new SparkContext((new SparkConf).setAppName("Attribute").setMaster("local[8]"))

    def getnInput = this.nInput

    def getnOutput = this.nOutput

    def getnAttributes = this.nAttributes

    def getOutputValues = this.outputValues

    def getAttributes = this.attributes

    def getInstances = this.instances

    def getnInstances = this.nInstances

    def loadDataSet(file: String, sc: SparkContext, numParticiones: Int) = {

        val lines = sc.textFile(file, numParticiones).cache()

        val head = lines.filter { line => line.startsWith("@") }
        val linesHead = head.filter { line => line.startsWith("@attribute") }
        nInput = head.filter(line => line.startsWith("@inputs")).first().split("( *),( *)|( +)").length - 1
        outputValues = head.filter(line => line.startsWith("@outputs")).first().split("( *),( *)|( +)")

        nOutput = outputValues.length - 1
        nAttributes = nInput + nOutput

        val linesInstances = lines.filter(line => !line.contains("@") && !line.trim().isEmpty)

        loadAttributes(linesHead)
        loadInstances(linesInstances)

        //println("Inputs: " + nInput + " Outputs: " + nOutput + " Total attributes: " + nAttributes)
    }

    def loadAttributes(lines: RDD[String]) = {

        var io = ' '
        val wordsLines = lines.map(line => line.split("( *)(\\{)( *)|( *)(\\})( *)|( *)(\\[)( *)|( *)(\\])( *)|( *)(,)( *)| "))

        //println("Atributos: ")

        attributes = wordsLines.collect().map {
            words =>
                //words.foreach((e: String) => print(e + " "))
                //println()
                if (outputValues.indexOf(words(1)) != -1) {
                    io = 'o'
                } else {
                    io = 'i'
                }
                if (words(2) == "real" || words(2) == "integer") {
                    //Constructor para atributo integer o real
                    Attribute(words(1), 'r', io, words(3).toDouble, words(4).toDouble, Array("Void"))
                } else {
                    //Constructor para atributo nominal
                    Attribute(words(1), 'n', io, 0, words.length - 3, words.slice(2, words.length))
                }
        }

        outputValues = attributes.last.valNominal
        attributes.last.name = outputValues(0)
        attributes.last.rangeMax = 1
        attributes.last.rangeMin = 0
        attributes.last.valNominal = Array("0", "1")
        for (i <- 1 to outputValues.length - 1) {
            attributes = attributes :+ Attribute(outputValues(i), 'n', 'o', 0.0, 1.0, Array("0", "1"))
        }

        nOutput = outputValues.length
        nAttributes = nInput + nOutput

        attributes.foreach(atb => if (atb.kind == 'n') atb.numInstValue = Array.fill(atb.valNominal.length)(0))
    }

    def loadInstances(lines: RDD[String]) = {

        //println("Numero de salidas: " + nOutput)

        val outV = outputValues

        val auxInstances = lines.map {
            line => line.split("( *)(,)( *)")
        }
        val auxRDD = auxInstances.map {
            line =>
                var i = -1;
                val auxVectorInput = new Array[Double](nInput);
                val auxVectorOutput = new Array[Double](nOutput);
                line.foreach {
                    x =>
                        i = i + 1;
                        if (attributes(i).IO == 'i') {

                            if (attributes(i).kind == 'r') {
                                auxVectorInput(i) = x.toDouble
                            }
                            else {
                                auxVectorInput(i) = attributes(i).valNominal.indexOf(x)
                                if (auxVectorInput(i) == -1) {
                                    println("ERROR al leer linea" + x)
                                }
                            }
                        } else {

                            if (attributes(i).kind == 'r') {
                                auxVectorOutput(0) = x.toDouble
                            }
                            else {
                                //put 1 into the active class
                                auxVectorOutput(outV.indexOf(x)) = 1
                            }

                        }
                }
                (auxVectorInput, auxVectorOutput)
        }
        this.nInstances = auxRDD.count().toInt
        instances = auxRDD.map(x => (x._1.toVector, (x._2).toVector))
    }

    def printAttributes(): Unit = {
        attributes.foreach {
            x =>
                println("Nombre: " + x.name + " Tipo: " + x.kind + " IO: " + x.IO + " Rango minimo: " + x.rangeMin + " Rango maximo: " + x.rangeMax)

                x.valNominal.foreach((e: String) => print(e + " "))

                println()
        }
    }

    def printInstances() = {
        println("Numero instancias: " + instances.count())
        instances.collect().foreach(println)
    }

    def vectorToDouble(vec: Vector[Double]): Double = {

        var doubleValue: Double = -1
        val array = vec.toArray

        for (i <- 0 to array.length - 1) {

            if (array.apply(i) == 1) {
                doubleValue = i
            }
        }

        return doubleValue
    }
}


