package prueba

case class Attribute(
                        var name: String,
                        var kind: Char,
                        var IO: Char,
                        var rangeMin: Double,
                        var rangeMax: Double,
                        var valNominal: Array[String] = null // contiene para cada atributo nominal sus valores originales y si no es nominal
                    ) {
    var numInstValue: Array[Double] = null // number of instances with this value
    var numInsValueClass: Array[Array[Double]] = null // number of instances with this value per class

}