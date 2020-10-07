package edu.arizona.cs

import org.clulab.processors.fastnlp.FastNLPProcessor
import org.clulab.processors.clu.CluProcessor
import scala.io.Source

class FirstScala(val fileToProcess:String) {
  def process() {
    val proc = new CluProcessor()//FastNLPProcessor()
    // read the whole text from a file; concatenate all sentences into one String
    //val doc = proc.annotate(Source.fromFile(fileToProcess).getLines().mkString("\n"))
    // just a for loop over the sentences in this text
    val doc = proc.mkDocument(Source.fromFile(fileToProcess).getLines().mkString("\n"))
    for(s <- doc.sentences) {
      // mkString converts any collection to a string,
      // where the elements are separated by the given separator string
      println(s"Words: ${s.words.mkString(", ")}")
      println("POS tags: " + s.tags.get.mkString(", "))
      val words = s.words
      val tags = s.tags.get
      // keep just the nouns in this text
      val justNouns = words.zip(tags).filter(_._2.startsWith("NN")).map(_._1)
      println("Just the nouns in this sentence: " + justNouns.mkString(", "))
    }
  }
}

// store your static methods/fields in the "object" construct
object FirstScala {
  // this is just like main() in Java
  def main(args:Array[String]) {
      val fs = new FirstScala(args(0))
      org.clulab.dynet.Utils.initializeDyNet()
      fs.process()
  }
}
