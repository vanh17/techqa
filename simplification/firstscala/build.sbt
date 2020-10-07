name := "firstscala"

version := "1.0-SNAPSHOT"

resolvers += "Artifactory" at "https://artifactory.cs.arizona.edu:8081/artifactory/sbt-release"

organization := "org.clulab"

scalaVersion := "2.12.6"

scalacOptions ++= Seq("-feature", "-unchecked", "-deprecation")

libraryDependencies ++= {
val      procVer = "8.2.1"
Seq(
  "org.clulab"                 %% "processors-main"          % procVer,
  "org.clulab"                 %% "processors-corenlp"       % procVer,
  "org.apache.lucene" % "lucene-core" % "7.6.0",
  "org.apache.lucene" % "lucene-queryparser" % "7.6.0",
  "org.apache.lucene" % "lucene-analyzers-common" % "7.6.0",
  "org.scalatest" %% "scalatest" % "3.0.1" % "test"
)
}
