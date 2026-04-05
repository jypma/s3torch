fork := true
scalaVersion := "3.8.2"
organization := "ypmania.net"
version := "0.0.1-SNAPSHOT"

name := "s3torch-examples"

libraryDependencies ++= Seq(
  "ypmania.net" %% "s3torch" % "0.0.1-SNAPSHOT",
  "org.scalatest" %% "scalatest" % "3.2.19" % "test"
)

scalacOptions ++= Seq(
  "-feature",
  "-deprecation",
  "-language:implicitConversions",
  "-Wunused:imports"
)
