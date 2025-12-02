scalaVersion := "3.7.4"
organization := "ypmania.net"
version := "0.0.1-SNAPSHOT"

scalacOptions ++= Seq(
  "-explain",
  "-feature",
  "-language:implicitConversions"

)

libraryDependencies ++= Seq(
  "org.bytedeco" % "pytorch" % "2.7.1-1.5.12",
  "org.bytedeco" % "pytorch" % "2.7.1-1.5.12" classifier "linux-x86_64",
  "org.bytedeco" % "openblas" % "0.3.30-1.5.12" classifier "linux-x86_64",
)

