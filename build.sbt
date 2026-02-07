scalaVersion := "3.8.1"
organization := "ypmania.net"
version := "0.0.1-SNAPSHOT"

scalacOptions ++= Seq(
  "-explain",
  "-explain-cyclic",
  "-feature",
  "-deprecation",
  "-language:implicitConversions"
)

val javaCppVer = "1.5.12"
val pytorchVer = s"2.7.1-${javaCppVer}"
val openblasVer = s"0.3.30-${javaCppVer}"
val cudaVer = s"12.9-9.10-${javaCppVer}"

libraryDependencies ++= Seq(
  "org.bytedeco" % "javacpp" % javaCppVer,
  "org.bytedeco" % "pytorch" % pytorchVer,
  "org.bytedeco" % "pytorch" % pytorchVer classifier "linux-x86_64-gpu",
  "org.bytedeco" % "openblas" % openblasVer,
  "org.bytedeco" % "openblas" % openblasVer classifier "linux-x86_64",
  "org.bytedeco" % "cuda" % cudaVer,
  "org.bytedeco" % "cuda" % cudaVer classifier "linux-x86_64",
  "org.bytedeco" % "cuda-redist" % cudaVer,
  "org.bytedeco" % "cuda-redist" % cudaVer classifier "linux-x86_64",
  "org.scalatest" %% "scalatest" % "3.2.19" % "test"
)
