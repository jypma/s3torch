fork := true
scalaVersion := "3.8.1"
organization := "ypmania.net"
version := "0.0.1-SNAPSHOT"

scalacOptions ++= Seq(
  "-explain",
  "-explain-cyclic",
  "-feature",
  "-deprecation",
  "-language:implicitConversions",
  "-Wunused:imports"
)

// Scaladoc has bugs that surface on our code
Compile/packageDoc/publishArtifact := false

// Enable scalafix:
inThisBuild(
  List(
    scalaVersion := "3.8.1",
    semanticdbEnabled := true,
    semanticdbVersion := scalafixSemanticdb.revision
  )
)

val enableGPU = settingKey[Boolean]("enable or disable GPU support")
ThisBuild / enableGPU := true /* TODO; currently, "false" doesn't load the correct native libraries. */
javaCppVersion := "1.5.13"
val pytorchVersion = "2.10.0"
val openblasVersion = "0.3.31"
val cudaVersion = "13.1-9.19"

javaCppPresetLibs ++= Seq(
  (if (enableGPU.value) "pytorch-gpu" else "pytorch") -> pytorchVersion,
  "openblas" -> openblasVersion
)

javaCppPlatform := org.bytedeco.sbt.javacpp.Platform.current

libraryDependencies ++= (if (enableGPU.value) Seq(
  "org.bytedeco" % "cuda" % "13.1-9.19-1.5.13",
  "org.bytedeco" % "cuda-redist" % "13.1-9.19-1.5.13" classifier "linux-x86_64",
) else Seq.empty)

libraryDependencies ++= Seq(
  "io.github.json4s" %% "json4s-native" % "4.2.0-M3",
  "org.scalatest" %% "scalatest" % "3.2.19" % "test"
)
