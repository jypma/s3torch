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
/* new
 This somehow fails to run the GPU code:

 java.lang.RuntimeException: Could not run 'aten::empty_strided' with arguments from the 'CUDA' backend. This could be because the operator doesn't exist for this backend, or was omitted during the selective/custom build process (if using custom build). If you are a Facebook employee using PyTorch on mobile, please visit https://fburl.com/ptmfixes for possible resolutions. 'aten::empty_strided' is only available for these backends: [CPU, Meta, QuantizedCPU, BackendSelect, Python, FuncTorchDynamicLayerBackMode, Functionalize, Named, Conjugate, Negative, ZeroTensor, ADInplaceOrView, AutogradOther, AutogradCPU, AutogradCUDA, AutogradHIP, AutogradXLA, AutogradMPS, AutogradIPU, AutogradXPU, AutogradHPU, AutogradVE, AutogradLazy, AutogradMTIA, AutogradPrivateUse1, AutogradPrivateUse2, AutogradPrivateUse3, AutogradMeta, AutogradNestedTensor, Tracer, AutocastCPU, AutocastMTIA, AutocastXPU, AutocastMPS, AutocastCUDA, FuncTorchBatched, BatchedNestedTensor, FuncTorchVmapMode, Batched, VmapMode, FuncTorchGradWrapper, PythonTLSSnapshot, FuncTorchDynamicLayerFrontMode, PreDispatch, PythonDispatcher].

val javaCppVer = "1.5.12"
val pytorchVer = s"2.7.1-${javaCppVer}"
val openblasVer = s"0.3.30-${javaCppVer}"
val cudaVer = s"12.9-9.10-${javaCppVer}"
 */
val javaCppVer = "1.5.10"
val pytorchVer = s"2.1.2-${javaCppVer}"
val openblasVer = s"0.3.26-${javaCppVer}"
val cudaVer = s"12.3-8.9-${javaCppVer}"

libraryDependencies ++= Seq(
  "org.bytedeco" % "pytorch" % pytorchVer,
  "org.bytedeco" % "pytorch" % pytorchVer classifier "linux-x86_64-gpu",
  "org.bytedeco" % "openblas" % openblasVer classifier "linux-x86_64",
  "org.bytedeco" % "cuda" % cudaVer,
  "org.bytedeco" % "cuda" % cudaVer classifier "linux-x86_64",
  // new "org.bytedeco" % "cuda-redist" % cudaVer classifier "linux-x86_64",
  "org.bytedeco" % "cuda" % cudaVer classifier "linux-x86_64-redist",
  "org.scalatest" %% "scalatest" % "3.2.19" % "test",
  "io.github.json4s" %% "json4s-native" % "4.2.0-M3"
)
