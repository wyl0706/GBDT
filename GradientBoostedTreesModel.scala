import scala.collection.mutable
import com.github.fommil.netlib.BLAS.{getInstance => blas}
import org.json4s._
import org.json4s.JsonDSL._
import org.json4s.jackson.JsonMethods._
import org.apache.spark.SparkContext
import org.apache.spark.annotation.{DeveloperApi, Since}
import org.apache.spark.api.java.JavaRDD
import org.apache.spark.internal.Logging
import org.apache.spark.mllib.linalg.Vector
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.tree.configuration.Algo
import org.apache.spark.mllib.tree.configuration.Algo._
import org.apache.spark.mllib.tree.configuration.EnsembleCombiningStrategy._
import org.apache.spark.mllib.tree.loss.Loss
import org.apache.spark.mllib.util.{Loader, Saveable}
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.SparkSession
import org.apache.spark.util.Utils

class GradientBoostedTreesModel (
   
  extends TreeEnsembleModel(algo, trees, treeWeights, combiningStrategy = Sum)
  with Saveable {

  require(trees.length == treeWeights.length)

  override def save(sc: SparkContext, path: String): Unit = {
    TreeEnsembleModel.SaveLoadV1_0.save(sc, path, this,
      GradientBoostedTreesModel.SaveLoadV1_0.thisClassName)
  }


  def evaluateEachIteration(
      data: RDD[LabeledPoint],
      loss: Loss): Array[Double] = {

    val sc = data.sparkContext
    val remappedData = algo match {
      case Classification => data.map(x => new LabeledPoint((x.label * 2) - 1, x.features))
      case _ => data
    }

    val broadcastTrees = sc.broadcast(trees)
    val localTreeWeights = treeWeights
    val treesIndices = trees.indices

    val dataCount = remappedData.count()
    val evaluation = remappedData.map { point =>
      treesIndices
        .map(idx => broadcastTrees.value(idx).predict(point.features) * localTreeWeights(idx))
        .scanLeft(0.0)(_ + _).drop(1)
        .map(prediction => loss.computeError(prediction, point.label))
    }
    .aggregate(treesIndices.map(_ => 0.0))(
      (aggregated, row) => treesIndices.map(idx => aggregated(idx) + row(idx)),
      (a, b) => treesIndices.map(idx => a(idx) + b(idx)))
    .map(_ / dataCount)

    broadcastTrees.destroy()
    evaluation.toArray
  }

  override protected def formatVersion: String = GradientBoostedTreesModel.formatVersion
}

object GradientBoostedTreesModel extends Loader[GradientBoostedTreesModel] {


  def computeInitialPredictionAndError(
      data: RDD[LabeledPoint],
      initTreeWeight: Double,
      initTree: DecisionTreeModel,
      loss: Loss): RDD[(Double, Double)] = {
    data.map { lp =>
      val pred = initTreeWeight * initTree.predict(lp.features)
      val error = loss.computeError(pred, lp.label)
      (pred, error)
    }
  }

  def updatePredictionError(
    data: RDD[LabeledPoint],
    predictionAndError: RDD[(Double, Double)],
    treeWeight: Double,
    tree: DecisionTreeModel,
    loss: Loss): RDD[(Double, Double)] = {

    val newPredError = data.zip(predictionAndError).mapPartitions { iter =>
      iter.map { case (lp, (pred, error)) =>
        val newPred = pred + tree.predict(lp.features) * treeWeight
        val newError = loss.computeError(newPred, lp.label)
        (newPred, newError)
      }
    }
    newPredError
  }

  private[mllib] def formatVersion: String = TreeEnsembleModel.SaveLoadV1_0.thisFormatVersion

  override def load(sc: SparkContext, path: String): GradientBoostedTreesModel = {
    val (loadedClassName, version, jsonMetadata) = Loader.loadMetadata(sc, path)
    val classNameV1_0 = SaveLoadV1_0.thisClassName
    (loadedClassName, version) match {
      case (className, "1.0") if className == classNameV1_0 =>
        val metadata = TreeEnsembleModel.SaveLoadV1_0.readMetadata(jsonMetadata)
        assert(metadata.combiningStrategy == Sum.toString)
        val trees =
          TreeEnsembleModel.SaveLoadV1_0.loadTrees(sc, path, metadata.treeAlgo)
        new GradientBoostedTreesModel(Algo.fromString(metadata.algo), trees, metadata.treeWeights)
      case _ => throw new Exception(s"GradientBoostedTreesModel.load did not recognize model" +
        s" with (className, format version): ($loadedClassName, $version).  Supported:\n" +
        s"  ($classNameV1_0, 1.0)")
    }
  }

  private object SaveLoadV1_0 {
    // Hard-code class name string in case it changes in the future
    def thisClassName: String = "org.apache.spark.mllib.tree.model.GradientBoostedTreesModel"
  }

}

private[tree] sealed class TreeEnsembleModel(
    protected val algo: Algo,
    protected val trees: Array[DecisionTreeModel],
    protected val treeWeights: Array[Double],
    protected val combiningStrategy: EnsembleCombiningStrategy) extends Serializable {

  require(numTrees > 0, "TreeEnsembleModel cannot be created without trees.")

  private val sumWeights = math.max(treeWeights.sum, 1e-15)

  private def predictBySumming(features: Vector): Double = {
    val treePredictions = trees.map(_.predict(features))
    blas.ddot(numTrees, treePredictions, 1, treeWeights, 1)
  }

  private def predictByVoting(features: Vector): Double = {
    val votes = mutable.Map.empty[Int, Double]
    trees.view.zip(treeWeights).foreach { case (tree, weight) =>
      val prediction = tree.predict(features).toInt
      votes(prediction) = votes.getOrElse(prediction, 0.0) + weight
    }
    votes.maxBy(_._2)._1
  }

  def predict(features: Vector): Double = {
    (algo, combiningStrategy) match {
      case (Regression, Sum) =>
        predictBySumming(features)
      case (Regression, Average) =>
        predictBySumming(features) / sumWeights
      case (Classification, Sum) => // binary classification
        val prediction = predictBySumming(features)
        // TODO: predicted labels are +1 or -1 for GBT. Need a better way to store this info.
        if (prediction > 0.0) 1.0 else 0.0
      case (Classification, Vote) =>
        predictByVoting(features)
      case _ =>
        throw new IllegalArgumentException(
          "TreeEnsembleModel given unsupported (algo, combiningStrategy) combination: " +
            s"($algo, $combiningStrategy).")
    }
  }

  def predict(features: RDD[Vector]): RDD[Double] = features.map(x => predict(x))

  def predict(features: JavaRDD[Vector]): JavaRDD[java.lang.Double] = {
    predict(features.rdd).toJavaRDD().asInstanceOf[JavaRDD[java.lang.Double]]
  }

  override def toString: String = {
    algo match {
      case Classification =>
        s"TreeEnsembleModel classifier with $numTrees trees\n"
      case Regression =>
        s"TreeEnsembleModel regressor with $numTrees trees\n"
      case _ => throw new IllegalArgumentException(
        s"TreeEnsembleModel given unknown algo parameter: $algo.")
    }
  }

  def toDebugString: String = {
    val header = toString + "\n"
    header + trees.zipWithIndex.map { case (tree, treeIndex) =>
      s"  Tree $treeIndex:\n" + tree.topNode.subtreeToString(4)
    }.fold("")(_ + _)
  }

  def numTrees: Int = trees.length

  def totalNumNodes: Int = trees.map(_.numNodes).sum
}

private[tree] object TreeEnsembleModel extends Logging {

  object SaveLoadV1_0 {

    import org.apache.spark.mllib.tree.model.DecisionTreeModel.SaveLoadV1_0.{NodeData, constructTrees}

    def thisFormatVersion: String = "1.0"

    case class Metadata(
        algo: String,
        treeAlgo: String,
        combiningStrategy: String,
        treeWeights: Array[Double])

    case class EnsembleNodeData(treeId: Int, node: NodeData)

    def save(sc: SparkContext, path: String, model: TreeEnsembleModel, className: String): Unit = {
      val spark = SparkSession.builder().sparkContext(sc).getOrCreate()
      val memThreshold = 768
      if (sc.isLocal) {
        val driverMemory = sc.getConf.getOption("spark.driver.memory")
          .orElse(Option(System.getenv("SPARK_DRIVER_MEMORY")))
          .map(Utils.memoryStringToMb)
          .getOrElse(Utils.DEFAULT_DRIVER_MEM_MB)
        if (driverMemory <= memThreshold) {
          logWarning(s"$className.save() was called, but it may fail because of too little" +
            s" driver memory (${driverMemory}m)." +
            s"  If failure occurs, try setting driver-memory ${memThreshold}m (or larger).")
        }
      } else {
        if (sc.executorMemory <= memThreshold) {
          logWarning(s"$className.save() was called, but it may fail because of too little" +
            s" executor memory (${sc.executorMemory}m)." +
            s"  If failure occurs try setting executor-memory ${memThreshold}m (or larger).")
        }
      }

      // Create JSON metadata.
      implicit val format = DefaultFormats
      val ensembleMetadata = Metadata(model.algo.toString, model.trees(0).algo.toString,
        model.combiningStrategy.toString, model.treeWeights)
      val metadata = compact(render(
        ("class" -> className) ~ ("version" -> thisFormatVersion) ~
          ("metadata" -> Extraction.decompose(ensembleMetadata))))
      sc.parallelize(Seq(metadata), 1).saveAsTextFile(Loader.metadataPath(path))

      // Create Parquet data.
      val dataRDD = sc.parallelize(model.trees.zipWithIndex).flatMap { case (tree, treeId) =>
        tree.topNode.subtreeIterator.toSeq.map(node => NodeData(treeId, node))
      }
      spark.createDataFrame(dataRDD).write.parquet(Loader.dataPath(path))
    }

    def readMetadata(metadata: JValue): Metadata = {
      implicit val formats = DefaultFormats
      (metadata \ "metadata").extract[Metadata]
    }


    def loadTrees(
        sc: SparkContext,
        path: String,
        treeAlgo: String): Array[DecisionTreeModel] = {
      val spark = SparkSession.builder().sparkContext(sc).getOrCreate()
      import spark.implicits._
      val nodes = spark.read.parquet(Loader.dataPath(path)).map(NodeData.apply)
      val trees = constructTrees(nodes.rdd)
      trees.map(new DecisionTreeModel(_, Algo.fromString(treeAlgo)))
    }
  }

}
