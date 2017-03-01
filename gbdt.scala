GradientBoostedTrees private def boost(  
     input: RDD[LabeledPoint],  
     boostingStrategy: BoostingStrategy): GradientBoostedTreesModel = {  
  
    val timer = new TimeTracker()  
    timer.start("total")  
    timer.start("init")  
  
    boostingStrategy.assertValid()  
  
// ��ʼ��GBDT����
//��������  
	val numIterations = boostingStrategy.numIterations  
//ÿ�ε����ľ�����ģ��  
    val baseLearners = new Array[DecisionTreeModel](numIterations)  
   //ÿ����ģ�͵�Ȩ��   
	val baseLearnerWeights = new Array[Double](numIterations)  
   //��ʧ�ļ��㷽ʽ���ο��±�  
	val loss = boostingStrategy.loss  
   //ѧϰ�ʣ�һ�㲻���������������е��ţ�����㷨ģ�Ͳ��ȶ������齵�����ֵ  
  val learningRate = boostingStrategy.learningRate  
    // Prepare strategy for individual trees, which use regression with variance impurity.  
	val treeStrategy = boostingStrategy.treeStrategy.copy  
//algo֧��classication��Regresion  
    treeStrategy.algo = Regression  
    treeStrategy.impurity = Variance  
    treeStrategy.assertValid()  
  
    // Cache input,������������  
    if (input.getStorageLevel == StorageLevel.NONE) {  
      input.persist(StorageLevel.MEMORY_AND_DISK)  
    }  
  
    timer.stop("init")  
  
    logDebug("##########")  
    logDebug("Building tree 0")  
    logDebug("##########")  
    var data = input  
  
    // Initialize tree  
    timer.start("building tree 0")  
    val firstTreeModel = new DecisionTree(treeStrategy).run(data)  
    baseLearners(0) = firstTreeModel  
    baseLearnerWeights(0) = 1.0  
    val startingModel = new GradientBoostedTreesModel(Regression, Array(firstTreeModel), Array(1.0))  
    logDebug("error of gbt = " + loss.computeError(startingModel, input))  
    // Note: A model of type regression is used since we require raw prediction  
    timer.stop("building tree 0")  
  
    // psuedo-residual for second iteration  
    data = input.map(point => LabeledPoint(loss.gradient(startingModel, point),  
      point.features))  
  
	var m = 1  
//����ѵ������ģ��  
    while (m < numIterations) {  
      timer.start(s"building tree $m")  
      logDebug("###################################################")  
      logDebug("Gradient boosting tree iteration " + m)  
      logDebug("###################################################")  
      val model = new DecisionTree(treeStrategy).run(data)  
      timer.stop(s"building tree $m")  
      // Create partial model  
      baseLearners(m) = model  
      // Note: The setting of baseLearnerWeights is incorrect for losses other than SquaredError.  
      //       Technically, the weight should be optimized for the particular loss.  
      //       However, the behavior should be reasonable, though not optimal.  
//��ǰ��ģ�͵�Ȩ��weight       
		baseLearnerWeights(m) = learningRate  
      // Note: A model of type regression is used since we require raw prediction  
   //ÿ������ģ��     
	val partialModel = new GradientBoostedTreesModel(  
        Regression, baseLearners.slice(0, m + 1), baseLearnerWeights.slice(0, m + 1))  
      logDebug("error of gbt = " + loss.computeError(partialModel, input))  
  
//���òв�(�ݶȷ���)�����������ݼ�,��Ϊ�¿���ģ��ѵ��������  
      data = input.map(point => LabeledPoint(-loss.gradient(partialModel, point),  
        point.features))  
      m += 1  
    }  
  
    timer.stop("total")  
  
    logInfo("Internal timing for DecisionTree:")  
    logInfo(s"$timer")  
//����ںϸ�������ģ��  
    new GradientBoostedTreesModel(  
      boostingStrategy.treeStrategy.algo, baseLearners, baseLearnerWeights)  
  }  