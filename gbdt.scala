GradientBoostedTrees private def boost(  
	input: RDD[LabeledPoint],  
	boostingStrategy: BoostingStrategy): GradientBoostedTreesModel = {  

	val timer = new TimeTracker()  
	timer.start("total")  
	timer.start("init")  

	boostingStrategy.assertValid()  

	// 初始化GBDT参数
	//迭代次数  
	val numIterations = boostingStrategy.numIterations  
	//每次迭代的决策树模型  
	val baseLearners = new Array[DecisionTreeModel](numIterations)  
	//每个树模型的权重   
	val baseLearnerWeights = new Array[Double](numIterations)  
	//损失的计算方式，参考下表  
	val loss = boostingStrategy.loss  
	//学习率
	val learningRate = boostingStrategy.learningRate  
	// Prepare strategy for individual trees, which use regression with variance impurity.  
	val treeStrategy = boostingStrategy.treeStrategy.copy  
	//algo支持c分类和回归，此处使用回归
	treeStrategy.algo = Regression  
	treeStrategy.impurity = Variance  
	treeStrategy.assertValid()  

	// Cache input,缓存样本集合  
	if (input.getStorageLevel == StorageLevel.NONE) {  
	  input.persist(StorageLevel.MEMORY_AND_DISK)  
	}  

	timer.stop("init")  

	logDebug("##########")  
	logDebug("Building tree 0")  
	logDebug("##########")  
	var data = input  

	//初始化树
	timer.start("building tree 0")  
	val firstTreeModel = new DecisionTree(treeStrategy).run(data)  
	baseLearners(0) = firstTreeModel  
	baseLearnerWeights(0) = 1.0  
	val startingModel = new GradientBoostedTreesModel(Regression, Array(firstTreeModel), Array(1.0))  
	logDebug("error of gbt = " + loss.computeError(startingModel, input))  
	timer.stop("building tree 0")  

	data = input.map(point => LabeledPoint(loss.gradient(startingModel, point),  point.features))  

	var m = 1  
	//迭代训练各树模型  
	while (m < numIterations) {  
		timer.start(s"building tree $m")  
		logDebug("###################################################")  
		logDebug("Gradient boosting tree iteration " + m)  
		logDebug("###################################################")  
		val model = new DecisionTree(treeStrategy).run(data)  
		timer.stop(s"building tree $m")  
		baseLearners(m) = model  
		//当前数模型的权重weight       
		baseLearnerWeights(m) = learningRate  
		//每个树的模型     
		val partialModel = new GradientBoostedTreesModel(  
		Regression, baseLearners.slice(0, m + 1), baseLearnerWeights.slice(0, m + 1))  
		logDebug("error of gbt = " + loss.computeError(partialModel, input))  
		//利用残差(梯度方向)更新样本数据集,作为下颗树模型训练的样本  
		data = input.map(point => LabeledPoint(-loss.gradient(partialModel, point),  
		point.features))  
		m += 1  
    }  
  
	timer.stop("total")  

	logInfo("Internal timing for DecisionTree:")  
	logInfo(s"$timer")  
	//最后融合各颗树的模型  
	new GradientBoostedTreesModel(boostingStrategy.treeStrategy.algo, baseLearners, baseLearnerWeights)  
  }  