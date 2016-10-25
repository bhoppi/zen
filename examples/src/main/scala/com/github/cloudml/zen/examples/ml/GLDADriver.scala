/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.github.cloudml.zen.examples.ml

import com.github.cloudml.zen.ml.semiSupervised.GLDA
import com.github.cloudml.zen.ml.semiSupervised.GLDADefines._
import com.github.cloudml.zen.ml.util.{SparkHacker, SparkUtils}
import org.apache.hadoop.fs.Path
import org.apache.spark.rdd.RDD
import org.apache.spark.storage.StorageLevel
import org.apache.spark.{SparkConf, SparkContext}

import scala.annotation.tailrec


object GLDADriver {
  def main(args: Array[String]) {
    val options = parseArgs(args)
    val appStartedTime = System.nanoTime

    val numTopics = options("numtopics").toInt
    val numGroups = options("numgroups").toInt
    val alpha = options("alpha").toFloat
    val beta = options("beta").toFloat
    val eta = options("eta").toFloat
    val mu = options("mu").toFloat
    val totalIter = options("totaliter").toInt
    val numPartitions = options("numpartitions").toInt
    assert(numTopics > 0, "numTopics must be greater than 0")
    assert(numGroups > 0, "numGroups must be greater than 0")
    assert(alpha > 0f && beta > 0f && eta > 0f && mu > 0f)
    assert(totalIter > 0, "totalIter must be greater than 0")
    assert(numPartitions > 0, "numPartitions must be greater than 0")
    val params = HyperParams(alpha, beta, eta, mu)

    val inputPath = options("inputpath")
    val outputPath = options("outputpath")
    val checkpointPath = outputPath + ".checkpoint"

    val storageLevelStr = options.getOrElse("storagelevel", "MEMORY_AND_DISK").toUpperCase
    val storageLevel = StorageLevel.fromString(storageLevelStr)
    val numThreadsStr = options.getOrElse("numthreads", "1")
    val numThreads = numThreadsStr.toInt
    val useKyro = options.get("usekryo").exists(_.toBoolean)

    val conf = new SparkConf()
    conf.set(cs_numPartitions, s"$numPartitions")
    conf.set(cs_inputPath, inputPath)
    conf.set(cs_outputpath, outputPath)
    setAppConfs(conf, options)

    if (numThreads > 1) {
      conf.set("spark.task.cpus", numThreadsStr)
    }
    if (useKyro) {
      conf.set("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
      registerKryoClasses(conf)
    } else {
      conf.set("spark.serializer", "org.apache.spark.serializer.JavaSerializer")
    }

    val outPath = new Path(outputPath)
    val fs = SparkUtils.getFileSystem(conf, outPath)
    if (fs.exists(outPath)) {
      println(s"Error: output path $outputPath already exists.")
      System.exit(2)
    }
    fs.delete(new Path(checkpointPath), true)

    val sc = new SparkContext(conf)
    try {
      sc.setCheckpointDir(checkpointPath)
      println("start GLDA training")
      println(s"appId: ${sc.applicationId}")
      println(s"numTopics = $numTopics, numGroups = $numGroups, totalIteration = $totalIter")
      println(s"alpha = $alpha, beta = $beta, eta = $eta, mu = $mu")
      println(s"inputDataPath = $inputPath")
      println(s"outputPath = $outputPath")

      val corpus = loadCorpus(sc, numTopics, numGroups, numThreads, numPartitions, storageLevel)
      val trainingTime = runTraining(corpus, numTopics, numGroups, numThreads, totalIter, params, storageLevel)
      println(s"Training time consumed: $trainingTime seconds")
    } finally {
      sc.stop()
      fs.deleteOnExit(new Path(checkpointPath))
      val appEndedTime = System.nanoTime
      println(s"Total time consumed: ${(appEndedTime - appStartedTime) / 1e9} seconds")
      fs.close()
    }
  }

  def runTraining(corpus: (RDD[(Int, DataBlock)], RDD[(Int, ParaBlock)]),
    numTopics: Int,
    numGroups: Int,
    numThreads: Int,
    totalIter: Int,
    params: HyperParams,
    storageLevel: StorageLevel): Double = {
    SparkHacker.gcCleaner(30 * 60, 30 * 60, "LDA_gcCleaner")
    val trainingStartedTime = System.nanoTime
    val glda = GLDA(corpus, numTopics, numGroups, numThreads, params, storageLevel)
    glda.fit(totalIter)
    val model = glda.toGLDAModel
    val trainingEndedTime = System.nanoTime
    println("save the model in term-topic view")
    model.save()
    (trainingEndedTime - trainingStartedTime) / 1e9
  }

  def loadCorpus(sc: SparkContext,
    numTopics: Int,
    numGroups: Int,
    numThreads: Int,
    numPartitions: Int,
    storageLevel: StorageLevel): (RDD[(Int, DataBlock)], RDD[(Int, ParaBlock)]) = {
    val conf = sc.getConf
    val inputPath = conf.get(cs_inputPath)
    val sampleRate = conf.get(cs_sampleRate).toFloat
    val labelsRate = conf.get(cs_labelsRate).toFloat
    val withReplacement = false
    var rawDocs = sc.textFile(inputPath, numPartitions).sample(withReplacement, sampleRate)
    if (rawDocs.getNumPartitions < numPartitions) {
      rawDocs = rawDocs.coalesce(numPartitions, shuffle=true)
    }
    GLDA.initCorpus(rawDocs, numTopics, numGroups, numThreads, labelsRate, storageLevel)
  }

  def parseArgs(args: Array[String]): OptionMap = {
    val usage = "Usage: GLDADriver <Args> [Options] <Input path> <Output path>\n" +
      "  Args: -numTopics=<Int> -numGroups=<Int> -alpha=<Float> -beta=<Float> -eta=<Float> -mu=<Float>\n" +
      "        -totalIter=<Int> -numPartitions=<Int>\n" +
      "  Options: -numThreads=<Int(*1)>\n" +
      "           -burninIter=<Int(*10)>\n" +
      "           -sampleRate=<Float(*1.0)>\n" +
      "           -chkptInterval=<Int(*10)> (0 or negative disables checkpoint)\n" +
      "           -evalMetric=<*None|{PPLX|LLH|COH}+>\n" +
      "           -saveInterval=<Int(*0)> (0 or negative disables save at intervals)\n" +
      "           -saveAsSolid=<Boolean(*true)>\n" +
      "           -labelsRate=<Float(*1.0)>\n" +
      "           -storageLevel=<StorageLevel(*MEMORY_AND_DISK)>\n" +
      "           -useKryo=<true|*false>"
    if (args.length < 10) {
      println(usage)
      System.exit(1)
    }
    val arglist = args.toList
    @tailrec def nextOption(map: OptionMap, list: List[String]): OptionMap = {
      def isSwitch(s: String) = s(0) == '-'
      list match {
        case Nil => map
        case head :: Nil if !isSwitch(head) =>
          nextOption(map ++ Map("outputpath" -> head), Nil)
        case head :: tail if !isSwitch(head) =>
          nextOption(map ++ Map("inputpath" -> head), tail)
        case head :: tail if isSwitch(head) =>
          var kv = head.toLowerCase.split("=", 2)
          if (kv.length == 1) {
            kv = head.toLowerCase.split(":", 2)
          }
          if (kv.length == 1) {
            println(s"Error: wrong command line format: $head")
            System.exit(1)
          }
          nextOption(map ++ Map(kv(0).substring(1) -> kv(1)), tail)
        case _ =>
          println(usage)
          System.exit(1)
          null.asInstanceOf[OptionMap]
      }
    }
    nextOption(Map(), arglist)
  }
}
