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

package com.github.cloudml.zen.ml.semiSupervised

import java.io.IOException
import java.util.Random
import java.util.concurrent.ConcurrentLinkedQueue

import breeze.linalg._
import breeze.numerics._
import com.github.cloudml.zen.ml.semiSupervised.GLDADefines._
import com.github.cloudml.zen.ml.util.Concurrent._
import com.github.cloudml.zen.ml.util._
import org.apache.hadoop.fs.{FileUtil, Path}
import org.apache.spark._
import org.apache.spark.broadcast.Broadcast
import org.apache.spark.rdd.RDD
import org.apache.spark.storage.StorageLevel

import scala.collection.JavaConverters._
import scala.collection.concurrent.TrieMap
import scala.collection.mutable


class GLDA(@transient var dataBlocks: RDD[(Int, DataBlock)],
  @transient var paraBlocks: RDD[(Int, ParaBlock)],
  val numTopics: Int,
  val numGroups: Int,
  val numThreads: Int,
  val numTerms: Int,
  val numDocs: Long,
  val numTokens: Long,
  val params: HyperParams,
  var algo: GLDATrainer,
  var storageLevel: StorageLevel) extends Serializable {

  @transient var globalVarsBc: Broadcast[GlobalVars] = _
  @transient var groupContext: GroupContext = _
  @transient lazy val seed = new XORShiftRandom().nextInt()
  @transient var dataCpFile: String = _
  @transient var paraCpFile: String = _

  @inline def scContext: SparkContext = dataBlocks.context

  @inline def scConf: SparkConf = scContext.getConf

  def init(): GLDA = {
    val initParaBlocks = algo.updateParaBlocks(dataBlocks, paraBlocks)
    initParaBlocks.persist(storageLevel).setName("ParaBlocks-0")
    initParaBlocks.count()
    paraBlocks.unpersist(blocking=false)
    paraBlocks = initParaBlocks

    val globalVars = algo.collectGlobalVariables(dataBlocks, params, numGroups + 1)
    globalVarsBc = scContext.broadcast(globalVars)

    val burninIter = scConf.get(cs_burninIter).toInt
    val docGrouperStr = scConf.get(cs_docGrouper).toLowerCase
    val priors = log(convert(globalVars.dG.slice(0, numGroups), Double) :+= 1.0)
    groupContext = new GroupContext(numGroups, priors, burninIter, docGrouperStr)

    this
  }

  def fit(totalIter: Int): Unit = {
    val evalMetrics = scConf.get(cs_evalMetric).split(raw"\+")
    val toEval = !evalMetrics.contains("none")
    val saveIntv = scConf.get(cs_saveInterval).toInt
    val chkptIntv = scConf.get(cs_chkptInterval).toInt
    val canChkpt = chkptIntv > 0 && scContext.getCheckpointDir.isDefined

    if (toEval) {
      println("Before Gibbs sampling:")
      GLDAMetrics(this, evalMetrics).foreach(_.output(println))
    }
    for (sampIter <- 1 to totalIter) {
      println(s"\nStart Gibbs sampling (Iteration $sampIter/$totalIter)")
      val startedAt = System.nanoTime

      groupContext.setIteration(sampIter)
      fitIteration(sampIter, canChkpt && sampIter % chkptIntv == 1)
      if (toEval) {
        GLDAMetrics(this, evalMetrics).foreach(_.output(println))
      }
      val globalVars = algo.collectGlobalVariables(dataBlocks, params, groupContext.totalGroups)
      globalVarsBc.unpersist(blocking=false)
      globalVarsBc = scContext.broadcast(globalVars)
      val countTok = sum(convert(globalVars.nK, Long))
      val countDoc = sum(globalVars.dG)
      assert(countTok == numTokens && countDoc == numDocs, s"countTok=$countTok, countDoc=$countDoc")
      println(s"dG: ${globalVars.dG}")

      if (saveIntv > 0 && sampIter % saveIntv == 0 && sampIter < totalIter) {
        val model = toGLDAModel
        val savPath = new Path(scConf.get(cs_outputpath) + s"-iter$sampIter")
        val fs = SparkUtils.getFileSystem(scConf, savPath)
        fs.delete(savPath, true)
        model.save(scContext, savPath.toString)
        println(s"Model saved after Iteration $sampIter")
      }
      val elapsedSeconds = (System.nanoTime - startedAt) / 1e9
      println(s"End Gibbs sampling (Iteration $sampIter/$totalIter) takes total: $elapsedSeconds secs")
    }
  }

  def fitIteration(sampIter: Int, needChkpt: Boolean): Unit = {
    val startedAt = System.nanoTime
    val shippeds = algo.ShipParaBlocks(dataBlocks, paraBlocks)
    val newDataBlocks = algo.SampleNGroup(dataBlocks, shippeds, globalVarsBc, numTerms, params, groupContext,
      seed, sampIter)
    newDataBlocks.persist(storageLevel).setName(s"DataBlocks-$sampIter")
    if (needChkpt) {
      newDataBlocks.checkpoint()
      newDataBlocks.count()
    }

    val newParaBlocks = algo.updateParaBlocks(newDataBlocks, paraBlocks)
    newParaBlocks.persist(storageLevel).setName(s"ParaBlocks-$sampIter")
    if (needChkpt) {
      newParaBlocks.checkpoint()
    }
    newParaBlocks.count()

    dataBlocks.unpersist(blocking=false)
    paraBlocks.unpersist(blocking=false)
    dataBlocks = newDataBlocks
    paraBlocks = newParaBlocks

    if (needChkpt) {
      if (dataCpFile != null && paraCpFile != null) {
        SparkUtils.deleteChkptDirs(scConf, Array(dataCpFile, paraCpFile))
      }
      dataCpFile = newDataBlocks.getCheckpointFile.get
      paraCpFile = newParaBlocks.getCheckpointFile.get
    }
    val elapsedSeconds = (System.nanoTime - startedAt) / 1e9
    println(s"Sampling & grouping & updating paras $sampIter takes: $elapsedSeconds secs")
  }

  def saveDocModel(): Unit = {
    val saveAsSolid = scConf.get(cs_saveAsSolid).toBoolean
    val outputPath = scConf.get(cs_outputpath)
    val docSave = outputPath + ".docsave"
    val docSavePath = new Path(docSave)
    val fs = SparkUtils.getFileSystem(scConf, docSavePath)
    fs.delete(docSavePath, true)
    dataBlocks.mapPartitions(_.flatMap(_._2.DocRecs.iterator.map { case DocRec(docId, docGrp, docData) =>
      val docTopics = SparseVector.zeros[Int](numTopics)
      var p = 0
      while (p < docData.length) {
        var ind = docData(p)
        if (ind >= 0) {
          docTopics(docData(p + 1)) += 1
          p += 2
        } else {
          p += 2
          while (ind < 0) {
            docTopics(docData(p)) += 1
            p += 1
            ind += 1
          }
        }
      }
      val list = docTopics.activeIterator.toSeq.sortBy(-_._2).map(t => s"${t._1}:${t._2}").mkString("\t")
      s"$docId:${docGrp.value}\t$list"
    })).saveAsTextFile(docSave)
    if (saveAsSolid) {
      val cpmgPath = new Path(outputPath + ".cpmg")
      fs.delete(cpmgPath, true)
      var suc = FileUtil.copyMerge(fs, docSavePath, fs, cpmgPath, true, scContext.hadoopConfiguration, null)
      if (suc) {
        suc = fs.rename(cpmgPath, docSavePath)
      }
      if (!suc) {
        fs.delete(cpmgPath, true)
        throw new IOException("Save doc model error!")
      }
    }
  }

  def toGLDAModel: DistributedGLDAModel = {
    val termTopicsRDD = paraBlocks.mapPartitions(_.flatMap { case (_, ParaBlock(_, index, attrs)) =>
      val totalTermSize = attrs.length
      val results = new Array[Vector[Int]](totalTermSize)

      parallelized_foreachSplit(totalTermSize, (ts, tn, _) => {
        val decomp = new BVDecompressor(numTopics)
        for (ti <- ts until tn) {
          results(ti) = decomp.CV2BV(attrs(ti))
        }
      }, closing=true)(newParaExecutionContext(numThreads))

      index.iterator.map { case (termId, termIdx) => (termId, results(termIdx)) }
    }, preservesPartitioning=true)
    termTopicsRDD.persist(storageLevel)
    new DistributedGLDAModel(termTopicsRDD, numTopics, numGroups, numTerms, params, storageLevel)
  }
}

object GLDA {
  def apply(corpus: (RDD[(Int, DataBlock)], RDD[(Int, ParaBlock)]),
    numTopics: Int,
    numGroups: Int,
    numThreads: Int,
    params: HyperParams,
    storageLevel: StorageLevel): GLDA = {
    val (dataBlocks, paraBlocks) = corpus
    val numTerms = paraBlocks.map(_._2.index.keySet.max).max() + 1
    val activeTerms = paraBlocks.map(_._2.attrs.length).reduce(_ + _)
    println(s"terms in the corpus: $numTerms, $activeTerms of which are active")
    val numDocs = dataBlocks.map(_._2.DocRecs.length.toLong).reduce(_ + _)
    println(s"docs in the corpus: $numDocs")

    val numTokens = dataBlocks.mapPartitions(_.map { dbp =>
      val docRecs = dbp._2.DocRecs

      parallelized_reduceSplit[Long](docRecs.length, (ds, dn, _) => {
        var numTokensThrd = 0L
        for (di <- ds until dn) {
          val docData = docRecs(di).docData
          var p = 0
          while (p < docData.length) {
            val ind = docData(p)
            if (ind >= 0) {
              numTokensThrd += 1
              p += 2
            } else {
              numTokensThrd += -ind
              p += 2 - ind
            }
          }
        }
        numTokensThrd
      }, _ + _, closing=true)(newParaExecutionContext(numThreads))
    }).reduce(_ + _)
    println(s"tokens in the corpus: $numTokens")

    val algo = new GLDATrainer(numTopics, numThreads)
    val glda = new GLDA(dataBlocks, paraBlocks, numTopics, numGroups, numThreads, numTerms, numDocs, numTokens,
      params, algo, storageLevel)
    glda.init()
  }

  def initCorpus(rawDocsRDD: RDD[String],
    numTopics: Int,
    numGroups: Int,
    numThreads: Int,
    labelsRate: Float,
    storageLevel: StorageLevel): (RDD[(Int, DataBlock)], RDD[(Int, ParaBlock)]) = {
    val bowDocsRDD = GLDA.parseRawDocs(rawDocsRDD, numGroups, numThreads, labelsRate)
    initCorpus(bowDocsRDD, numTopics, numGroups, numThreads, storageLevel)
  }

  def initCorpus(bowDocsRDD: RDD[DocBow],
    numTopics: Int,
    numGroups: Int,
    numThreads: Int,
    storageLevel: StorageLevel): (RDD[(Int, DataBlock)], RDD[(Int, ParaBlock)]) = {
    val dataBlocks = GLDA.convertBowDocs(bowDocsRDD, numTopics, numGroups, numThreads)
    dataBlocks.persist(storageLevel).setName("DataBlocks-0")
    val paraBlocks = GLDA.buildParaBlocks(dataBlocks)
    (dataBlocks, paraBlocks)
  }

  def parseRawDocs(rawDocsRDD: RDD[String],
    numGroups: Int,
    numThreads: Int,
    labelsRate: Float): RDD[DocBow] = {
    rawDocsRDD.mapPartitions { iter =>
      val docs = iter.toArray
      val totalDocSize = docs.length
      val docBows = new Array[DocBow](totalDocSize)

      parallelized_foreachSplit(totalDocSize, (ds, dn, thid) => {
        val gen = new XORShiftRandom(System.nanoTime * numThreads + thid)
        for (di <- ds until dn) {
          val fields = docs(di).split(raw"\t|\s+").view
          val docInfo = fields.head.split(":")
          val docId = docInfo(0).toLong
          val docGrp = if (docInfo.length > 1 && gen.nextFloat() < labelsRate) {
            docInfo(1).toInt | 0x10000
          } else {
            numGroups
          }
          val docTerms = SparseVector.zeros[Int](Int.MaxValue)
          fields.tail.foreach { field =>
            val pair = field.split(":")
            val termId = pair(0).toInt
            var termCnt = if (pair.length > 1) pair(1).toInt else 1
            if (termCnt > 0) {
              docTerms(termId) += termCnt
            }
          }
          docBows(di) = DocBow(docId, docGrp, docTerms)
        }
      }, closing=true)(newParaExecutionContext(numThreads))

      docBows.iterator
    }
  }

  def convertBowDocs(bowDocsRDD: RDD[DocBow],
    numTopics: Int,
    numGroups: Int,
    numThreads: Int): RDD[(Int, DataBlock)] = {
    val numParts = bowDocsRDD.partitions.length
    bowDocsRDD.mapPartitionsWithIndex { (pid, iter) =>
      val docs = iter.toArray
      val totalDocSize = docs.length
      val docRecs = new Array[DocRec](totalDocSize)
      val termSet = new TrieMap[Int, Int]()

      val npg = numTopics / numGroups
      def randTopic(docGrp: Int, gen: Random): Int = {
        if (docGrp >= 0x10000 && gen.nextDouble() < 0.05) {
          val grp = docGrp & 0xFFFF
          val start = npg * grp
          val span = if (grp < numGroups - 1) npg else numTopics - start
          start + gen.nextInt(span)
        } else {
          gen.nextInt(numTopics)
        }
      }

      implicit val pec = newParaExecutionContext(numThreads)

      parallelized_foreachSplit(totalDocSize, (ds, dn, thid) => {
        val gen = new XORShiftRandom(System.nanoTime * numThreads + thid)
        for (di <- ds until dn) {
          val DocBow(docId, docGrp, docTerms) = docs(di)
          val docData = new mutable.ArrayBuffer[Int]()
          docTerms.activeIterator.foreach { case (termId, termCnt) =>
            if (termCnt == 1) {
              docData += termId
              docData += randTopic(docGrp, gen)
            } else if (termCnt > 1) {
              docData += -termCnt
              docData += termId
              var c = 0
              while (c < termCnt) {
                docData += randTopic(docGrp, gen)
                c += 1
              }
            }
            termSet.putIfAbsent(termId, 1)
          }
          docRecs(di) = DocRec(docId, IntWrapper(docGrp), docData.toArray)
        }
      })

      val localTerms = termSet.keys.toArray
      val numLocalTerms = localTerms.length
      val l2g = new Array[Int](numLocalTerms)
      val g2l = new mutable.HashMap[Int, Int]()
      val tqs = Array.fill(numLocalTerms)(new ConcurrentLinkedQueue[(Int, Int)]())
      for ((termId, localIdx) <- localTerms.iterator.zipWithIndex) {
        l2g(localIdx) = termId
        g2l(termId) = localIdx
      }

      parallelized_foreachSplit(totalDocSize, (ds, dn, _) => {
        for (di <- ds until dn) {
          val docData = docRecs(di).docData
          var p = 0
          while (p < docData.length) {
            val ind = docData(p)
            if (ind >= 0) {
              val termIdx = g2l(ind)
              docData(p) = termIdx
              tqs(termIdx).add((di, p))
              p += 2
            } else {
              val termIdx = g2l(docData(p + 1))
              docData(p + 1) = termIdx
              tqs(termIdx).add((di, p))
              p += 2 - ind
            }
          }
        }
      }, closing=true)

      val termRecs = new Array[TermRec](numLocalTerms)
      Range(0, numLocalTerms).par.foreach { i =>
        termRecs(i) = TermRec(l2g(i), tqs(i).asScala.flatMap(t => Iterator(t._1, t._2)).toArray)
      }
      Iterator.single((pid, DataBlock(termRecs, docRecs)))
    }.partitionBy(new HashPartitioner(numParts))
  }

  def buildParaBlocks(dataBlocks: RDD[(Int, DataBlock)]): RDD[(Int, ParaBlock)] = {
    val numParts = dataBlocks.partitions.length
    val routesRdd = dataBlocks.mapPartitions(_.flatMap { case (pid, db) =>
      db.termRecs.iterator.map(tr => (tr.termId, pid))
    }).partitionBy(new HashPartitioner(numParts))

    routesRdd.mapPartitionsWithIndex((pid, iter) => {
      val routes = Array.fill(numParts)(new mutable.ArrayBuffer[Int]())
      var cnt = 0
      val index = new mutable.HashMap[Int, Int]()
      iter.foreach { case (termId, termPid) =>
        routes(termPid) += termId
        index.getOrElseUpdate(termId, {
          cnt += 1
          cnt - 1
        })
      }
      val attrs = new Array[CompressedVector](cnt)
      Iterator.single((pid, ParaBlock(routes.map(_.toArray), index, attrs)))
    }, preservesPartitioning=true)
  }
}
