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

import java.util.concurrent.ConcurrentLinkedQueue

import breeze.linalg._
import com.github.cloudml.zen.ml.semiSupervised.GLDADefines._
import com.github.cloudml.zen.ml.util.Concurrent._
import com.github.cloudml.zen.ml.util._
import org.apache.hadoop.fs.Path
import org.apache.spark._
import org.apache.spark.broadcast.Broadcast
import org.apache.spark.rdd.RDD
import org.apache.spark.storage.StorageLevel

import scala.collection.JavaConverters._
import scala.collection.concurrent.TrieMap
import scala.collection.mutable
import scala.concurrent.Future


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
    this
  }

  def fit(totalIter: Int): Unit = {
    val evalMetrics = scConf.get(cs_evalMetric).split(raw"\+")
    val toEval = !evalMetrics.contains("none")
    val saveIntv = scConf.get(cs_saveInterval).toInt
    if (toEval) {
      println("Before Gibbs sampling:")
      GLDAMetrics(this, evalMetrics).foreach(_.output(println))
    }
    val burninIter = scConf.get(cs_burninIter).toInt
    val chkptIntv = scConf.get(cs_chkptInterval).toInt
    val canChkpt = chkptIntv > 0 && scContext.getCheckpointDir.isDefined
    for (iter <- 1 to totalIter) {
      println(s"\nStart Gibbs sampling (Iteration $iter/$totalIter)")
      val startedAt = System.nanoTime
      val needChkpt = canChkpt && iter % chkptIntv == 1

      val globalVars = algo.collectGlobalVariables(dataBlocks, params, numTerms)
      assert(sum(convert(globalVars.nK, Long)) == numTokens && sum(globalVars.dG) == numDocs)
      globalVarsBc = scContext.broadcast(globalVars)
      fitIteration(iter, burninIter, needChkpt)
      if (toEval) {
        GLDAMetrics(this, evalMetrics).foreach(_.output(println))
      }
      globalVarsBc.unpersist(blocking=false)

      if (saveIntv > 0 && iter % saveIntv == 0 && iter < totalIter) {
        val model = toGLDAModel
        val savPath = new Path(scConf.get(cs_outputpath) + s"-iter$iter")
        val fs = SparkUtils.getFileSystem(scConf, savPath)
        fs.delete(savPath, true)
        model.save(scContext, savPath.toString)
        println(s"Model saved after Iteration $iter")
      }
      val elapsedSeconds = (System.nanoTime - startedAt) / 1e9
      println(s"End Gibbs sampling (Iteration $iter/$totalIter) takes total: $elapsedSeconds secs")
    }
  }

  def fitIteration(sampIter: Int, burninIter: Int, needChkpt: Boolean): Unit = {
    val startedAt = System.nanoTime
    val shippeds = algo.ShipParaBlocks(dataBlocks, paraBlocks)
    val newDataBlocks = algo.SampleNGroup(dataBlocks, shippeds, globalVarsBc, params, seed, sampIter, burninIter)
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

  def toGLDAModel: DistributedGLDAModel = {
    val termTopicsRDD = paraBlocks.mapPartitions(_.flatMap { case (_, ParaBlock(routes, index, attrs)) =>
      val totalSize = attrs.length
      val results = new Array[Vector[Int]](totalSize)
      val sizePerthrd = {
        val npt = totalSize / numThreads
        if (npt * numThreads == totalSize) npt else npt + 1
      }
      implicit val es = initExecutionContext(numThreads)
      val allDecomp = Range(0, numThreads).map(thid => withFuture {
        val decomp = new BVDecompressor(numTopics)
        val posN = math.min(sizePerthrd * (thid + 1), totalSize)
        var pos = sizePerthrd * thid
        while (pos < posN) {
          results(pos) = decomp.CV2BV(attrs(pos))
          pos += 1
        }
      })
      withAwaitReadyAndClose(Future.sequence(allDecomp))

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
      val totalDocSize = docRecs.length
      val sizePerThrd = {
        val npt = totalDocSize / numThreads
        if (npt * numThreads == totalDocSize) npt else npt + 1
      }
      implicit val es = initExecutionContext(numThreads)
      val allToken = Range(0, numThreads).map(thid => withFuture {
        var numTokensThrd = 0L
        val posN = math.min(sizePerThrd * (thid + 1), totalDocSize)
        var pos = sizePerThrd * thid
        while (pos < posN) {
          val docData = docRecs(pos).docData
          var i = 0
          while (i < docData.length) {
            val ind = docData(i)
            if (ind >= 0) {
              numTokensThrd += 1
              i += 2
            } else {
              numTokensThrd += -ind
              i += 2 - ind
            }
          }
          pos += 1
        }
        numTokensThrd
      })
      withAwaitResultAndClose(Future.reduce(allToken)(_ + _))
    }).reduce(_ + _)
    println(s"tokens in the corpus: $numTokens")

    val algo = new GLDATrainer(numTopics, numGroups, numThreads)
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
    initCorpus(bowDocsRDD, numTopics, numThreads, storageLevel)
  }

  def initCorpus(bowDocsRDD: RDD[DocBow],
    numTopics: Int,
    numThreads: Int,
    storageLevel: StorageLevel): (RDD[(Int, DataBlock)], RDD[(Int, ParaBlock)]) = {
    val dataBlocks = GLDA.convertBowDocs(bowDocsRDD, numTopics, numThreads)
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
      val sizePerThrd = {
        val npt = totalDocSize / numThreads
        if (npt * numThreads == totalDocSize) npt else npt + 1
      }
      implicit val es = initExecutionContext(numThreads)
      val allParsing = Range(0, numThreads).map(thid => withFuture {
        val gen = new XORShiftRandom(System.nanoTime * numThreads + thid)
        val posN = math.min(sizePerThrd * (thid + 1), totalDocSize)
        var pos = sizePerThrd * thid
        while (pos < posN) {
          val line = docs(pos)
          val fields = line.split(raw"\t|\s+").view
          val docInfo = fields.head.split(":")
          val docId = docInfo(0).toLong
          val docGrp = if (docInfo.length > 1 && gen.nextFloat() < labelsRate) {
            docInfo(1).toInt | 0x10000
          } else {
            gen.nextInt(numGroups)
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
          docBows(pos) = DocBow(docId, docGrp, docTerms)
          pos += 1
        }
      })
      withAwaitReadyAndClose(Future.sequence(allParsing))

      docBows.iterator
    }
  }

  def convertBowDocs(bowDocsRDD: RDD[DocBow],
    numTopics: Int,
    numThreads: Int): RDD[(Int, DataBlock)] = {
    val numParts = bowDocsRDD.partitions.length
    bowDocsRDD.mapPartitionsWithIndex { (pid, iter) =>
      val docs = iter.toArray
      val totalDocSize = docs.length

      val docRecs = new Array[DocRec](totalDocSize)
      val termSet = new TrieMap[Int, Null]()
      val sizePerThrd = {
        val npt = totalDocSize / numThreads
        if (npt * numThreads == totalDocSize) npt else npt + 1
      }
      implicit val es = initExecutionContext(numThreads)
      val allConv = Range(0, numThreads).map(thid => withFuture {
        val gen = new XORShiftRandom(System.nanoTime * numThreads + thid)
        val posN = math.min(sizePerThrd * (thid + 1), totalDocSize)
        var pos = sizePerThrd * thid
        while (pos < posN) {
          val DocBow(docId, docGrp, docTerms) = docs(pos)
          val docData = new mutable.ArrayBuffer[Int]()
          docTerms.activeIterator.foreach { case (termId, termCnt) =>
            if (termCnt == 1) {
              docData += termId
              docData += gen.nextInt(numTopics)
            } else if (termCnt > 1) {
              docData += -termCnt
              docData += termId
              var c = 0
              while (c < termCnt) {
                docData += gen.nextInt(numTopics)
                c += 1
              }
            }
            termSet.putIfAbsent(termId, null)
          }
          docRecs(pos) = DocRec(docId, IntWrapper(docGrp), docData.toArray)
          pos += 1
        }
      })
      withAwaitReady(Future.sequence(allConv))

      val localTerms = termSet.keys.toArray
      val numLocalTerms = localTerms.length
      val l2g = new Array[Int](numLocalTerms)
      val g2l = new mutable.HashMap[Int, Int]()
      val tqs = Array.fill(numLocalTerms)(new ConcurrentLinkedQueue[(Int, Int)]())
      for ((termId, localIdx) <- localTerms.iterator.zipWithIndex) {
        l2g(localIdx) = termId
        g2l(termId) = localIdx
      }
      val allInvIdx = Range(0, numThreads).map(thid => withFuture {
        val posN = math.min(sizePerThrd * (thid + 1), totalDocSize)
        var pos = sizePerThrd * thid
        while (pos < posN) {
          val docData = docRecs(pos).docData
          var i = 0
          while (i < docData.length) {
            val ind = docData(i)
            if (ind >= 0) {
              val termIdx = g2l(ind)
              docData(i) = termIdx
              tqs(termIdx).add((pos, i))
              i += 2
            } else {
              val termIdx = g2l(docData(i + 1))
              docData(i + 1) = termIdx
              tqs(termIdx).add((pos, i))
              i += 2 - ind
            }
          }
          pos += 1
        }
      })
      withAwaitReadyAndClose(Future.sequence(allInvIdx))

      val termRecs = new Array[TermRec](numLocalTerms)
      Range(0, numLocalTerms).par.foreach { li =>
        termRecs(li) = TermRec(l2g(li), tqs(li).asScala.flatMap(t => Iterator(t._1, t._2)).toArray)
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
